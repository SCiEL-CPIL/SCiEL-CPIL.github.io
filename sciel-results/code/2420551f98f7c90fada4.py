import time
import random
import numpy as np
from scipy.optimize import least_squares, lsq_linear

DIMENSIONS = [2, 4, 6]

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    R = U.shape[1]
    
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    
    T = np.einsum('ijr,jkr,pir->ijjkpi', U3, V3, W3, optimize=True)
    
    I, J, JP, K, KP, IP = np.meshgrid(
        range(n), range(m), range(m), range(p), range(p), range(n),
        indexing='ij'
    )
    target = ((I == IP) & (J == JP) & (K == KP)).astype(np.float64)
    
    return np.allclose(T, target, atol=1e-9, rtol=1e-9)

def _snap_to_half_integers(U, V, W):
    """Snap arrays to nearest half-integers and verify."""
    U_s = (2 * U).round() / 2
    V_s = (2 * V).round() / 2
    W_s = (2 * W).round() / 2
    return U_s, V_s, W_s

def _solve_rank(n, m, p, R, time_limit=10.0):
    """Try to find a decomposition of rank R using least_squares."""
    start_time = time.time()
    num_vars = R * (n * m + m * p + p * n)
    
    # Precompute target tensor
    I, J, JP, K, KP, IP = np.meshgrid(
        range(n), range(m), range(m), range(p), range(p), range(n),
        indexing='ij'
    )
    target = ((I == IP) & (J == JP) & (K == KP)).astype(np.float64).flatten()
    
    # Define the residual function
    def residual(params):
        U = params[:R * n * m].reshape(n, m, R)
        V = params[R * n * m : R * (n * m + m * p)].reshape(m, p, R)
        W = params[R * (n * m + m * p):].reshape(p, n, R)
        
        T = np.einsum('ijr,jkr,pir->ijjkpi', U, V, W, optimize=True)
        
        return (T - target).flatten()

    # Try a few random initializations
    for attempt in range(5):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization with small integers
        params = np.random.randint(-2, 3, size=num_vars)
        
        try:
            result = least_squares(residual, params, method='lm', max_nfev=1000)
            if result.success and np.allclose(result.fun, 0, atol=1e-8):
                U = result.x[:R * n * m].reshape(n * m, R)
                V = result.x[R * n * m : R * (n * m + m * p)].reshape(m * p, R)
                W = result.x[R * (n * m + m * p):].reshape(p * n, R)
                # Verify
                if _verify_fast(U, V, W, n, m, p):
                    return U, V, W
        except Exception:
            pass
            
    return None

def _solve_rank_lsq_linear(n, m, p, R, time_limit=10.0):
    """
    Try to find a decomposition of rank R using lsq_linear on tensor unfolding.
    
    Strategy: Fix U and W, solve for V (or fix V and W, solve for U, etc.)
    using the linear structure of the tensor equation.
    """
    start_time = time.time()
    
    # Precompute target tensor
    I, J, JP, K, KP, IP = np.meshgrid(
        range(n), range(m), range(m), range(p), range(p), range(n),
        indexing='ij'
    )
    target = ((I == IP) & (J == JP) & (K == KP)).astype(np.float64)
    
    for attempt in range(10):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization for U and W
        U = np.random.randint(-1, 2, size=(n * m, R)).astype(np.float64)
        W = np.random.randint(-1, 2, size=(p * n, R)).astype(np.float64)
        
        # For each rank component r, we have:
        # target[i,j,j',k,k',i'] = sum_r U[i,j,r] * V[j,k,r] * W[k',i',r]
        # 
        # Rearranging: for fixed i,j,k',i', 
        # target[i,j,j',k,k',i'] = sum_r U[i,j,r] * W[k',i',r] * V[j,k,r]
        #
        # This is linear in V[j,k,r] for fixed U and W.
        #
        # Let's vectorize: for each (j, k) pair in V, we have R unknowns V[j,k,0..R-1]
        # The equation for each (i, j', k', i') where j' = j and k' = k is:
        # target[i,j,j,k,k',i'] = sum_r U[i,j,r] * W[k',i',r] * V[j,k,r]
        
        # Actually, let's think of it differently. The tensor is:
        # T[i,j,j',k,k',i'] = sum_r U[i,j,r] * V[j,k,r] * W[k',i',r]
        # with constraints j=j', k=k', i=i' for the target.
        #
        # For the target, T[i,j,j,k,k,i] = 1 if (j=j, k=k, i=i) which is always true,
        # so target[i,j,j,k,k,i] = 1 for all valid i,j,k.
        #
        # Actually the target is: T[i,j,j',k,k',i'] = 1 if i=i' and j=j' and k=k'
        # So target[i,j,j,k,k,i] = 1, and target[i,j,j',k,k',i'] = 0 otherwise.
        
        # Let's build a linear system for V given U and W.
        # For each triple (i, j, k) where i in [0,n), j in [0,m), k in [0,p):
        # The (i,j,j,k,k,i) component of T is:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        #
        # This gives n*m*p equations, each linear in the R entries of V[j*p+k, :].
        # But we also need to satisfy the "off-diagonal" constraints:
        # For (i,j,j',k,k',i') with not (i=i' and j=j' and k=k'):
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k'*n+i', r] = 0
        
        # This is getting complex. Let's use a different approach.
        # 
        # Alternative: Use the fact that the tensor multiplication can be written as:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # In tensor form: T[i,j,j',k,k',i'] A[i,j] B[j',k] = delta_{j,j'} delta_{k,k'} C[i,k]
        #
        # For our decomposition:
        # sum_r U[i,j,r] V[j',k,r] W[k',i',r] A[i,j] B[j',k] = delta_{j,j'} delta_{k,k'} C[i,k]
        #
        # This must hold for all A, B. So:
        # sum_r U[i,j,r] V[j',k,r] W[k',i',r] = delta_{j,j'} delta_{k,k'} delta_{i,i'}
        # (where the right side is 1 when i=i', j=j', k=k', else 0)
        
        # Let's build the linear system properly.
        # Unknowns: V[j*p+k, r] for j in [0,m), k in [0,p), r in [0,R)
        # Total unknowns: m * p * R
        
        # Equations: For each (i, j, j', k, k', i') in [0,n)x[0,m)x[0,m)x[0,p)x[0,p)x[0,n):
        # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = target[i,j,j',k,k',i']
        
        # This is a linear system in V. Let's build A_matrix and b_vector.
        
        num_unknowns = m * p * R
        num_eqs = n * m * m * p * p * n
        
        # Build the system
        A_mat = np.zeros((num_eqs, num_unknowns))
        b_vec = np.zeros(num_eqs)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                # Equation: sum_r U[i,j,r] * V[jp,k,r] * W[kp,ip,r] = target
                                target_val = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                b_vec[idx] = target_val
                                
                                for r in range(R):
                                    # Coefficient of V[jp*p+k, r]
                                    coeff = U[i * m + j, r] * W[kp * n + ip, r]
                                    A_mat[idx, jp * p * R + k * R + r] = coeff
                                
                                idx += 1
        
        # Solve the linear system
        try:
            result = lsq_linear(A_mat, b_vec, method='bvls', max_iter=1000)
            V_flat = result.x
            V = V_flat.reshape(m * p, R)
            
            # Verify
            if _verify_fast(U, V, W, n, m, p):
                return U, V, W
            
            # Try snapping to half-integers
            U_s, V_s, W_s = _snap_to_half_integers(U, V, W)
            if _verify_fast(U_s, V_s, W_s, n, m, p):
                return U_s, V_s, W_s
        except Exception:
            pass
        
    return None

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    n, m, p = DIMENSIONS
    
    # Schoolbook solution
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try to find lower rank
    # Lower bound is max(nm, mp, pn) = 24
    # We try ranks from 47 down to 40 (limited search)
    for target_rank in range(47, 39, -1):
        if time.time() > DEADLINE - 10:
            break
        
        # Try least_squares approach
        result = _solve_rank(n, m, p, target_rank, time_limit=8.0)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
            break # Found a solution, return it
        
        # Try lsq_linear approach if time permits
        if time.time() < DEADLINE - 15:
            result = _solve_rank_lsq_linear(n, m, p, target_rank, time_limit=8.0)
            if result is not None:
                U, V, W = result
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break # Found a solution, return it
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
