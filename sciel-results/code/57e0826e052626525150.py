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

def _build_linear_system_U(n, m, p, R, V, W):
    """Build linear system to solve for U given V and W using Kronecker products."""
    # For each (i, j, jp, k, kp, ip):
    # sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r] = target
    # 
    # Group by (jp, k, kp, ip): for fixed (jp,k,kp,ip), we have n*m equations in n*m*R unknowns
    # Actually, let's think of it differently.
    # 
    # The equation is: for all (i,j,jp,k,kp,ip):
    # sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r] = delta_{i,ip} * delta_{j,jp} * delta_{k,kp}
    #
    # For fixed (jp, k, kp, ip), define b_r = V[jp*p+k, r] * W[kp*n+ip, r]
    # Then: sum_r U[i*m+j, r] * b_r = target[i,j,jp,k,kp,ip]
    #
    # This is a linear system in U. Let's build it block by block.
    # 
    # Actually, let's build the full system more carefully.
    # Unknowns: U[i*m+j, r] for i in [0,n), j in [0,m), r in [0,R) -> n*m*R unknowns
    # Equations: for all (i,j,jp,k,kp,ip) -> n*m*m*p*p*n equations
    #
    # For equation (i,j,jp,k,kp,ip), the coefficient of U[i'*m+j', r] is:
    # V[jp*p+k, r] * W[kp*n+ip, r] if i'=i and j'=j, else 0
    #
    # So the system is block-diagonal in (i,j)! For each (i,j), we have:
    # sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r] = delta_{j,jp} * delta_{k,kp} * delta_{i,ip}
    # for all (jp, k, kp, ip)
    #
    # This gives m*p*p*n equations per (i,j) pair, with R unknowns.
    
    num_unknowns = n * m * R
    num_eqs = n * m * m * p * p * n
    
    # Build using Kronecker products for efficiency
    # For each (i,j), the sub-system is:
    # For all (jp, k, kp, ip): sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r] = rhs
    #
    # Let's define for each (i,j): a matrix A_{ij} of size (m*p*p*n) x R
    # where A_{ij}[idx, r] = V[jp*p+k, r] * W[kp*n+ip, r] for the corresponding (jp,k,kp,ip)
    
    # This is equivalent to: for each r, the column is the outer product of V_col and W_col
    # where V_col = V.reshape(m, p, R)[:, :, r].flatten() and W_col = W.reshape(p, n, R)[:, :, r].flatten()
    
    # Let's build it efficiently
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    
    # For each r, the coefficient vector for a given (i,j) is:
    # vec(V[:, :, r]) * vec(W[:, :, r]) element-wise? No...
    # Actually for fixed (i,j), the equation index (jp, k, kp, ip) maps to:
    # V[jp, k, r] * W[kp, ip, r]
    #
    # So for fixed (i,j) and r, the coefficient is V[jp,k,r] * W[kp,ip,r]
    # This is the outer product of V[:,:,r].flatten() and W[:,:,r].flatten()
    
    # Build the full system
    A_mat = np.zeros((num_eqs, num_unknowns))
    b_vec = np.zeros(num_eqs)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            target_val = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            b_vec[idx] = target_val
                            
                            for r in range(R):
                                coeff = V[jp * p + k, r] * W[kp * n + ip, r]
                                A_mat[idx, (i * m + j) * R + r] = coeff
                            
                            idx += 1
    
    return A_mat, b_vec

def _solve_rank_with_polish(n, m, p, R, time_limit=10.0):
    """Try to find a decomposition of rank R using least_squares with polish."""
    start_time = time.time()
    num_vars = R * (n * m + m * p + p * n)
    
    # Precompute target tensor
    I, J, JP, K, KP, IP = np.meshgrid(
        range(n), range(m), range(m), range(p), range(p), range(n),
        indexing='ij'
    )
    target = ((I == IP) & (J == JP) & (K == KP)).astype(np.float64).flatten()
    
    def residual(params):
        U = params[:R * n * m].reshape(n, m, R)
        V = params[R * n * m : R * (n * m + m * p)].reshape(m, p, R)
        W = params[R * (n * m + m * p):].reshape(p, n, R)
        
        T = np.einsum('ijr,jkr,pir->ijjkpi', U, V, W, optimize=True)
        
        return (T - target).flatten()
    
    best_result = None
    best_residual_norm = np.inf
    
    # Try multiple random initializations
    num_attempts = 8
    for attempt in range(num_attempts):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization with small values
        params = np.random.randn(num_vars) * 0.5
        
        try:
            result = least_squares(residual, params, method='lm', max_nfev=2000, xtol=1e-12)
            res_norm = np.linalg.norm(result.fun)
            
            if res_norm < best_residual_norm:
                best_residual_norm = res_norm
                best_result = result
            
            if res_norm < 1e-8:
                U = result.x[:R * n * m].reshape(n * m, R)
                V = result.x[R * n * m : R * (n * m + m * p)].reshape(m * p, R)
                W = result.x[R * (n * m + m * p):].reshape(p * n, R)
                
                # Verify
                if _verify_fast(U, V, W, n, m, p):
                    return U, V, W
                
                # Try snapping to half-integers
                U_s, V_s, W_s = _snap_to_half_integers(U, V, W)
                if _verify_fast(U_s, V_s, W_s, n, m, p):
                    return U_s, V_s, W_s
                    
        except Exception:
            pass
    
    # If we have a near-solution, try polishing
    if best_result is not None and best_residual_norm < 1e-4:
        U = best_result.x[:R * n * m].reshape(n * m, R)
        V = best_result.x[R * n * m : R * (n * m + m * p)].reshape(m * p, R)
        W = best_result.x[R * (n * m + m * p):].reshape(p * n, R)
        
        # Snap and verify
        U_s, V_s, W_s = _snap_to_half_integers(U, V, W)
        if _verify_fast(U_s, V_s, W_s, n, m, p):
            return U_s, V_s, W_s
    
    return None

def _solve_rank_alternating(n, m, p, R, time_limit=10.0):
    """
    Alternating least squares: fix two factors, solve for the third.
    This exploits the linear structure of the tensor equation.
    """
    start_time = time.time()
    
    # Precompute target tensor
    I, J, JP, K, KP, IP = np.meshgrid(
        range(n), range(m), range(m), range(p), range(p), range(n),
        indexing='ij'
    )
    target = ((I == IP) & (J == JP) & (K == KP)).astype(np.float64)
    
    # Initialize with random values
    U = np.random.randn(n * m, R) * 0.5
    V = np.random.randn(m * p, R) * 0.5
    W = np.random.randn(p * n, R) * 0.5
    
    # Precompute some useful matrices
    # For solving U: given V, W, the system is block-diagonal in (i,j)
    # For each (i,j): sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r] = target[i,j,jp,k,kp,ip]
    
    # Let's precompute the Gram matrices for each factor
    # For U update: for each (i,j), A_{ij} is (m*p*p*n) x R
    # A_{ij}[idx, r] = V[jp*p+k, r] * W[kp*n+ip, r]
    
    # This can be written as: A_{ij} = (V_flat ⊗ W_flat) but with specific structure
    # Actually, A_{ij} has entries A_{ij}[a, r] where a = (jp, k, kp, ip)
    # A_{ij}[a, r] = V[jp, k, r] * W[kp, ip, r]
    #
    # Let V2 = V.reshape(m, p, R) and W2 = W.reshape(p, n, R)
    # Then A_{ij}[:, r] = V2[:, :, r].flatten() * W2[:, :, r].flatten() (element-wise product)
    # Wait, that's not quite right. V2[jp, k, r] * W2[kp, ip, r]
    # If we flatten V2[:,:,r] to length m*p and W2[:,:,r] to length p*n,
    # then the product is not a simple outer product.
    #
    # Let me just build it directly but efficiently.
    
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    
    # Precompute for U update: for each r, the coefficient vector
    # coeff_U[r] = V3[:,:,r].flatten() * W3[:,:,r].flatten() but with proper indexing
    # Actually, for fixed r, the coefficient of U[i*m+j, r] in equation (i,j,jp,k,kp,ip) is:
    # V[jp,k,r] * W[kp,ip,r]
    #
    # Let's define for each r: C_U[r] = np.zeros(m*p*p*n)
    # C_U[r][(jp, k, kp, ip)] = V[jp,k,r] * W[kp,ip,r]
    
    # This is the element-wise product of V3[:,:,r].reshape(m*p) and W3[:,:,r].reshape(p*n)
    # but with the right indexing. Let me just build it.
    
    def solve_U_given_VW(V, W):
        """Solve for U given V and W."""
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        
        U_new = np.zeros((n * m, R))
        
        for i in range(n):
            for j in range(m):
                # Build the sub-system for U[i*m+j, :]
                # Equations: for all (jp, k, kp, ip):
                # sum_r U[i*m+j, r] * V[jp,k,r] * W[kp,ip,r] = target[i,j,jp,k,kp,ip]
                
                num_eqs_sub = m * p * p * n
                A_sub = np.zeros((num_eqs_sub, R))
                b_sub = np.zeros(num_eqs_sub)
                
                idx = 0
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                b_sub[idx] = 1.0 if (j == jp and k == kp and i == ip) else 0.0
                                for r in range(R):
                                    A_sub[idx, r] = V[jp * p + k, r] * W[kp * n + ip, r]
                                idx += 1
                
                # Solve the least squares problem
                try:
                    sol, residuals, rank_sv, sv = np.linalg.lstsq(A_sub, b_sub, rcond=None)
                    U_new[i * m + j, :] = sol
                except:
                    pass
        
        return U_new
    
    def solve_V_given_UW(U, W):
        """Solve for V given U and W."""
        U3 = U.reshape(n, m, R)
        W3 = W.reshape(p, n, R)
        
        V_new = np.zeros((m * p, R))
        
        for jp in range(m):
            for k in range(p):
                # Build the sub-system for V[jp*p+k, :]
                # Equations: for all (i, j, kp, ip):
                # sum_r U[i,j,r] * V[jp,k,r] * W[kp,ip,r] = target[i,j,jp,k,kp,ip]
                
                num_eqs_sub = n * m * p * n
                A_sub = np.zeros((num_eqs_sub, R))
                b_sub = np.zeros(num_eqs_sub)
                
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for kp in range(p):
                            for ip in range(n):
                                b_sub[idx] = 1.0 if (j == jp and k == kp and i == ip) else 0.0
                                for r in range(R):
                                    A_sub[idx, r] = U[i * m + j, r] * W[kp * n + ip, r]
                                idx += 1
                
                try:
                    sol, residuals, rank_sv, sv = np.linalg.lstsq(A_sub, b_sub, rcond=None)
                    V_new[jp * p + k, :] = sol
                except:
                    pass
        
        return V_new
    
    def solve_W_given_UV(U, V):
        """Solve for W given U and V."""
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        
        W_new = np.zeros((p * n, R))
        
        for kp in range(p):
            for ip in range(n):
                # Build the sub-system for W[kp*n+ip, :]
                # Equations: for all (i, j, jp, k):
                # sum_r U[i,j,r] * V[jp,k,r] * W[kp,ip,r] = target[i,j,jp,k,kp,ip]
                
                num_eqs_sub = n * m * m * p
                A_sub = np.zeros((num_eqs_sub, R))
                b_sub = np.zeros(num_eqs_sub)
                
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for k in range(p):
                                b_sub[idx] = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                for r in range(R):
                                    A_sub[idx, r] = U[i * m + j, r] * V[jp * p + k, r]
                                idx += 1
                
                try:
                    sol, residuals, rank_sv, sv = np.linalg.lstsq(A_sub, b_sub, rcond=None)
                    W_new[kp * n + ip, :] = sol
                except:
                    pass
        
        return W_new
    
    # Alternating minimization
    for iteration in range(50):
        if time.time() - start_time > time_limit:
            break
        
        U = solve_U_given_VW(V, W)
        V = solve_V_given_UW(U, W)
        W = solve_W_given_UV(U, V)
        
        # Check convergence
        if _verify_fast(U, V, W, n, m, p):
            return U, V, W
        
        # Try snapping
        U_s, V_s, W_s = _snap_to_half_integers(U, V, W)
        if _verify_fast(U_s, V_s, W_s, n, m, p):
            return U_s, V_s, W_s
    
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
    # Try ranks from 47 down to 24
    for target_rank in range(47, 23, -1):
        if time.time() > DEADLINE - 15:
            break
        
        # Try alternating least squares first (faster)
        if time.time() < DEADLINE - 20:
            result = _solve_rank_alternating(n, m, p, target_rank, time_limit=8.0)
            if result is not None:
                U, V, W = result
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
        
        # Try least_squares with polish
        if time.time() < DEADLINE - 25:
            result = _solve_rank_with_polish(n, m, p, target_rank, time_limit=8.0)
            if result is not None:
                U, V, W = result
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
