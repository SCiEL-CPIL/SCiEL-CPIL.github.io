"""Rewritten program for 3x3 * 3x7 matrix multiplication tensor decomposition."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    Returns: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
    # Schoolbook decomposition as baseline (rank 63)
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    best_result = {'U': U_sb, 'V': V_sb, 'W': W_sb}
    best_rank = schoolbook_rank

    def verify(U, V, W, tol=1e-8):
        """Verify exact decomposition using numpy."""
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if len(U) == 0 or len(V) == 0 or len(W) == 0:
            return False
        r = len(U[0])
        if len(V[0]) != r or len(W[0]) != r or r <= 0:
            return False
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def clean(x):
        """Snap to nearest half-integer if close."""
        h = round(x * 2) / 2.0
        if abs(x - h) < 1e-6:
            return h
        return x

    def make_result(U, V, W):
        Uc = [[clean(x) for x in row] for row in U]
        Vc = [[clean(x) for x in row] for row in V]
        Wc = [[clean(x) for x in row] for row in W]
        return {'U': Uc, 'V': Vc, 'W': Wc}

    def try_update(U, V, W):
        nonlocal best_result, best_rank
        if verify(U, V, W):
            r = len(U[0])
            if r < best_rank:
                best_result = make_result(U, V, W)
                best_rank = r
                return True
        return False

    # Strategy: Try to find lower-rank decompositions using optimization
    # The tensor T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is equivalent to: for each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # sums to the all-ones tensor.
    
    # Approach: Start with a random initialization and use gradient descent
    # to minimize the sum of squared errors.
    
    def tensor_residual(U, V, W):
        """Compute the residual tensor and its norm."""
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        r = Ua.shape[1]
        residual = np.zeros((n, m, p))
        for rr in range(r):
            # U[:, rr] is (n*m,), V[:, rr] is (m*p,), W[:, rr] is (p*n,)
            # Reshape to (n, m), (m, p), (p, n)
            U_rm = Ua[:, rr].reshape((n, m))
            V_rm = Va[:, rr].reshape((m, p))
            W_rm = Wa[:, rr].reshape((p, n))
            # The contribution to C[i,k] is sum_j U_rm[i,j] * V_rm[j,k] * W_rm[k,i]
            # This is a bit tricky. Let's compute it differently.
            # For each i, k: sum_j U_rm[i,j] * V_rm[j,k] * W_rm[k,i]
            # = sum_j U_rm[i,j] * V_rm[j,k] * W_rm[k,i]
            # Let's compute M = U_rm @ V_rm which is (n, p)
            # Then contribution to C[i,k] is M[i,k] * W_rm[k,i]
            M = U_rm @ V_rm
            # Now C[i,k] += M[i,k] * W_rm[k,i]
            # W_rm is (p, n), so W_rm[k,i] is element (k,i)
            # We need to multiply M[i,k] by W_rm[k,i] for each i,k
            # This is element-wise product with a permuted W
            # W_rm.T is (n, p), so W_rm.T[i,k] = W_rm[k,i]
            residual += M * W_rm.T
        return residual

    def objective(params, r):
        """Objective function for optimization."""
        if time.time() > deadline - 5:
            return np.inf
        U_flat, V_flat, W_flat = params[:n*m*r], params[n*m*r:n*m*r+m*p*r], params[n*m*r+m*p*r:]
        U = U_flat.reshape((n*m, r))
        V = V_flat.reshape((m*p, r))
        W = W_flat.reshape((p*n, r))
        res = tensor_residual(U, V, W)
        return np.sum(res**2)

    def gradient(params, r):
        """Compute gradient using finite differences (slow but simple)."""
        if time.time() > deadline - 5:
            return np.zeros_like(params)
        eps = 1e-7
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            f_plus = objective(params_plus, r)
            params_minus = params.copy()
            params_minus[i] -= eps
            f_minus = objective(params_minus, r)
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad

    # Try different ranks from low to high
    for target_rank in [20, 21, 22, 23, 24, 25]:
        if time.time() > deadline - 10:
            break
        
        r = target_rank
        # Random initialization
        np.random.seed(42)
        U_init = np.random.randn(n*m, r) * 0.1
        V_init = np.random.randn(m*p, r) * 0.1
        W_init = np.random.randn(p*n, r) * 0.1
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Use Nelder-Mead or L-BFGS-B
        try:
            result = minimize(
                objective,
                params_init,
                args=(r,),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < 1e-6:
                U_opt, V_opt, W_opt = result.x[:n*m*r], result.x[n*m*r:n*m*r+m*p*r], result.x[n*m*r+m*p*r:]
                U_opt = U_opt.reshape((n*m, r))
                V_opt = V_opt.reshape((m*p, r))
                W_opt = W_opt.reshape((p*n, r))
                
                if try_update(U_opt.tolist(), V_opt.tolist(), W_opt.tolist()):
                    print(f"Found rank {r} decomposition!")
                    break
        except Exception as e:
            pass
    
    # Also try a structured approach: block decomposition
    # Split the 7 columns into blocks and use Strassen-like schemes
    
    # For 3x3 * 3x3, we can try to find a rank-7 decomposition
    def build_rank7_3x3():
        """Try to construct a rank-7 decomposition for 3x3 * 3x3."""
        r = 7
        U = np.zeros((9, r))
        V = np.zeros((9, r))
        W = np.zeros((9, r))
        
        # Use the known 7-multiplication scheme for 2x2 matrices extended
        # Actually, let's use a different approach: start with schoolbook and try to merge terms
        
        # Initialize with schoolbook
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    col = i*3 + j*3 + k  # This doesn't work for 7 terms
        
        # Let's try a specific construction
        # Term 0: (A00+A22)(B00+B22)
        U[0,0] = 1; U[8,0] = 1
        V[0,0] = 1; V[8,0] = 1
        W[0,0] = 1; W[6,0] = 1; W[2,0] = 1; W[8,0] = 1
        
        # Term 1: (A20+A22)B02
        U[6,1] = 1; U[8,1] = 1
        V[2,1] = 1
        W[6,1] = 1; W[8,1] = 1
        
        # Term 2: A00(B02-B22)
        U[0,2] = 1
        V[2,2] = 1; V[8,2] = -1
        W[0,2] = 1; W[2,2] = 1
        
        # Term 3: A22(B20-B00)
        U[8,3] = 1
        V[6,3] = 1; V[0,3] = -1
        W[6,3] = 1; W[8,3] = 1
        
        # Term 4: (A00+A02)B22
        U[0,4] = 1; U[2,4] = 1
        V[8,4] = 1
        W[0,4] = 1; W[2,4] = 1
        
        # Term 5: (A20-A00)(B00+B02)
        U[6,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[2,5] = 1
        W[6,5] = 1; W[8,5] = 1
        
        # Term 6: (A02+A22)(B20+B22)
        U[2,6] = 1; U[8,6] = 1
        V[6,6] = 1; V[8,6] = 1
        W[0,6] = 1; W[2,6] = 1
        
        return U, V, W
    
    U7, V7, W7 = build_rank7_3x3()
    if verify(U7.tolist(), V7.tolist(), W7.tolist()):
        print("Valid rank-7 3x3 decomposition found!")
        # Now use this for 3x3 * 3x7 by splitting into 3+3+1
        # First two 3x3 blocks use rank 7 each, last 3x1 uses rank 9
        total_rank = 7 + 7 + 9
        U_full = np.zeros((9, total_rank))
        V_full = np.zeros((21, total_rank))
        W_full = np.zeros((21, total_rank))
        
        # Copy first 3x3 block
        U_full[:, 0:7] = U7
        V_full[0:9, 0:7] = V7
        W_full[0:9, 0:7] = W7
        
        # Copy second 3x3 block (columns 3-5 of B)
        U_full[:, 7:14] = U7
        V_full[9:18, 7:14] = V7
        W_full[9:18, 7:14] = W7
        
        # Last 3x1 block (column 6 of B) - schoolbook rank 9
        for i in range(3):
            for j in range(3):
                col = 14 + i*3 + j
                U_full[i*3+j, col] = 1
                V_full[j*3+6, col] = 1
                W_full[6*3+i, col] = 1  # k=6, so k*n+i = 6*3+i = 18+i
        
        if try_update(U_full.tolist(), V_full.tolist(), W_full.tolist()):
            print(f"Found rank {total_rank} decomposition via blocks!")
            break
    
    # Final check and return
    if time.time() > deadline - 2:
        pass
        
    return best_result

# EVOLVE-BLOCK-END
