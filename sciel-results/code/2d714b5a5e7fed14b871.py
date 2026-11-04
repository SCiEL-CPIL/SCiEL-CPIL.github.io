import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 4]
START_TIME = time.time()
DEADLINE = START_TIME + 110.0

def _time_left():
    return DEADLINE - time.time()

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, tol=1e-7):
    n, m, p = DIMENSIONS
    r = U.shape[1]
    # Check half-integer constraint
    for mat in [U, V, W]:
        for val in mat:
            if abs(val * 2 - round(val * 2)) > tol:
                return False
    # Check tensor equation
    # Sum_r W[k*n+i, r] * (Sum_ij U[i*m+j, r] A[i,j]) * (Sum_jk V[j*p+k, r] B[j,k]) == Sum_j A[i,j] B[j,k]
    # We check this by comparing coefficients of A_ij B_jk'
    # For each i, k:
    # LHS coeff of A_ij B_jk is Sum_r W[k*n+i, r] U[i*m+j, r] V[j*p+k, r]
    # This must be 1 if j is the middle index, else 0?
    # Wait, the term A_ij B_jk contributes to C_ik.
    # So for fixed i, k, and any j, the coeff of A_ij B_jk in C_ik is 1.
    # For any other combination A_i'j' B_j''k'' where j' != j'', the coeff in C_ik is 0.
    
    # Let's just use random A, B to verify.
    for _ in range(20):
        A = np.random.randint(0, 3, (n, m)).astype(float)
        B = np.random.randint(0, 3, (m, p)).astype(float)
        C_true = A @ B
        C_calc = np.zeros((n, p))
        for c in range(r):
            LA = np.sum(U[:, c].reshape(n, m) * A)
            LB = np.sum(V[:, c].reshape(m, p) * B)
            Wcol = W[:, c].reshape(p, n).T
            C_calc += LA * LB * Wcol
        if not np.allclose(C_calc, C_true, atol=tol):
            return False
    return True

def _try_find_lower_rank(target_rank):
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Number of variables
    num_vars = (dim_U + dim_V + dim_W) * target_rank
    
    def unpack_vars(x):
        U_flat = x[:dim_U * target_rank]
        V_flat = x[dim_U * target_rank : dim_U * target_rank + dim_V * target_rank]
        W_flat = x[dim_U * target_rank + dim_V * target_rank :]
        U = U_flat.reshape((dim_U, target_rank))
        V = V_flat.reshape((dim_V, target_rank))
        W = W_flat.reshape((dim_W, target_rank))
        return U, V, W

    def objective(x):
        U, V, W = unpack_vars(x)
        res = 0.0
        # Check C_ik = Sum_j A_ij B_jk
        # We check the identity for all i, k, j
        # Coeff of A_ij B_jk in C_ik is Sum_r W[k*n+i, r] U[i*m+j, r] V[j*p+k, r]
        # This must be 1.
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    val = 0
                    for c in range(target_rank):
                        val += W[k * n + i, c] * U[i * m + j, c] * V[j * p + k, c]
                    res += (val - 1.0) ** 2
        
        # Check that A_i'j' B_j''k'' with j' != j'' does not contribute to C_ik
        # Coeff of A_i'j' B_j''k'' in C_ik is Sum_r W[k*n+i, r] U[i'*m+j', r] V[j''*p+k'', r]
        # This must be 0.
        # To reduce computation, we only check a subset or all?
        # Number of such terms: n * p * (n*m) * (m*p) * (m-1)
        # 2 * 4 * 6 * 12 * 2 = 1152 terms.
        # Each term involves summing over target_rank (23).
        # 1152 * 23 * 3 mults ~ 80k ops.
        # This is fast.
        for i in range(n):
            for k in range(p):
                for i_prime in range(n):
                    for j_prime in range(m):
                        for j_double_prime in range(m):
                            if j_prime == j_double_prime:
                                continue
                            for k_prime in range(p):
                                val = 0
                                for c in range(target_rank):
                                    val += W[k * n + i, c] * U[i_prime * m + j_prime, c] * V[j_double_prime * p + k_prime, c]
                                res += val ** 2
        return res

    np.random.seed(42)
    best_solution = None
    
    # Try a few restarts
    for restart in range(10):
        if _time_left() < 5:
            break
            
        # Initialize with noise around schoolbook (truncated to target_rank)
        # Or random sparse init
        if restart == 0:
            # Use first target_rank columns of schoolbook
            U_sb, V_sb, W_sb = _build_schoolbook()
            U0 = U_sb[:, :target_rank]
            V0 = V_sb[:, :target_rank]
            W0 = W_sb[:, :target_rank]
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            x0 += np.random.randn(len(x0)) * 0.1
        else:
            x0 = np.random.randn(num_vars) * 0.5
            # Sparse init: set 70% to 0
            x0[np.random.choice(num_vars, int(num_vars * 0.7), replace=False)] = 0
        
        try:
            from scipy.optimize import minimize
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < 1e-8:
                U, V, W = unpack_vars(res.x)
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                if _verify(U_snap, V_snap, W_snap):
                    return (U_snap, V_snap, W_snap)
        except Exception:
            pass
            
    return None

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Get schoolbook solution
    U_sb, V_sb, W_sb = _build_schoolbook()
    
    # Try to find rank 23
    if _time_left() > 20:
        sol = _try_find_lower_rank(rank_schoolbook - 1)
        if sol is not None:
            U, V, W = sol
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback to schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
