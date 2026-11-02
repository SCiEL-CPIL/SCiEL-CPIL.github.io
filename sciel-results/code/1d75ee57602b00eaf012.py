"""Matrix multiplication 3x3x3: rank-minimal tensor decomposition."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

def _verify(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n = m = p = 3
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
        
    rank = U.shape[1]
    
    # Construct the tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want T to be the all-ones tensor for the multiplication identity?
    # No, the identity is: (AB)[i,k] = sum_j A[i,j] B[j,k].
    # The tensor decomposition represents the multiplication tensor T_{ijk} = delta_{i,j} delta_{j,k} is wrong.
    # The multiplication tensor M has entries M[i, j, k] = 1 if we are computing C[i,k] = sum_j A[i,j]B[j,k].
    # The condition for a rank-r decomposition is:
    # For all i, j, k: sum_{r=1}^r U[i,j,r] V[j,k,r] W[k,i,r] = 1 if (i,j,k) is a "valid" triplet?
    # No. The standard definition:
    # T = sum_{r=1}^R u_r \otimes v_r \otimes w_r
    # where T is the matrix multiplication tensor.
    # T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} appears in C_{i,k}.
    # Specifically, C_{i,k} = sum_{j=0}^{m-1} A_{i,j} B_{j,k}.
    # So the tensor T has T_{i,j,k} = 1 for all i,j,k in range(3)?
    # Let's check the trace convention W[k*n+i, r].
    # The evaluator checks: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1.0 for all i,j,k.
    # This implies the target tensor is the ALL-ONES tensor of shape (3,3,3).
    # This is correct for the "multiplication tensor" in the context of bilinear maps where
    # we want to compute all products A[i,j]*B[j,k] and sum them?
    # Wait. If T_{ijk} = 1 for all i,j,k, then sum_r U_i_j^r V_j_k^r W_k_i^r = 1.
    # This is the condition provided in the task description's `_verify` function.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _rank26_solution():
    """
    Returns a valid rank-26 decomposition.
    Uses Strassen's 7-term algorithm for the 2x2 core and schoolbook for borders.
    """
    n = m = p = 3
    rank = 26
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    terms = []
    
    # --- Strassen core for C[0:2,0:2] (7 terms) ---
    # M1 = (A00+A11)*(B00+B11)
    terms.append(([ (u_idx(0,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1), (w_idx(1,1),1) ]))
    # M2 = (A10+A11)*B00
    terms.append(([ (u_idx(1,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,1),1), (w_idx(1,1),-1) ]))
    # M3 = A00*(B01-B11)
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,1),1), (v_idx(1,1),-1) ], [ (w_idx(1,0),1), (w_idx(1,1),1) ]))
    # M4 = A11*(B10-B00)
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,0),1), (v_idx(0,0),-1) ], [ (w_idx(0,0),1), (w_idx(0,1),1) ]))
    # M5 = (A00+A01)*B11
    terms.append(([ (u_idx(0,0),1), (u_idx(0,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(0,0),-1), (w_idx(1,0),1) ]))
    # M6 = (A10-A00)*(B00+B01)
    terms.append(([ (u_idx(1,0),1), (u_idx(0,0),-1) ], [ (v_idx(0,0),1), (v_idx(0,1),1) ], [ (w_idx(1,1),1) ]))
    # M7 = (A01-A11)*(B10+B11)
    terms.append(([ (u_idx(0,1),1), (u_idx(1,1),-1) ], [ (v_idx(1,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1) ]))
    
    # --- Border: C[0:2, 2] ---
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,0),1) ]))
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    terms.append(([ (u_idx(1,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,1),1) ]))
    
    # --- Border: C[2, 0:2] ---
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,2),1) ]))
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,2),1) ]))
    
    # --- Corner: C[2,2] ---
    # C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,2),1) ]))
    
    # --- Cross terms: A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2] ---
    # These are the terms that were missing in the simple block decomposition to make it rank 26.
    # C[0,0] gets + A[0,2]*B[2,0]
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,0),1) ]))
    # C[0,1] gets + A[0,2]*B[2,1]
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,0),1) ]))
    # C[1,0] gets + A[1,2]*B[2,0]
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,1),1) ]))
    # C[1,1] gets + A[1,2]*B[2,1]
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,1),1) ]))
    
    # Total terms: 7 + 6 + 6 + 3 + 4 = 26.
    
    for r, (u_list, v_list, w_list) in enumerate(terms):
        for (row, coeff) in u_list:
            U[row, r] += coeff
        for (row, coeff) in v_list:
            V[row, r] += coeff
        for (row, coeff) in w_list:
            W[row, r] += coeff
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _search_rank23(deadline):
    """
    Attempts to find a rank-23 decomposition via numerical optimization.
    This is a heuristic search and may fail.
    """
    n = m = p = 3
    target_rank = 23
    
    # We need to find U (9x23), V (9x23), W (9x23)
    # such that sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k.
    
    # Initialize with random values, or better, perturb the rank-26 solution.
    # Since we don't have a rank-26 to 23 mapping, we use random initialization
    # with a structured prior: many entries are 0, 1, or -1.
    
    # Strategy: Use a "sparse" random initialization.
    # Each column of U, V, W has a few non-zeros.
    
    np.random.seed(42) # For reproducibility in debugging, but we want variety in production?
    # Let's use a fixed seed to ensure deterministic behavior if it fails.
    
    U = np.random.randn(n*m, target_rank) * 0.1
    V = np.random.randn(m*p, target_rank) * 0.1
    W = np.random.randn(p*n, target_rank) * 0.1
    
    # Scale to make the product of magnitudes roughly 1
    # Average product is 0.001. We need sum of 23 products to be 1.
    # So each term should be around 1/23 ~ 0.04.
    # If U,V,W are ~0.5, product is 0.125. 23 * 0.125 ~ 2.8. Too high.
    # If U,V,W are ~0.3, product is 0.027. 23 * 0.027 ~ 0.6. Close.
    
    U *= 0.5
    V *= 0.5
    W *= 0.5
    
    def residuals(x):
        U_flat = x[:n*m*target_rank].reshape(n*m, target_rank)
        V_flat = x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_flat = x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
        
        res = np.zeros(27)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(target_rank):
                        s += U_flat[i*m+j, r] * V_flat[j*p+k, r] * W_flat[k*n+i, r]
                    res[idx] = s - 1.0
                    idx += 1
        return res
    
    # Use least_squares
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        result = least_squares(residuals, x0, method='lm', max_nfev=1000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if result.cost < 1e-10:
            U_sol = result.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_sol = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_sol = result.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
            
            # Verify
            if _verify(U_sol.tolist(), V_sol.tolist(), W_sol.tolist()):
                return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
    except Exception:
        pass
        
    return None

def construct():
    """Construct a tensor decomposition for 3x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Try to find rank-23 (high value, low probability)
    if time.time() < deadline - 20:
        result_23 = _search_rank23(deadline - 20)
        if result_23 is not None:
            return result_23
            
    # 2. Fall back to rank-26 (high value, high probability)
    result_26 = _rank26_solution()
    if _verify(result_26["U"], result_26["V"], result_26["W"]):
        return result_26
        
    # 3. Fallback to schoolbook rank-27
    n = m = p = 3
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
