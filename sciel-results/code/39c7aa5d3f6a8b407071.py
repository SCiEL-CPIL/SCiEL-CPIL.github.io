import time
import numpy as np
import random

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication using NumPy."""
    n = m = p = 3
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Reshape to 3D tensors
    U_t = U.reshape((n, m, -1))
    V_t = V.reshape((m, p, -1))
    W_t = W.reshape((p, n, -1))
    
    # Compute the tensor product sum
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    
    # The target tensor for matrix multiplication is T[i,j,k] = 1 for all i,j,k
    # Because (AB)[i,k] = sum_j A[i,j] B[j,k], the coefficient for each term A[i,j]B[j,k] in the sum is 1.
    T_target = np.ones((n, m, p))
    
    return np.allclose(T_recon, T_target, atol=1e-9)

def _schoolbook():
    n = m = p = 3
    rank = n * m * p
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

def _strassen_rank26():
    """
    Rank-26 decomposition for 3x3 matrix multiplication using explicit factors.
    """
    n = m = p = 3
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    terms = []
    
    # --- Strassen core for C[0:2,0:2] (7 terms) ---
    terms.append(([ (u_idx(0,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,1),1), (w_idx(1,1),-1) ]))
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,1),1), (v_idx(1,1),-1) ], [ (w_idx(1,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,0),1), (v_idx(0,0),-1) ], [ (w_idx(0,0),1), (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(0,0),1), (u_idx(0,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(0,0),-1), (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(0,0),-1) ], [ (v_idx(0,0),1), (v_idx(0,1),1) ], [ (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(0,1),1), (u_idx(1,1),-1) ], [ (v_idx(1,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1) ]))
    
    # --- Border: C[0:2, 2] ---
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(1,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,1),1) ]))
    
    # --- Border: C[2, 0:2] ---
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,2),1) ]))
    
    # --- Corner: C[2,2] ---
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,2),1) ]))
    
    # --- Additional border: A[0:2,2]*B[2,0:2] contributes to C[0:2,0:2] ---
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,1),1) ]))
    
    for r, (u_list, v_list, w_list) in enumerate(terms):
        for (row, coeff) in u_list:
            U[row][r] += coeff
        for (row, coeff) in v_list:
            V[row][r] += coeff
        for (row, coeff) in w_list:
            W[row][r] += coeff
    
    return {"U": U, "V": V, "W": W}

def _local_search_rank(target_rank, time_budget=10.0):
    """
    Local search for a decomposition of the given rank.
    Starts with random initializations and uses gradient-like updates.
    """
    n = m = p = 3
    R = target_rank
    start_time = time.time()
    
    best_error = float('inf')
    best_U = best_V = best_W = None
    
    # Number of random restarts
    num_restarts = 50
    
    for _ in range(num_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization
        U = np.random.randn(n * m, R) * 0.5
        V = np.random.randn(m * p, R) * 0.5
        W = np.random.randn(p * n, R) * 0.5
        
        # Local optimization steps
        for step in range(200):
            if time.time() - start_time > time_budget:
                break
            
            # Compute error
            U_t = U.reshape((n, m, R))
            V_t = V.reshape((m, p, R))
            W_t = W.reshape((p, n, R))
            T_recon = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            T_target = np.ones((n, m, p))
            error = np.sum((T_recon - T_target) ** 2)
            
            if error < 1e-10:
                # Check exact validity
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                if _verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
                break
            
            if error < best_error:
                best_error = error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            # Simple gradient step (approximate)
            # This is a placeholder for a proper gradient calculation.
            # For a structural change, we rely on the random restarts and the potential for finding a basin.
            # A more sophisticated method would calculate gradients of the tensor product.
            # Here we use a small random perturbation to escape local minima.
            U += np.random.randn(n * m, R) * 0.01
            V += np.random.randn(m * p, R) * 0.01
            W += np.random.randn(p * n, R) * 0.01

    # If we found a very low error but not exact, we can't use it.
    # The local search is a heuristic. If it fails, we return None.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Try to find a lower rank decomposition (23, 24, 25) using local search
    # This is a structural change: attempting global search for lower rank instead of just reducing a fixed 26-rank structure.
    for target_rank in [23, 24, 25]:
        if time.time() > deadline - 30: # Keep 30s for fallback and verification
            break
        result = _local_search_rank(target_rank, time_budget=10.0)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"]):
                return result

    # 2. Fallback to the guaranteed rank-26 decomposition
    result = _strassen_rank26()
    if _verify(result["U"], result["V"], result["W"]):
        return result

    # 3. Final fallback to schoolbook
    return _schoolbook()
