import numpy as np
import time
from itertools import product, combinations, combinations_with_replacement

DIMENSIONS = [2, 3, 7]
TARGET_RANK = 38

def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    rank = U.shape[1]
    
    # Build the tensor T such that T[i,m,j] = A[i,k]*B[k,j] summed over k
    # We check: sum_r U[i,k,r] * V[k,j,r] * W[j,i,r] == A[i,k]*B[k,j] for each (i,k,j)
    # Actually the tensor for matrix multiplication C = A*B where A is n x m, B is m x p
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # The bilinear tensor has entries T[(i,k),(k,j),(j,i)] = 1 if they form a valid term
    # More precisely: T[a,b,c] where a indexes A (i,k), b indexes B (k,j), c indexes C (j,i)
    # T[a,b,c] = 1 if a=(i,k), b=(k',j), c=(j',i') and i=i', k=k', j=j'
    
    # Let's construct the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a_idx = i * m + k
                b_idx = k * p + j
                c_idx = j * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Check: sum_r U[:,a,r] * V[:,b,r] * W[:,c,r] should equal T[a,b,c]
    # We can compute this efficiently
    # For each r, the contribution is outer(U[:,r], V[:,r]) contracted with W[:,r]
    # Let's do it term by term
    result = np.zeros_like(T)
    for r in range(rank):
        # result[a,b,c] += U[a,r]*V[b,r]*W[c,r]
        # This is an outer product of U[:,r] and V[:,r] times W[:,r]
        mat = np.outer(U[:, r], V[:, r])  # shape (n*m, m*p)
        # For each c, result[:,:,c] += mat * W[c,r]
        for c in range(p*n):
            if W[c, r] != 0:
                result[:, :, c] += mat * W[c, r]
    
    return np.allclose(result, T, atol=1e-10)

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a_idx = i * m + k
                b_idx = k * p + j
                c_idx = j * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    return T

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Leave 10s margin for the 150s limit
    
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    # Start with a known valid decomposition at rank 36
    # This is the Strassen-based approach from the seed program
    def make_seed_decomp():
        rank = 36
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        
        def add_term(u_indices, v_indices, w_indices):
            nonlocal col
            for idx, val in u_indices:
                U[idx, col] += val
            for idx, val in v_indices:
                V[idx, col] += val
            for idx, val in w_indices:
                W[idx, col] += val
            col += 1
        
        def u_idx(i, k): return i * m + k
        def v_idx(k, j): return k * p + j
        def w_idx(j, i): return j * n + i
        
        for b in range(3):
            j0 = 2 * b
            j1 = 2 * b + 1
            
            # Strassen 7 terms for 2x2 * 2x2
            add_term(
                [(u_idx(0,0), 1), (u_idx(1,1), 1)],
                [(v_idx(0,j0), 1), (v_idx(1,j1), 1)],
                [(w_idx(j0,0), 1), (w_idx(j1,0), -1), (w_idx(j0,1), -1), (w_idx(j1,1), 1)]
            )
            add_term(
                [(u_idx(1,0), 1), (u_idx(1,1), 1)],
                [(v_idx(0,j0), 1)],
                [(w_idx(j0,1), 1), (w_idx(j1,1), -1)]
            )
            add_term(
                [(u_idx(0,0), 1)],
                [(v_idx(0,j1), 1), (v_idx(1,j1), -1)],
                [(w_idx(j1,0), 1), (w_idx(j1,1), 1)]
            )
            add_term(
                [(u_idx(1,1), 1)],
                [(v_idx(1,j0), 1), (v_idx(0,j0), -1)],
                [(w_idx(j0,0), 1), (w_idx(j0,1), 1)]
            )
            add_term(
                [(u_idx(0,0), 1), (u_idx(0,1), 1)],
                [(v_idx(1,j1), 1)],
                [(w_idx(j0,0), -1), (w_idx(j1,0), 1)]
            )
            add_term(
                [(u_idx(1,0), 1), (u_idx(0,0), -1)],
                [(v_idx(0,j0), 1), (v_idx(0,j1), 1)],
                [(w_idx(j1,1), 1)]
            )
            add_term(
                [(u_idx(0,1), 1), (u_idx(1,1), -1)],
                [(v_idx(1,j0), 1), (v_idx(1,j1), 1)],
                [(w_idx(j0,0), 1), (w_idx(j1,1), 1)]
            )
            
            # Outer product for A[:,2] * B[2, j0:j1]
            add_term([(u_idx(0,2), 1)], [(v_idx(2,j0), 1)], [(w_idx(j0,0), 1)])
            add_term([(u_idx(0,2), 1)], [(v_idx(2,j1), 1)], [(w_idx(j1,0), 1)])
            add_term([(u_idx(1,2), 1)], [(v_idx(2,j0), 1)], [(w_idx(j0,1), 1)])
            add_term([(u_idx(1,2), 1)], [(v_idx(2,j1), 1)], [(w_idx(j1,1), 1)])
        
        # Last column j=6
        add_term(
            [(u_idx(0,0), 1), (u_idx(1,0), 1)],
            [(v_idx(0,6), 1)],
            [(w_idx(6,0), 1), (w_idx(6,1), 1)]
        )
        add_term(
            [(u_idx(0,1), 1), (u_idx(1,1), 1)],
            [(v_idx(1,6), 1)],
            [(w_idx(6,0), 1), (w_idx(6,1), 1)]
        )
        add_term(
            [(u_idx(0,2), 1), (u_idx(1,2), 1)],
            [(v_idx(2,6), 1)],
            [(w_idx(6,0), 1), (w_idx(6,1), 1)]
        )
        
        return U, V, W, rank
    
    U, V, W, rank = make_seed_decomp()
    
    # Verify the seed decomposition
    if not _verify(U, V, W):
        # Fallback to a trivial decomposition
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k, col] = 1
                    V[k * p + j, col] = 1
                    W[j * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now try to find a lower rank decomposition
    # The theoretical minimum rank for 2x3 * 3x7 is at least max(2*3, 3*7, 2*7) = 21
    # But we need to find an exact decomposition
    
    # Try to use a random search approach to find lower rank decompositions
    # We'll try ranks from 35 down to some minimum
    
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
    
    # Try to verify if we can do better
    # For now, return the verified seed decomposition
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
