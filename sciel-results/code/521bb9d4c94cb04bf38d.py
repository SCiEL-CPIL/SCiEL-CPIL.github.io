"""Program for matrix multiplication 2,2,10 using block-diagonal Strassen decomposition with rank reduction search."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def _build_base_strassen():
    """Build the base 35-rank Strassen decomposition."""
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    
    terms = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (1,0,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    rank = num_blocks * 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r in range(num_blocks):
        for t, (u_list, v_list, w_list) in enumerate(terms):
            col = r * 7 + t
            for i, j, val in u_list:
                U[i*m + j][col] = val
            for j, k, val in v_list:
                V[j*p + (2*r + k)][col] = val
            for k, i, val in w_list:
                W[(2*r + k)*n + i][col] = val
                
    return U, V, W, rank

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly represents matrix multiplication."""
    # Convert to numpy arrays
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    # Build the multiplication tensor from the decomposition
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # U has shape (n*m, rank), indexed by (i*m + j)
    # V has shape (m*p, rank), indexed by (j*p + k)  
    # W has shape (p*n, rank), indexed by (k*n + i)
    
    # For each (i,j,k), compute the product
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                # The tensor entry should be 1 if this is a valid multiplication term
                # For matrix multiplication C[i,k] = sum_j A[i,j] * B[j,k]
                # The tensor T[i,j,k] = delta_{i,k} is not right...
                
                # Actually, the matrix multiplication tensor has:
                # T[i,j,k] = 1 if we're computing C[i,k] and using A[i,j]*B[j,k]
                # So T[i,j,k] = 1 for all valid (i,j,k) triples
                # But that's not how tensor rank decomposition works for matmul.
                
                # Let me reconsider. The matmul tensor is defined as:
                # T_{i,j,k} = delta_{i,k} ... no.
                
                # The correct definition: for multiplying A (n x m) by B (m x p),
                # the tensor T has entries T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
                # and T[i,j,k] = 1 if and only if this term contributes to C[i,k]
                # which is always true for valid indices. But that gives rank n*m*p.
                
                # I think the correct formulation is:
                # The matmul tensor M has M[i,j,k] = 1 for all i,j,k in their ranges
                # and the decomposition satisfies:
                # M[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                
                # Wait, that doesn't seem right either. Let me think more carefully.
                
                # The standard formulation: C = AB where C[i,k] = sum_j A[i,j] B[j,k]
                # The tensor T is defined by T[i,j,k] = 1 (all ones) and we want:
                # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                
                # No, that's not right. The correct tensor for matmul is:
                # T_{i,j,k} = delta_{i,k} ... no.
                
                # Actually, I recall now. The matmul tensor is:
                # T[i,j,k] = 1 for all valid i,j,k, and the decomposition is:
                # 1 = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                
                # But that can't be right because the rank would be at least n*m*p.
                
                # Let me look at this differently. The standard way:
                # We want to compute C[i,k] = sum_j A[i,j] * B[j,k]
                # This is a 3-tensor with indices (i,j,k) and T[i,j,k] = 1
                # The rank decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                
                # Hmm, but for 2x2x2 matmul, the rank is 7 (Strassen), not 8.
                # So the tensor must be something else.
                
                # I think the correct tensor is:
                # T[i,j,k] = 1 if we're computing C[i,k] using A[i,j]*B[j,k]
                # But C[i,k] = sum_j A[i,j]*B[j,k], so for fixed (i,k), we sum over j.
                # The tensor entry T[i,j,k] should be 1 for all (i,j,k) in the valid range.
                
                # For 2x2x2: there are 8 entries, all 1, and rank is 7.
                # That means sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
                
                # Let me verify with Strassen for 2x2x2:
                # The 7 terms should satisfy this.
                
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                
                if abs(actual - expected) > 1e-10:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """Try to find a lower-rank decomposition by identifying linear dependencies."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    original_rank = U.shape[1]
    best_rank = original_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Strategy: try to find columns that are linear combinations of others
    # by checking if removing a column and adjusting others still gives a valid decomposition.
    
    # For each column, check if it can be expressed as a combination of others
    # This is expensive, so we'll use a heuristic approach.
    
    # Try removing one column at a time and see if we can compensate
    for col_to_remove in range(original_rank):
        if time.time() > deadline:
            break
            
        if original_rank - 1 < best_rank:
            continue
            
        # Create matrices without this column
        mask = [i for i in range(original_rank) if i != col_to_remove]
        U_reduced = U[:, mask]
        V_reduced = V[:, mask]
        W_reduced = W[:, mask]
        
        # Check if the reduced decomposition still works
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    actual = 0.0
                    for r in range(len(mask)):
                        actual += U_reduced[u_idx, r] * V_reduced[v_idx, r] * W_reduced[w_idx, r]
                    
                    if abs(actual - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best_rank = original_rank - 1
            best_U, best_V, best_W = U_reduced.copy(), V_reduced.copy(), W_reduced.copy()
    
    # Convert back to lists
    best_U_list = best_U.tolist()
    best_V_list = best_V.tolist()
    best_W_list = best_W.tolist()
    
    return best_U_list, best_V_list, best_W_list, best_rank

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the base decomposition
    U, V, W, base_rank = _build_base_strassen()
    
    # Try to reduce the rank
    U_reduced, V_reduced, W_reduced, reduced_rank = _try_reduce_rank(U, V, W, n, m, p, deadline)
    
    return {"U": U_reduced, "V": V_reduced, "W": W_reduced}
# EVOLVE-BLOCK-END
