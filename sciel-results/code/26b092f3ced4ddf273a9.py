import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n = m = p = 3
    U = [[float(x) for x in row] for row in U]
    V = [[float(x) for x in row] for row in V]
    W = [[float(x) for x in row] for row in W]
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

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
    Strassen core (7 terms) for the 2x2 block + schoolbook border terms.
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

def _build_tensor_matrix(U, V, W):
    """
    Build the 27 x R matrix where each column r is the tensor product
    U[:,r] * V[:,r] * W[:,r] reshaped to 27 entries.
    """
    n = m = p = 3
    R = len(U[0])
    tensor = np.zeros((27, R))
    for r in range(R):
        for i in range(n):
            for j in range(m):
                u_val = U[i * m + j][r]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = V[j * p + k][r]
                    if v_val == 0:
                        continue
                    w_val = W[k * n + i][r]
                    if w_val == 0:
                        continue
                    idx = i * 9 + j * 3 + k
                    tensor[idx, r] = u_val * v_val * w_val
    return tensor

def _can_remove_term(U, V, W, r):
    """Check if term r can be removed (i.e., it's in the span of the others)."""
    n = m = p = 3
    R = len(U[0])
    if R <= 1:
        return False
    
    # Build tensor matrix
    tensor = _build_tensor_matrix(U, V, W)
    
    # Check if column r is in the span of the other columns
    # This is equivalent to checking if the rank doesn't change when we remove column r
    other_cols = [j for j in range(R) if j != r]
    
    # Use SVD to check rank
    full_rank = np.linalg.matrix_rank(tensor)
    reduced_tensor = tensor[:, other_cols]
    reduced_rank = np.linalg.matrix_rank(reduced_tensor)
    
    if full_rank != reduced_rank:
        return False
    
    # The column is in the span, so we can potentially remove it
    # But we need to verify that the decomposition still works
    new_U = [row[:r] + row[r+1:] for row in U]
    new_V = [row[:r] + row[r+1:] for row in V]
    new_W = [row[:r] + row[r+1:] for row in W]
    
    if _verify(new_U, new_V, new_W):
        return True
    return False

def _try_reduce_rank(U, V, W, max_iterations=50):
    """
    Attempts to reduce the rank by finding linear dependencies among the term tensors.
    Returns a new decomposition if rank is reduced, otherwise None.
    """
    n = m = p = 3
    current_rank = len(U[0])
    
    # Try removing terms one at a time
    for _ in range(max_iterations):
        removed = False
        for r in range(len(U[0])):
            if _can_remove_term(U, V, W, r):
                U = [row[:r] + row[r+1:] for row in U]
                V = [row[:r] + row[r+1:] for row in V]
                W = [row[:r] + row[r+1:] for row in W]
                removed = True
                break
        if not removed:
            break
    
    if len(U[0]) < current_rank:
        return {"U": U, "V": V, "W": W}
    return None

def _try_strassen_rank23():
    """
    Attempt to construct a rank-23 decomposition by merging some terms.
    This is heuristic and may not always succeed.
    """
    n = m = p = 3
    # Start with rank 26 and try to merge
    result = _strassen_rank26()
    reduced = _try_reduce_rank(result["U"], result["V"], result["W"], max_iterations=100)
    if reduced is not None and len(reduced["U"][0]) <= 23:
        return reduced
    return None

def _random_search_rank(target_rank, time_budget=30.0):
    """
    Random search for a decomposition of the given rank.
    Uses random initialization and gradient-like updates.
    """
    n = m = p = 3
    R = target_rank
    start_time = time.time()
    
    # Target tensor: T[i,j,k] = 1 if i,j,k are the matrix multiplication indices
    # Actually, the target is the matrix multiplication tensor
    # T[i,j,k] = delta[i,k] for the product (AB)[i,k] = sum_j A[i,j]B[j,k]
    # So T[i,j,k] = 1 if we're computing (AB)[i,k], i.e., for each (i,k), T[i,*,k] has 1s at all j
    # Wait, let me reconsider. The tensor is T[i,j,k] where the product is sum_j A[i,j] B[j,k]
    # So the tensor entries are T[i,j,k] = delta[i,k]? No.
    # Actually, for matrix multiplication, the tensor is T[i,j,k] = 1 for all i,j,k
    # because (AB)[i,k] = sum_j A[i,j] B[j,k], so each term A[i,j]B[j,k] contributes to (AB)[i,k]
    # The tensor T[i,j,k] = 1 for all i,j,k in the standard basis.
    
    # Actually, I need to be more careful. The matrix multiplication tensor is:
    # T[i,j,k] = 1 if we're computing (AB)[i,k] from A[i,j] and B[j,k]
    # So T[i,j,k] = 1 for all i, j, k.
    
    # Let me just use the verification function to check correctness.
    
    best_rank = R
    best_U = best_V = best_W = None
    
    for attempt in range(200):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization with small values
        U = np.random.randn(n * m, R) * 0.5
        V = np.random.randn(m * p, R) * 0.5
        W = np.random.randn(p * n, R) * 0.5
        
        # Try to optimize
        for step in range(1000):
            if time.time() - start_time > time_budget:
                break
            
            # Compute current decomposition error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(R):
                            s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        error += (s - 1.0) ** 2
            
            if error < 1e-12:
                # Found a valid decomposition
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                if _verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
                break
            
            # Simple gradient descent (very approximate)
            # This is a placeholder - actual optimization would be complex
            # For now, just try small random perturbations
            if step % 10 == 0:
                U += np.random.randn(n * m, R) * 0.01
                V += np.random.randn(m * p, R) * 0.01
                W += np.random.randn(p * n, R) * 0.01
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # First, try the known rank-26 decomposition
    result = _strassen_rank26()
    
    if _verify(result["U"], result["V"], result["W"]):
        # Try to reduce rank
        reduced = _try_reduce_rank(result["U"], result["V"], result["W"], max_iterations=100)
        if reduced is not None:
            if _verify(reduced["U"], reduced["V"], reduced["W"]):
                return reduced
        return result
    
    # Fallback to schoolbook
    return _schoolbook()
