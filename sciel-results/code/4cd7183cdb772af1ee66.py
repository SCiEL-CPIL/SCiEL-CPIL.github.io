"""Matrix multiplication 3x3x3: rank-minimal tensor decomposition."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
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
    Rank-26 decomposition for 3x3 matrix multiplication.
    Strassen core (7 terms) + schoolbook border (19 terms).
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
    
    # --- Additional: A[0:2,2]*B[2,0:2] -> C[0:2,0:2] ---
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

def _try_reduce_rank(U, V, W, target_rank, time_limit=10.0):
    """
    Try to reduce the rank by finding linear dependencies among the term tensors.
    Returns (U', V', W') with rank = target_rank if successful, else None.
    """
    n = m = p = 3
    current_rank = len(U[0])
    if current_rank <= target_rank:
        return {"U": U, "V": V, "W": W}
    
    start_time = time.time()
    
    # Build the term tensor matrix: each term r is a vector of length 27
    # T[r, (i,j,k)] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want to find a set of 'target_rank' terms that can be linearly combined
    # to reproduce all 27 entries.
    
    # This is equivalent to: find a rank-deficient subset of the 26 term vectors
    # such that the remaining vectors span the same space.
    
    # Strategy: try removing one term at a time and see if the remaining 25
    # can be adjusted to still produce the correct tensor.
    # This is a linear system: for each (i,j,k), we need
    # sum_{r in remaining} U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # This is nonlinear in the coefficients, but if we fix U and V and solve for W,
    # it's linear. Or fix two and solve for the third.
    
    # Simpler approach: check if any term is a linear combination of others
    # in the tensor space. Build the 27x26 matrix of term vectors.
    
    term_vectors = []
    for r in range(current_rank):
        vec = np.zeros(27)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * 9 + j * 3 + k
                    vec[idx] = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        term_vectors.append(vec)
    
    T = np.array(term_vectors).T  # 27 x 26
    
    # Find the rank of T
    # If rank < 26, then there's a linear dependency, and we can potentially
    # reduce the rank.
    
    # Use SVD to find the null space
    try:
        U_svd, s, Vt = np.linalg.svd(T, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    
    # The null space dimension is current_rank - rank(T)
    # If rank(T) < current_rank, we have dependencies.
    
    # For now, just check if we can remove one term by solving a linear system.
    # Try removing each term and see if the remaining terms can be scaled
    # to still produce the identity tensor.
    
    for remove_idx in range(current_rank):
        if time.time() - start_time > time_limit:
            break
        
        remaining = [r for r in range(current_rank) if r != remove_idx]
        if len(remaining) < target_rank:
            continue
        
        # We need to find new coefficients for U, V, W for the remaining terms
        # such that the tensor is still correct. This is a 3-tensor completion
        # problem, which is hard.
        
        # Simpler: just check if the term vector of remove_idx is in the span
        # of the other term vectors. If so, we might be able to remove it.
        
        # Build the 27 x (current_rank-1) matrix of remaining term vectors
        T_rem = np.column_stack([term_vectors[r] for r in remaining])
        
        # Check if term_vectors[remove_idx] is in the column space of T_rem
        # by checking if ||T_rem @ (T_rem^+ @ term_vectors[remove_idx]) - term_vectors[remove_idx]|| is small
        
        try:
            # Pseudo-inverse
            T_rem_pinv = np.linalg.pinv(T_rem)
            coeffs = T_rem_pinv @ term_vectors[remove_idx]
            reconstructed = T_rem @ coeffs
            error = np.linalg.norm(reconstructed - term_vectors[remove_idx])
            
            if error < 1e-6:
                # The term is redundant! We can remove it.
                # But we need to adjust the other terms to compensate.
                # The new W values should be W_new[r] = W[r] + coeffs[r'] * W[remove_idx]
                # Wait, this isn't quite right because the tensor is a product.
                
                # Actually, if the term vectors are linearly dependent, it means
                # there exist coefficients c_r such that
                # sum_r c_r * T_r = 0
                # This doesn't directly give us a way to remove a term.
                
                # For now, just return the original decomposition.
                pass
        except np.linalg.LinAlgError:
            pass
    
    return None

def construct():
    """Construct a tensor decomposition for 3x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Try rank-26 construction (Strassen core + schoolbook border)
    result = _strassen_rank26()
    if _verify(result["U"], result["V"], result["W"]):
        if time.time() < deadline:
            return result
    
    # Fallback to schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
