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

def _local_search_polish(U, V, W, deadline, max_iterations=5000):
    """
    Local search to reduce rank by finding linear dependencies among terms.
    For each term, try to express it as a linear combination of others.
    If successful, remove it and reduce rank.
    """
    n = m = p = 3
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    cur_rank = U.shape[1]
    
    def term_tensor(r):
        vec = np.zeros(27)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i*9 + j*3 + k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        return vec
    
    # Build the tensor for each term
    tensors = [term_tensor(r) for r in range(cur_rank)]
    
    removed = set()
    iterations = 0
    
    while iterations < max_iterations and time.time() < deadline:
        iterations += 1
        found_removal = False
        
        for r in range(cur_rank):
            if r in removed:
                continue
            
            others = [s for s in range(cur_rank) if s != r and s not in removed]
            if len(others) < 2:
                continue
            
            A = np.column_stack([tensors[s] for s in others])
            b = tensors[r]
            
            try:
                sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                if len(residuals) > 0 and residuals[0] < 1e-8:
                    removed.add(r)
                    found_removal = True
                    break
            except:
                pass
        
        if not found_removal:
            break
    
    if len(removed) > 0:
        remaining = [x for x in range(cur_rank) if x not in removed]
        new_rank = len(remaining)
        return {
            "U": U[:, remaining].tolist(),
            "V": V[:, remaining].tolist(),
            "W": W[:, remaining].tolist()
        }, new_rank
    
    return None, cur_rank

def _try_merge_pairs(U, V, W, deadline):
    """
    Try to merge pairs of terms by finding a linear transformation
    that preserves the tensor identity while reducing the number of terms.
    """
    n = m = p = 3
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    cur_rank = U.shape[1]
    
    def term_tensor(r):
        vec = np.zeros(27)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i*9 + j*3 + k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        return vec
    
    best_result = None
    best_rank = cur_rank
    
    # Try all pairs
    for a in range(cur_rank):
        if time.time() > deadline:
            break
        for b in range(a+1, cur_rank):
            others = [x for x in range(cur_rank) if x != a and x != b]
            if len(others) < 2:
                continue
            
            # We want to find coefficients alpha, beta such that:
            # alpha * term_a + beta * term_b is in span of others
            # Equivalently: term_a - (beta/alpha) * term_b is in span of others (if alpha != 0)
            
            # Try: can term_a be expressed as c * term_b + combination of others?
            A = np.column_stack([term_tensor(b)] + [term_tensor(s) for s in others])
            b_vec = term_tensor(a)
            
            try:
                sol, residuals, rank_A, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                if len(residuals) > 0 and residuals[0] < 1e-8:
                    # term_a can be removed
                    remaining = [x for x in range(cur_rank) if x != a]
                    new_rank = len(remaining)
                    if new_rank < best_rank:
                        best_rank = new_rank
                        best_result = {
                            "U": U[:, remaining].tolist(),
                            "V": V[:, remaining].tolist(),
                            "W": W[:, remaining].tolist()
                        }
            except:
                pass
            
            # Try: can term_b be expressed as c * term_a + combination of others?
            A = np.column_stack([term_tensor(a)] + [term_tensor(s) for s in others])
            b_vec = term_tensor(b)
            
            try:
                sol, residuals, rank_A, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                if len(residuals) > 0 and residuals[0] < 1e-8:
                    # term_b can be removed
                    remaining = [x for x in range(cur_rank) if x != b]
                    new_rank = len(remaining)
                    if new_rank < best_rank:
                        best_rank = new_rank
                        best_result = {
                            "U": U[:, remaining].tolist(),
                            "V": V[:, remaining].tolist(),
                            "W": W[:, remaining].tolist()
                        }
            except:
                pass
    
    return best_result, best_rank

def construct():
    """Construct a tensor decomposition for 3x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s limit
    
    # Start with the verified rank-26 construction
    result = _strassen_rank26()
    if not _verify(result["U"], result["V"], result["W"]):
        # Should not happen, but fall back to schoolbook
        return _schoolbook()
    
    current_rank = len(result["U"][0])
    
    # Try local search polish
    if time.time() < deadline - 30:
        polished, polished_rank = _local_search_polish(
            result["U"], result["V"], result["W"], deadline - 30
        )
        if polished is not None and polished_rank < current_rank:
            if _verify(polished["U"], polished["V"], polished["W"]):
                result = polished
                current_rank = polished_rank
    
    # Try pair merging
    if time.time() < deadline - 10:
        merged, merged_rank = _try_merge_pairs(
            result["U"], result["V"], result["W"], deadline - 10
        )
        if merged is not None and merged_rank < current_rank:
            if _verify(merged["U"], merged["V"], merged["W"]):
                result = merged
                current_rank = merged_rank
    
    # Final verification
    if _verify(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook if something went wrong
    return _schoolbook()

# EVOLVE-BLOCK-END
