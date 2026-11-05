import numpy as np
import time
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [2, 3, 5]

def _verify_decomposition_fast(U, V, W, n, m, p):
    """
    Fast verification using NumPy. Returns True if valid.
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Build tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U_np[idx_u], V_np[idx_v] * W_np[idx_w])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _verify_decomposition_exact(U, V, W, n, m, p):
    """
    Exact verification using Fraction arithmetic.
    """
    rank = len(U[0])
    
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                total = Fraction(0)
                for r in range(rank):
                    total += U_frac[idx_u][r] * V_frac[idx_v][r] * W_frac[idx_w][r]
                
                if total != 1:
                    return False
    return True

def _try_remove_terms(U, V, W, n, m, p, terms_to_remove):
    """
    Try to remove a set of terms. Returns (new_U, new_V, new_W) if successful.
    """
    rank = len(U[0])
    if rank <= len(terms_to_remove):
        return None
    
    U_new = [row[:] for row in U]
    V_new = [row[:] for row in V]
    W_new = [row[:] for row in W]
    
    for t in terms_to_remove:
        for i in range(len(U_new)):
            U_new[i][t] = 0
        for i in range(len(V_new)):
            V_new[i][t] = 0
        for i in range(len(W_new)):
            W_new[i][t] = 0
    
    # Fast check first
    if not _verify_decomposition_fast(U_new, V_new, W_new, n, m, p):
        return None
    
    # Exact check
    if not _verify_decomposition_exact(U_new, V_new, W_new, n, m, p):
        return None
    
    # Compact
    keep = [j for j in range(rank) if j not in terms_to_remove]
    U_compact = [[row[j] for j in keep] for row in U_new]
    V_compact = [[row[j] for j in keep] for row in V_new]
    W_compact = [[row[j] for j in keep] for row in W_new]
    return (U_compact, V_compact, W_compact)

def _construct_rank_28():
    """
    Constructs the known rank-28 decomposition for 2x3 * 3x5.
    """
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A2 B2 (j=2, k=0..4)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: A1 B13 (j=0,1, k=4)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen (7 terms)
    
    # M1
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M2
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1
    W[1 * n + 1][r] = -1
    r += 1
    
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1
    V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1
    V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1
    W[0 * n + 1][r] = 1
    r += 1
    
    # M5
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1
    W[1 * n + 0][r] = 1
    r += 1
    
    # M6
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1
    V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M7
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen (7 terms)
    
    # M1
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M2
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1
    W[3 * n + 1][r] = -1
    r += 1
    
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1
    W[2 * n + 1][r] = 1
    r += 1
    
    # M5
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1
    W[3 * n + 0][r] = 1
    r += 1
    
    # M6
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1
    V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M7
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    return {"U": U, "V": V, "W": W}

def _try_linear_reduction(U, V, W, n, m, p):
    """
    Try to find a linear combination that eliminates one term.
    For each term t, check if the tensor contribution of term t can be
    expressed as a linear combination of the other terms.
    This is a necessary condition for removal but not sufficient in general.
    We use it as a heuristic to prioritize which terms to try removing.
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # For each term t, compute its "signature" across all (i,j,k) entries
    # and check if it's linearly dependent on others
    removable_candidates = []
    
    for t in range(rank):
        # Build matrix where each column is a term's contribution pattern
        # For term r, the contribution to (i,j,k) is U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # We want to know if column t is in the span of other columns
        
        num_entries = n * m * p
        patterns = np.zeros((num_entries, rank))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    for r in range(rank):
                        patterns[idx, r] = U_np[idx_u, r] * V_np[idx_v, r] * W_np[idx_w, r]
        
        # Check if column t is in span of others
        others = list(range(rank))
        others.remove(t)
        
        # Use SVD to check rank
        A = patterns[:, others]
        b = patterns[:, t]
        
        # Check if b is in column space of A
        # Use least squares
        try:
            sol, residuals, rank_svd, sv = np.linalg.lstsq(A, b, rcond=None)
            residual = np.linalg.norm(A @ sol - b)
            if residual < 1e-8:
                removable_candidates.append(t)
        except:
            pass
    
    return removable_candidates

def construct():
    start_time = time.time()
    deadline = 140.0  # seconds, with safety margin before 150s kill
    
    # Construct the known optimal rank-28 decomposition
    res_28 = _construct_rank_28()
    U = res_28['U']
    V = res_28['V']
    W = res_28['W']
    
    # Verify it to ensure correctness
    if not _verify_decomposition_exact(U, V, W, 2, 3, 5):
        raise ValueError("Verification failed for rank-28 decomposition")
    
    n, m, p = 2, 3, 5
    rank = len(U[0])
    best_result = {'U': U, 'V': V, 'W': W}
    best_rank = rank
    
    # Try to reduce rank by removing single terms
    for term_idx in range(rank):
        if time.time() - start_time > deadline - 20:
            break
        
        result = _try_remove_terms(U, V, W, n, m, p, [term_idx])
        if result is not None:
            U, V, W = result
            rank = len(U[0])
            best_result = {'U': U, 'V': V, 'W': W}
            best_rank = rank
            print(f"Reduced rank to {rank} by removing term {term_idx}")
    
    # Try to reduce further by removing pairs
    for pair in combinations(range(rank), 2):
        if time.time() - start_time > deadline - 15:
            break
        
        result = _try_remove_terms(U, V, W, n, m, p, list(pair))
        if result is not None:
            U, V, W = result
            rank = len(U[0])
            best_result = {'U': U, 'V': V, 'W': W}
            best_rank = rank
            print(f"Reduced rank to {rank} by removing pair {pair}")
            break
    
    # Try triple removals if still time
    if time.time() - start_time < deadline - 10:
        for triple in combinations(range(rank), 3):
            if time.time() - start_time > deadline - 10:
                break
            result = _try_remove_terms(U, V, W, n, m, p, list(triple))
            if result is not None:
                U, V, W = result
                rank = len(U[0])
                best_result = {'U': U, 'V': V, 'W': W}
                best_rank = rank
                print(f"Reduced rank to {rank} by removing triple {triple}")
                break
    
    # Final single removal pass if rank was reduced
    if best_rank < 28:
        for term_idx in range(len(U[0])):
            if time.time() - start_time > deadline - 5:
                break
            result = _try_remove_terms(U, V, W, n, m, p, [term_idx])
            if result is not None:
                U, V, W = result
                best_result = {'U': U, 'V': V, 'W': W}
                best_rank = len(U[0])
                print(f"Further reduced rank to {best_rank} by removing term {term_idx}")
    
    return best_result
