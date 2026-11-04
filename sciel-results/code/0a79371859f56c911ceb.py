"""
Simulated Annealing solver for QAPLIB wil100.
Combines multi-start greedy+random initialization with highly optimized
steepest-descent local search using vectorized delta matrix computation,
followed by Or-opt (cyclic block move) local search and a final SA polish.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _greedy_initial(a, b):
    """
    Create an initial permutation using a greedy heuristic.
    Matches facilities with high flow to locations with low distance.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy = np.argsort(-a_row_sums, kind="stable")
    # Sort locations by ascending distance sum
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _random_initial(n):
    """Generate a random permutation."""
    return np.random.permutation(n).astype(np.int64)

def _hungarian_initial(a, b):
    """
    Generate an initial permutation using the Hungarian algorithm on a
    linearization of the QAP cost.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Cost[i, j] approximates the marginal cost of assigning facility i to location j
    # based on the average distance and flow.
    cost = np.outer(a_row_sums, b_row_sums)
    
    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost)
    
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p

def _compute_delta_matrix(a, b, p):
    """
    Compute the full n x n matrix of swap deltas.
    D[i, j] is the cost change if we swap positions i and j.
    Uses O(n^3) matrix multiplications, which is fast for n=100.
    """
    n = len(p)
    Bp = b[np.ix_(p, p)]
    
    # M1[i, j] = sum_k a[k, i] Bp[k, j]
    M1 = a.T @ Bp
    # M2[i, j] = sum_k a[i, k] Bp[k, j]
    M2 = a @ Bp
    
    # Diagonals
    diag_M1 = np.diag(M1)
    diag_M2 = np.diag(M2)
    diag_Bp = np.diag(Bp)
    
    # Raw terms (sum over all k)
    # T1_raw[i, j] = M1[i, j] - M1[i, i] - M1[j, j] + M1[j, i]
    T1_raw = M1 - diag_M1[:, None] - diag_M1[None, :] + M1.T
    
    # T2_raw[i, j] = M2[i, j] - M2[i, i] - M2[j, j] + M2[j, i]
    T2_raw = M2 - diag_M2[:, None] - diag_M2[None, :] + M2.T
    
    # Corrections for k=i and k=j
    # C1[i, j] = (a[i, i] - a[i, j]) * (Bp[i, j] - Bp[i, i]) 
    #          + (a[j, i] - a[j, j]) * (Bp[j, j] - Bp[j, i])
    # Note: a[i, j] for all i, j is just a.
    # Bp[i, i] is diag_Bp[i].
    
    # Term 1 of C1: (a[i, i] - a[i, j]) * (Bp[i, j] - Bp[i, i])
    # a_diag = diag(a)
    a_diag = np.diag(a)
    # We need a[i, i] - a[i, j].
    # This is a_diag[:, None] - a
    # And Bp[i, j] - Bp[i, i] is Bp - diag_Bp[:, None]
    C1_part1 = (a_diag[:, None] - a) * (Bp - diag_Bp[:, None])
    
    # Term 2 of C1: (a[j, i] - a[j, j]) * (Bp[j, j] - Bp[j, i])
    # a[j, i] is a.T[i, j]
    # a[j, j] is a_diag[j]
    # So (a.T - a_diag[None, :])
    # Bp[j, j] - Bp[j, i] is diag_Bp[None, :] - Bp
    C1_part2 = (a.T - a_diag[None, :]) * (diag_Bp[None, :] - Bp)
    
    C1 = C1_part1 + C1_part2
    
    # C2[i, j] = (a[i, i] - a[j, i]) * (Bp[j, i] - Bp[i, i])
    #          + (a[i, j] - a[j, j]) * (Bp[j, j] - Bp[i, j])
    
    # Term 1 of C2: (a[i, i] - a[j, i]) * (Bp[j, i] - Bp[i, i])
    # a[i, i] - a[j, i] is a_diag[:, None] - a.T
    # Bp[j, i] - Bp[i, i] is Bp.T - diag_Bp[:, None]
    C2_part1 = (a_diag[:, None] - a.T) * (Bp.T - diag_Bp[:, None])
    
    # Term 2 of C2: (a[i, j] - a[j, j]) * (Bp[j, j] - Bp[i, j])
    # a[i, j] - a[j, j] is a - a_diag[None, :]
    # Bp[j, j] - Bp[i, j] is diag_Bp[None, :] - Bp.T
    C2_part2 = (a - a_diag[None, :]) * (diag_Bp[None, :] - Bp.T)
    
    C2 = C2_part1 + C2_part2
    
    # T3[i, j] = (a[i, j] - a[j, i]) * (Bp[j, i] - Bp[i, j])
    T3 = (a - a.T) * (Bp.T - Bp)
    
    D = (T1_raw - C1) + (T2_raw - C2) + T3
    
    # Set diagonal to 0 (no swap)
    np.fill_diagonal(D, 0)
    
    return D

def _local_search_fast(a, b, p, time_limit=None, start_time=None):
    """
    Perform steepest descent local search using the vectorized delta matrix.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 110.0
        
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    
    while time.time() - start_time < time_limit:
        D = _compute_delta_matrix(a, b, p)
        best_delta, best_idx = np.min(D), np.unravel_index(np.argmin(D), D.shape)
        
        if best_delta >= 0:
            break
            
        i, j = best_idx
        p[i], p[j] = p[j], p[i]
        current_cost += int(best_delta)
        
    return p, current_cost

def _compute_oropt_delta(a, b, p, i, j, k):
    """
    Compute the cost delta for an Or-opt move: remove the contiguous block
    p[i..j] (inclusive) and reinsert it before position k in the resulting
    sequence. The block length is L = j - i + 1.
    
    After removal, the remaining sequence has n-L elements. The block is
    inserted before position k in this shortened sequence.
    
    Returns the integer cost delta.
    """
    n = len(p)
    L = j - i + 1
    
    # Build the new permutation
    # Before: p[0..i-1], p[i..j], p[j+1..n-1]
    # After removal: p[0..i-1], p[j+1..n-1]
    # Insert block before position k in the shortened array:
    #   if k <= i: block goes at position k in original (shifts p[k..i-1] right)
    #   if k > i: block goes at position k+L-1... need careful indexing
    
    # Let's think in terms of the final permutation.
    # Original positions: 0, 1, ..., n-1
    # Block occupies positions i, i+1, ..., j
    # After removing block, positions shift:
    #   positions 0..i-1 stay
    #   positions j+1..n-1 shift left by L to become i..n-L-1
    # Now insert block of length L before position k in the shortened array.
    #   If k <= i: new block positions are k, k+1, ..., k+L-1
    #              (positions k..i-1 in original shift right by L)
    #   If k > i: in shortened array, position k corresponds to original
    #              position k+L (since positions >= j+1 shifted left by L,
    #              so shortened pos k = original pos k+L for k >= i)
    #              Wait, let me be more careful.
    
    # Shortened array S has elements:
    #   S[0..i-1] = p[0..i-1]
    #   S[i..n-L-1] = p[j+1..n-1]
    # Length of S is n-L.
    
    # Insert block B = p[i..j] before position k in S (0 <= k <= n-L).
    # Final permutation P':
    #   If k <= i:
    #     P'[0..k-1] = S[0..k-1] = p[0..k-1]
    #     P'[k..k+L-1] = B = p[i..j]
    #     P'[k+L..i-1+L] = S[k..i-1] = p[k..i-1]
    #     P'[i+L..n-1] = S[i..n-L-1] = p[j+1..n-1]
    #   If k > i:
    #     P'[0..i-1] = S[0..i-1] = p[0..i-1]
    #     P'[i..k+L-2] = S[i..k-1] = p[j+1..j+k-i]
    #     Wait, S[i] = p[j+1], S[i+1] = p[j+2], ..., S[k-1] = p[j+k-i]
    #     P'[i..k+L-2] has length k-i, elements S[i..k-1]
    #     P'[k+L-1..k+2L-2] = B = p[i..j]
    #     P'[k+2L-1..n-1] = S[k..n-L-1] = p[j+k-i+1..n-1]
    
    # This is getting complex. Let me just construct the new permutation
    # directly and compute the cost difference using a formula.
    
    # Actually, for n=100, we can afford to compute the delta more directly.
    # Let's build the new permutation and use the existing cost computation
    # approach but incrementally.
    
    # Build new permutation
    before = p[:i].copy()
    block = p[i:j+1].copy()
    after = p[j+1:].copy()
    
    # Shortened: before + after
    # Insert block before position k in (before + after)
    combined = np.concatenate([before, after])
    
    if k <= len(before):
        new_p = np.concatenate([combined[:k], block, combined[k:]])
    else:
        # k in range [len(before)+1, len(combined)]
        # combined[k-1] is the last element before insertion point
        new_p = np.concatenate([combined[:k], block, combined[k:]])
    
    # Compute cost delta
    old_cost = np.sum(a * b[np.ix_(p, p)])
    new_cost = np.sum(a * b[np.ix_(new_p, new_p)])
    return int(new_cost - old_cost)

def _oropt_local_search(a, b, p, time_limit=None, start_time=None):
    """
    Perform Or-opt local search: try moving contiguous blocks of length 1-3
    to new positions. Use first-improvement for speed.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 110.0
        
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    max_block_len = 3
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try blocks of length 1, 2, 3
        for L in range(1, max_block_len + 1):
            for i in range(n - L + 1):
                j = i + L - 1
                # Try inserting before each position k
                # k ranges from 0 to n-L (in the shortened array)
                # Skip trivial positions (k == i or k == j+1 which is same position)
                for k in range(n - L + 1):
                    if k == i or k == j + 1:
                        continue
                    if time.time() - start_time > time_limit:
                        return p, current_cost
                    
                    delta = _compute_oropt_delta(a, b, p, i, j, k)
                    if delta < 0:
                        # Apply the move
                        before = p[:i].copy()
                        block = p[i:j+1].copy()
                        after = p[j+1:].copy()
                        combined = np.concatenate([before, after])
                        if k <= len(before):
                            p = np.concatenate([combined[:k], block, combined[k:]])
                        else:
                            p = np.concatenate([combined[:k], block, combined[k:]])
                        current_cost += delta
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return p, current_cost

def _compute_swap_delta(a, b, p, i, j):
    """
    Compute the cost delta for swapping positions i and j in permutation p.
    Returns the integer delta.
    """
    pi = p[i]
    pj = p[j]
    
    # Term 1: sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k],p[j]] - b[p[k],p[i]])
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_all_T1 = np.dot(diff_a_col, diff_b_row)
    
    # Subtract contributions from k=i and k=j
    term_1_i = (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    term_1_j = (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    T1 = sum_all_T1 - term_1_i - term_1_j
    
    # Term 2: sum_{k != i,j} (a[i,k] - a[j,k]) * (b[p[j],p[k]] - b[p[i],p[k]])
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_all_T2 = np.dot(diff_a_row, diff_b_col)
    
    # Subtract contributions from k=i and k=j
    term_2_i = (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    term_2_j = (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    T2 = sum_all_T2 - term_2_i - term_2_j
    
    # Term 3: (a[i,j] - a[j,i]) * (b[p[j],p[i]] - b[p[i],p[j]])
    T3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    return T1 + T2 + T3

def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    best_p = None
    best_cost = float('inf')
    
    # Start 1: Greedy initialization
    p = _greedy_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    
    # 2-swap local search to refine
    p, current_cost = _local_search_fast(a, b, p, time_limit=total_deadline - 20, start_time=start_time)
    
    # Or-opt local search to escape 2-swap local minima
    p, current_cost = _oropt_local_search(a, b, p, time_limit=total_deadline - 10, start_time=start_time)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # If we still have time, try Hungarian initialization
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 15:
        p = _hungarian_initial(a, b)
        current_cost = _qap_cost(a, b, p)
        
        # 2-swap local search to refine
        p, current_cost = _local_search_fast(a, b, p, time_limit=total_deadline - 15, start_time=start_time)
        
        # Or-opt polish
        p, current_cost = _oropt_local_search(a, b, p, time_limit=total_deadline - 8, start_time=start_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
    # If we still have time, try additional random starts
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 10:
        # Try a few random starts
        num_random_starts = min(3, int(remaining_time // 12))
        for _ in range(num_random_starts):
            if time.time() - start_time > total_deadline - 8:
                break
            
            p = _random_initial(n)
            current_cost = _qap_cost(a, b, p)
            
            # 2-swap local search to refine
            p, current_cost = _local_search_fast(a, b, p, time_limit=total_deadline - 10, start_time=start_time)
            
            # Or-opt polish
            p, current_cost = _oropt_local_search(a, b, p, time_limit=total_deadline - 5, start_time=start_time)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    # Final polish with SA on the best solution found
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 2 and best_p is not None:
        p = best_p.copy()
        current_cost = best_cost
        
        T_start = max(1.0, best_cost * 0.01)
        T_end = 0.1
        elapsed_total = total_deadline
        iters = 0
        
        while time.time() - start_time < total_deadline - 1:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            delta = _compute_swap_delta(a, b, p, i, j)
            new_cost = current_cost + delta
            
            elapsed = time.time() - start_time
            T = T_start * (T_end / T_start) ** (elapsed / elapsed_total)
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            
            iters += 1
    
    if best_p is None:
        best_p = _greedy_initial(a, b)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
