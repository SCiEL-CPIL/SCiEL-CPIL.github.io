"""QAP solver: Multi-start VND with Segment Reversal (Or-opt) moves."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s kill (evaluator kills at 150s, but contract says 120s. Use 115s to be safe.)

    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))

    def get_swap_delta(p, i, j):
        """
        Calculate delta cost for swapping facilities at indices i and j in permutation p.
        """
        u = p[i]
        v = p[j]
        
        # Direct pair term
        delta = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
        
        # Diagonal terms
        delta += a[i, i] * (b[v, v] - b[u, u])
        delta += a[j, j] * (b[u, u] - b[v, v])
        
        # Cross terms with all other k
        diff_a_row = a[i] - a[j]
        diff_a_col = a[:, i] - a[:, j]
        
        diff_b_row = b[v, p] - b[u, p]
        diff_b_col = b[p, v] - b[p, u]
        
        sum_T1 = np.dot(diff_a_row, diff_b_row)
        sum_T2 = np.dot(diff_a_col, diff_b_col)
        
        # Subtract k=i terms
        sum_T1 -= (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        sum_T2 -= (a[i, i] - a[i, j]) * (b[u, v] - b[u, u])
        
        # Subtract k=j terms
        sum_T1 -= (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        sum_T2 -= (a[j, i] - a[j, j]) * (b[v, v] - b[v, u])
        
        delta += sum_T1 + sum_T2
        return int(delta)

    def get_reversal_delta(p, i, j):
        """
        Calculate delta cost for reversing the segment p[i:j+1].
        This is a structural move that changes the relative order of many facilities.
        """
        if i == j:
            return 0
        
        # Extract the segment
        seg = p[i:j+1]
        # The new segment will be seg[::-1]
        
        # The cost change involves:
        # 1. Terms within the segment: a[fac_k, fac_l] * b[loc_k, loc_l]
        #    When reversed, facility at pos i+k goes to pos j-k.
        #    So a[seg[k], seg[l]] * b[p[i+k], p[j-l]] becomes a[seg[k], seg[l]] * b[p[j-l], p[i+k]]
        #    Wait, the assignment is facility f at location L.
        #    Cost = sum_{f1, f2} a[f1, f2] * b[L1, L2]
        #    In segment reversal, the set of facilities {seg} stays the same, and the set of locations {p[i:j+1]} stays the same.
        #    But the pairing changes.
        #    Old: fac seg[k] is at loc p[i+k].
        #    New: fac seg[k] is at loc p[j-k].
        
        # Let's compute the cost difference directly for the sub-matrix.
        # This is O(len(seg)^2). For n=100, max len is 100, so 10k ops. 
        # If we do this for many segments, it might be slow.
        # However, we only evaluate a subset of segments in VND.
        
        # Let's try to vectorize or optimize this.
        # Cost_old_segment = sum_{k=0..L-1} sum_{l=0..L-1} a[seg[k], seg[l]] * b[p[i+k], p[i+l]]
        # Cost_new_segment = sum_{k=0..L-1} sum_{l=0..L-1} a[seg[k], seg[l]] * b[p[j-l], p[i+k]]  <-- Wait.
        
        # Let's trace carefully.
        # Position r in [i, j].
        # Old facility at r: fac_old[r] = p[r].
        # New facility at r: fac_new[r] = p[i + j - r].
        
        # Contribution of pair (facility X at pos r1, facility Y at pos r2):
        # Old: a[X, Y] * b[p[r1], p[r2]]
        # New: a[X, Y] * b[p[i+j-r1], p[i+j-r2]]
        
        # This is getting complicated to do with simple dot products because the indices in b are permuted non-linearly relative to the segment start.
        
        # Alternative: Just compute the cost of the whole permutation before and after?
        # No, that's O(n^2) = 10k ops per evaluation.
        # A full scan of O(n^2) segments would be 10k * 10k = 100M ops. Too slow for Python loops.
        
        # We need a faster delta for reversal.
        # Is there a way to compute it in O(n) or O(n log n)?
        # The change in cost for reversing segment [i, j] only affects terms where at least one facility is in the segment.
        # Let S be the set of facilities in the segment. Let L be the set of locations in the segment.
        # Terms entirely outside S and L are unchanged.
        # Terms with one in S and one outside:
        #   Facility f in S moves from loc l_old to l_new.
        #   Interaction with facility g outside S at loc l_g.
        #   Change: a[f, g] * (b[l_new, l_g] - b[l_old, l_g]) + a[g, f] * (b[l_g, l_new] - b[l_g, l_old])
        #   Sum over all f in S, g not in S.
        
        # Terms entirely inside S:
        #   Both facilities move within L.
        #   Change: sum_{f, g in S} a[f, g] * (b[l_new(f), l_new(g)] - b[l_old(f), l_old(g)])
        
        # This still requires summing over |S| * (n - |S|) + |S|^2 terms.
        # For small segments (len <= 5), this is fast.
        # For large segments, it's slow.
        
        # Strategy: Use short segments (len 2, 3, 4) for fast VND.
        # Len 2 is just a swap.
        # Len 3 is a 3-cycle/reversal of 3.
        # Len 4 is a 4-reversal.
        
        # Let's implement a specialized delta for short reversals (len <= 4) to keep it fast.
        # For len > 4, we can skip or use a slower check less frequently.
        
        L = j - i + 1
        if L > 4:
            # Skip long reversals in the fast pass to maintain speed
            return 0 

        # Get the facilities in the segment
        seg_facs = p[i:j+1]
        # Get the locations in the segment
        seg_locs = p[i:j+1]
        
        # Old cost contribution of these facilities with themselves and others
        # New cost contribution
        
        # Let's just compute the difference by summing the affected terms.
        # Affected facilities: seg_facs
        # Affected locations: seg_locs
        
        # 1. Internal terms (both facilities in seg)
        # Old: sum_{k=0}^{L-1} sum_{l=0}^{L-1} a[seg_facs[k], seg_facs[l]] * b[seg_locs[k], seg_locs[l]]
        # New: sum_{k=0}^{L-1} sum_{l=0}^{L-1} a[seg_facs[k], seg_facs[l]] * b[seg_locs[L-1-k], seg_locs[L-1-l]]
        
        A_sub = a[np.ix_(seg_facs, seg_facs)]
        B_old = b[np.ix_(seg_locs, seg_locs)]
        B_new = b[np.ix_(seg_locs[::-1], seg_locs[::-1])]
        
        delta_internal = np.sum(A_sub * B_new) - np.sum(A_sub * B_old)
        
        # 2. External terms (one in seg, one outside)
        # For each f in seg_facs:
        #   old_loc = old position in p
        #   new_loc = new position in p (reversed)
        #   Delta = sum_{g not in seg} ( a[f, g] * (b[new_loc, loc_g] - b[old_loc, loc_g]) + a[g, f] * (b[loc_g, new_loc] - b[loc_g, old_loc]) )
        
        # We can vectorize this over the external facilities.
        # Let mask_ext be boolean array of facilities NOT in seg.
        # Let loc_ext be p[mask_ext]
        
        # This is getting complex to vectorize perfectly for arbitrary short segments without precomputation.
        # Given the time limit, let's stick to SWAPS for the main VND, but add a "kick" of random reversals in the ILS phase.
        
        # Actually, let's implement a simpler structural change: 
        # Instead of complex delta for reversals, we will use a **2-opt style** move which is equivalent to a reversal.
        # But since we don't have a fast delta for it, we will rely on **Swaps** for the local search, 
        # but use **Random Reversals** for the perturbation in the ILS loop.
        # This is a standard ILS strategy: Local Search with swaps, Perturb with larger structural moves.
        
        return 0 # Placeholder, we will use swaps for LS and reversals for perturbation

    def local_search_swaps(p, current_cost):
        """
        Best-improvement local search using swaps.
        """
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > deadline - 2.0:
                break
            
            best_delta = 0
            best_i = -1
            best_j = -1
            
            for i in range(n):
                if time.time() - start_time > deadline - 2.0:
                    break
                for j in range(i + 1, n):
                    delta = get_swap_delta(p, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            
            if best_i != -1:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                improved = True
        
        return p, current_cost

    def hungarian_init():
        """
        Initialize using Hungarian algorithm on surrogate cost.
        """
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        
        C = -np.outer(a_row_sums, b_row_sums)
        
        row_ind, col_ind = linear_sum_assignment(C)
        
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def perturb_reversal(p, max_len=10):
        """
        Apply a random segment reversal to p.
        """
        p = p.copy()
        if n < 2:
            return p
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i > j:
            i, j = j, i
        
        # Limit length to avoid huge disruptions that might be too hard to recover from
        if j - i + 1 > max_len:
            # Try again or just pick a shorter one
            # For simplicity, just accept it if it's not too long, or resample
            # Let's just resample if too long
            for _ in range(5):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i > j:
                    i, j = j, i
                if j - i + 1 <= max_len:
                    break
            
        p[i:j+1] = p[i:j+1][::-1]
        return p

    # 1. Initialization
    p = hungarian_init()
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost

    # 2. Initial Local Search (Best Improvement)
    p, current_cost = local_search_swaps(p, current_cost)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()

    # 3. ILS Loop with Reversal Perturbation
    no_improve_count = 0
    rng = np.random.default_rng()
    
    while time.time() - start_time < deadline - 2.0:
        # Adaptive perturbation: increase max_len if stuck
        if no_improve_count < 5:
            max_len = 5
        elif no_improve_count < 15:
            max_len = 10
        elif no_improve_count < 30:
            max_len = 20
        else:
            max_len = 50
            
        # Perturb with reversal
        p_perturbed = perturb_reversal(best_p, max_len)
        
        # Local Search (Swaps)
        p_perturbed, cost_perturbed = local_search_swaps(p_perturbed, calc_cost(p_perturbed))
        
        if cost_perturbed < best_cost:
            best_cost = cost_perturbed
            best_p = p_perturbed.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
            # If stuck for a long time, try a random restart
            if no_improve_count > 50:
                p_restart = hungarian_init()
                p_restart, cost_restart = local_search_swaps(p_restart, calc_cost(p_restart))
                if cost_restart < best_cost:
                    best_cost = cost_restart
                    best_p = p_restart.copy()
                no_improve_count = 0

    return best_p.tolist()
# EVOLVE-BLOCK-END
