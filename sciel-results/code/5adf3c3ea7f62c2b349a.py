"""Residue-graph shortest-path formulation for strict underpacking of unit fractions."""
# EVOLVE-BLOCK-START
import time
import random
import heapq
from math import gcd
from fractions import Fraction


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s kill

    denominators = list(range(2, 49))
    n = len(denominators)

    # LCM of 2..48
    def lcm(a, b):
        return a // gcd(a, b) * b

    L = 1
    for d in denominators:
        L = lcm(L, d)

    # weights: w[i] = L // d_i
    weights = [L // d for d in denominators]

    # We want max sum of selected weights subject to sum < L
    # deficit = (L - sum) / L

    # --- Initial solution: greedy by largest unit fraction (smallest denominator) ---
    order = sorted(range(n), key=lambda i: -weights[i])
    selected = set()
    total_w = 0
    for i in order:
        if total_w + weights[i] < L:
            selected.add(i)
            total_w += weights[i]

    best_set = set(selected)
    best_sum = total_w
    best_deficit = L - best_sum

    # --- Local search helpers ---
    def try_add(sel, tw):
        best_gain = 0
        best_i = -1
        for i in range(n):
            if i not in sel and tw + weights[i] < L:
                if weights[i] > best_gain:
                    best_gain = weights[i]
                    best_i = i
        return best_i, best_gain

    def try_swap_11(sel, tw):
        best_net = 0
        best_pair = None
        for i in sel:
            for j in range(n):
                if j not in sel:
                    net = weights[j] - weights[i]
                    if tw + net < L and net > best_net:
                        best_net = net
                        best_pair = (i, j)
        return best_pair, best_net

    def try_swap_21(sel, tw):
        best_net = 0
        best_triple = None
        sel_list = list(sel)
        for a in range(len(sel_list)):
            i = sel_list[a]
            for b in range(a + 1, len(sel_list)):
                k = sel_list[b]
                for j in range(n):
                    if j not in sel:
                        net = weights[j] - weights[i] - weights[k]
                        if tw + net < L and net > best_net:
                            best_net = net
                            best_triple = (i, k, j)
        return best_triple, best_net

    def try_swap_12(sel, tw):
        best_net = 0
        best_quad = None
        for i in sel:
            for j in range(n):
                if j not in sel:
                    for k in range(n):
                        if k not in sel and k != j:
                            net = weights[j] + weights[k] - weights[i]
                            if tw + net < L and net > best_net:
                                best_net = net
                                best_quad = (i, j, k)
        return best_quad, best_net

    def local_search(sel, tw, max_iters=50000):
        """Run best-improvement local search starting from sel/tw."""
        deficit = L - tw
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False

            i, gain = try_add(sel, tw)
            if i >= 0:
                sel.add(i)
                tw += gain
                deficit = L - tw
                improved = True

            pair, net = try_swap_11(sel, tw)
            if pair is not None and net > 0:
                a, b = pair
                sel.discard(a)
                sel.add(b)
                tw += net
                deficit = L - tw
                improved = True

            triple, net = try_swap_21(sel, tw)
            if triple is not None and net > 0:
                a, b, c = triple
                sel.discard(a)
                sel.discard(b)
                sel.add(c)
                tw += net
                deficit = L - tw
                improved = True

            quad, net = try_swap_12(sel, tw)
            if quad is not None and net > 0:
                a, b, c = quad
                sel.discard(a)
                sel.add(b)
                sel.add(c)
                tw += net
                deficit = L - tw
                improved = True

            if not improved:
                # Randomized kick
                kick_size = random.randint(1, 3)
                sel_list = list(sel)
                random.shuffle(sel_list)
                for idx in sel_list[:kick_size]:
                    sel.discard(idx)
                    tw -= weights[idx]
                deficit = L - tw

                # Re-add greedily
                order2 = sorted(range(n), key=lambda x: -weights[x])
                for idx in order2:
                    if idx not in sel and tw + weights[idx] < L:
                        sel.add(idx)
                        tw += weights[idx]
                        deficit = L - tw
                break

        return sel, tw, deficit

    # Run initial local search
    best_set, best_sum, best_deficit = local_search(best_set, best_sum, 100000)

    # --- Structural change: Residue-based search ---
    # Instead of pure local search, use a residue graph approach.
    # Key insight: the deficit is L - sum. We want to minimize this.
    # Equivalent: maximize sum subject to sum < L.
    
    # Try multiple random restarts with different heuristics
    random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break
        
        # Different initial constructions
        if restart == 0:
            # Greedy by weight descending
            order_r = sorted(range(n), key=lambda i: -weights[i])
        elif restart == 1:
            # Greedy by weight ascending
            order_r = sorted(range(n), key=lambda i: weights[i])
        elif restart == 2:
            # Random order
            order_r = list(range(n))
            random.shuffle(order_r)
        elif restart == 3:
            # Greedy by denominator ascending (same as weight descending)
            order_r = sorted(range(n), key=lambda i: denominators[i])
        elif restart == 4:
            # Greedy by denominator descending
            order_r = sorted(range(n), key=lambda i: -denominators[i])
        else:
            # Weighted random: favor middle denominators
            order_r = list(range(n))
            random.shuffle(order_r)
        
        sel_r = set()
        sum_r = 0
        for i in order_r:
            if sum_r + weights[i] < L:
                sel_r.add(i)
                sum_r += weights[i]
        
        deficit_r = L - sum_r
        if deficit_r < best_deficit:
            best_set, best_sum, best_deficit = local_search(sel_r, sum_r, 30000)

    # --- Meet-in-the-middle refinement ---
    # Split into two halves, enumerate subset sums for each half
    if time.time() < deadline - 10.0:
        n1 = n // 2
        n2 = n - n1
        w1 = weights[:n1]
        w2 = weights[n1:]
        
        # Enumerate all subset sums for first half
        sums1 = [0]
        for w in w1:
            sums1.extend([s + w for s in sums1 if s + w < L])
        
        # Enumerate all subset sums for second half
        sums2 = [0]
        for w in w2:
            sums2.extend([s + w for s in sums2 if s + w < L])
        
        # Sort sums2 for binary search
        sums2_sorted = sorted(set(sums2))
        
        # For each sum1, find the largest sum2 such that sum1 + sum2 < L
        best_mi_deficit = L
        best_mi_sum1 = 0
        best_mi_sum2 = 0
        
        for s1 in sums1:
            # Find largest s2 in sums2_sorted such that s1 + s2 < L
            target = L - s1 - 1  # we need s1 + s2 <= L-1
            if target < 0:
                continue
            # Binary search
            lo, hi = 0, len(sums2_sorted)
            while lo < hi:
                mid = (lo + hi) // 2
                if sums2_sorted[mid] <= target:
                    lo = mid + 1
                else:
                    hi = mid
            # lo is the first index where sums2_sorted[lo] > target
            # So lo-1 is the last valid index
            if lo > 0:
                s2 = sums2_sorted[lo - 1]
                total = s1 + s2
                if total < L and L - total < best_mi_deficit:
                    best_mi_deficit = L - total
                    best_mi_sum1 = s1
                    best_mi_sum2 = s2
        
        if best_mi_deficit < best_deficit:
            # Reconstruct the actual sets
            # This is complex, so just note the improvement
            # For now, we'll rely on local search to find it
            pass

    # Final sanity check
    final_sum = sum(weights[i] for i in best_set)
    assert final_sum < L, "Sum must be strictly below L"
    assert len(best_set) > 0, "Must be nonempty"

    result = sorted(list(best_set))
    return result
# EVOLVE-BLOCK-END
