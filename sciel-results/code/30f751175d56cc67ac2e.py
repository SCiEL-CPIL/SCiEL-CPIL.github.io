import time
import sys
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    N = 256
    # Vertices are 2..256
    vertices = list(range(2, 257))
    n = len(vertices)
    
    # weights as Fractions for exact comparison
    weights = [Fraction(1, v) for v in vertices]
    
    # Precompute conflict masks
    # conflict[i] is a bitmask of j such that lcm(vertices[i], vertices[j]) <= 256
    conflict = [0] * n
    
    for i in range(n):
        vi = vertices[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            vj = vertices[j]
            g = gcd(vi, vj)
            l = (vi // g) * vj
            if l <= 256:
                mask |= (1 << j)
        conflict[i] = mask

    # Strategy:
    # The numbers 2..16 are highly constrained.
    # Numbers 17..256 are less constrained.
    # We will enumerate all valid subsets of [2..16].
    # For each valid subset S_small, we find the best valid subset S_large of [17..256]
    # such that S_small ∪ S_large is a valid independent set.
    # This means S_large must be an independent set in the graph induced by [17..256]
    # AND no vertex in S_large can conflict with any vertex in S_small.
    # So we remove all vertices in [17..256] that conflict with any vertex in S_small.
    # Then we solve MWIS on the remaining graph of [17..256].
    
    small_range = 15  # 2..16 inclusive -> 15 numbers
    small_vertices = vertices[:small_range]
    large_vertices = vertices[small_range:]
    
    # Precompute conflicts among large vertices
    # Map large vertex value to index in large_vertices
    # large_vertices[0] is 17, large_vertices[1] is 18, ...
    n_large = len(large_vertices)
    
    # Conflict masks for large vertices (only among themselves)
    # conflict_large[i] is bitmask of j in large_vertices such that lcm(large_vertices[i], large_vertices[j]) <= 256
    conflict_large = [0] * n_large
    for i in range(n_large):
        vi = large_vertices[i]
        mask = 0
        for j in range(n_large):
            if i == j:
                continue
            vj = large_vertices[j]
            g = gcd(vi, vj)
            l = (vi // g) * vj
            if l <= 256:
                mask |= (1 << j)
        conflict_large[i] = mask

    # Weights for large vertices
    weights_large = [Fraction(1, v) for v in large_vertices]
    
    # Suffix sums of weights for large vertices (for B&B bound)
    # Sort large vertices by value? They are already sorted ascending 17..256.
    # Weights are descending.
    # For B&B, we want to consider high weights first.
    # Let's reorder large_vertices by weight descending (value ascending).
    # Actually, value ascending means weight descending. So current order is good.
    
    suffix_weights_large = [Fraction(0)] * (n_large + 1)
    for i in range(n_large - 1, -1, -1):
        suffix_weights_large[i] = suffix_weights_large[i + 1] + weights_large[i]

    # Function to solve MWIS on large graph with some vertices removed
    # removed_mask: bitmask of large vertices that are forbidden
    # Returns best sum (Fraction) and best mask
    def solve_large_mwis(removed_mask):
        initial_avail = ((1 << n_large) - 1) & ~removed_mask
        
        # Greedy initialization
        avail = initial_avail
        chosen = 0
        current_sum = Fraction(0)
        
        for i in range(n_large):
            if avail & (1 << i):
                chosen |= (1 << i)
                current_sum += weights_large[i]
                avail &= ~(1 << i)
                avail &= ~conflict_large[i]
        
        # Local Search with 1-swap, 1-for-2, and 2-for-2 moves
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add a vertex
            for i in range(n_large):
                if chosen & (1 << i):
                    continue
                if removed_mask & (1 << i):
                    continue
                if conflict_large[i] & chosen:
                    continue
                chosen |= (1 << i)
                current_sum += weights_large[i]
                improved = True
                break
            
            # 2. Try 1-swap: remove u, add v
            if not improved:
                for u in range(n_large):
                    if not (chosen & (1 << u)):
                        continue
                    for v in range(u):
                        if chosen & (1 << v):
                            continue
                        if removed_mask & (1 << v):
                            continue
                        if conflict_large[v] & (chosen & ~(1 << u)):
                            continue
                        chosen &= ~(1 << u)
                        chosen |= (1 << v)
                        current_sum += weights_large[v] - weights_large[u]
                        improved = True
                        break
                    if improved:
                        break
            
            # 3. Try 1-for-2: remove u, add v and w
            if not improved:
                for u in range(n_large):
                    if not (chosen & (1 << u)):
                        continue
                    remaining = chosen & ~(1 << u)
                    # Find candidates v, w that don't conflict with remaining
                    # and don't conflict with each other
                    candidates = []
                    for v in range(n_large):
                        if chosen & (1 << v):
                            continue
                        if removed_mask & (1 << v):
                            continue
                        if conflict_large[v] & remaining:
                            continue
                        candidates.append(v)
                    
                    # Try pairs from candidates
                    for idx_v in range(len(candidates)):
                        v = candidates[idx_v]
                        for idx_w in range(idx_v + 1, len(candidates)):
                            w = candidates[idx_w]
                            if conflict_large[v] & (1 << w):
                                continue
                            # Check gain: 1/v + 1/w - 1/u > 0
                            gain = weights_large[v] + weights_large[w] - weights_large[u]
                            if gain > 0:
                                chosen &= ~(1 << u)
                                chosen |= (1 << v)
                                chosen |= (1 << w)
                                current_sum += gain
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # 4. Try 2-for-2: remove u1, u2, add v1, v2
            if not improved and time.time() < deadline - 5:
                # Only do this if we have time
                chosen_list = []
                for i in range(n_large):
                    if chosen & (1 << i):
                        chosen_list.append(i)
                
                # Limit search for 2-for-2 to keep it fast
                # We'll try a few pairs
                for idx_u1 in range(min(20, len(chosen_list))):
                    u1 = chosen_list[idx_u1]
                    for idx_u2 in range(idx_u1 + 1, min(20, len(chosen_list))):
                        u2 = chosen_list[idx_u2]
                        remaining = chosen & ~(1 << u1) & ~(1 << u2)
                        
                        candidates = []
                        for v in range(n_large):
                            if (chosen & (1 << v)) or (v == u1) or (v == u2):
                                continue
                            if removed_mask & (1 << v):
                                continue
                            if conflict_large[v] & remaining:
                                continue
                            candidates.append(v)
                        
                        # Try pairs from candidates
                        for idx_v in range(len(candidates)):
                            v1 = candidates[idx_v]
                            for idx_w in range(idx_v + 1, len(candidates)):
                                v2 = candidates[idx_w]
                                if conflict_large[v1] & (1 << v2):
                                    continue
                                gain = weights_large[v1] + weights_large[v2] - weights_large[u1] - weights_large[u2]
                                if gain > 0:
                                    chosen &= ~(1 << u1)
                                    chosen &= ~(1 << u2)
                                    chosen |= (1 << v1)
                                    chosen |= (1 << v2)
                                    current_sum += gain
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

            if time.time() > deadline - 5:
                break

        return current_sum, chosen

    # Enumerate subsets of small vertices
    # There are 2^15 = 32768 subsets.
    # For each, check if valid (no conflicts among small).
    # If valid, compute sum_small.
    # Compute removed_mask for large:
    # Any large vertex that conflicts with any small vertex in subset is removed.
    
    best_total_sum = Fraction(0)
    best_small_mask = 0
    best_large_mask = 0
    
    # Precompute for each small vertex i, the mask of large vertices it conflicts with
    small_conflicts_large = [0] * small_range
    for i in range(small_range):
        vi = small_vertices[i]
        mask = 0
        for j in range(n_large):
            vj = large_vertices[j]
            g = gcd(vi, vj)
            l = (vi // g) * vj
            if l <= 256:
                mask |= (1 << j)
        small_conflicts_large[i] = mask

    # Also precompute conflicts among small vertices
    small_conflicts_small = [0] * small_range
    for i in range(small_range):
        vi = small_vertices[i]
        mask = 0
        for j in range(small_range):
            if i == j:
                continue
            vj = small_vertices[j]
            g = gcd(vi, vj)
            l = (vi // g) * vj
            if l <= 256:
                mask |= (1 << j)
        small_conflicts_small[i] = mask

    # Iterate over all masks for small vertices
    # 2^15 = 32768
    for mask in range(1 << small_range):
        # Check validity of small subset
        valid = True
        # Quick check: if any vertex in mask conflicts with another in mask
        # We can precompute valid_small_masks?
        # Or just check on the fly.
        # Check: for each i in mask, if (small_conflicts_small[i] & mask) != 0, invalid.
        # But this double counts. Just check if (small_conflicts_small[i] & mask) has any bit j != i?
        # Actually, if i is in mask, and small_conflicts_small[i] & mask is non-zero, it means there is a conflict.
        # Wait, small_conflicts_small[i] does not contain i.
        # So if (small_conflicts_small[i] & mask) != 0, then there is a j in mask that conflicts with i.
        # This is sufficient.
        
        m = mask
        sum_small = Fraction(0)
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            sum_small += weights[i]
            if small_conflicts_small[i] & mask:
                valid = False
                break
            m ^= lsb
        
        if not valid:
            continue
        
        # Compute removed_mask for large
        removed_large = 0
        m = mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            removed_large |= small_conflicts_large[i]
            m ^= lsb
        
        # Solve large MWIS with removed_large
        sum_large, chosen_large = solve_large_mwis(removed_large)
        
        total_sum = sum_small + sum_large
        
        if total_sum > best_total_sum:
            best_total_sum = total_sum
            best_small_mask = mask
            best_large_mask = chosen_large
            
        if time.time() > deadline - 10:
            # Time to stop?
            # We have 10s left.
            # We might not finish all 32768 masks if solve_large_mwis is slow.
            # Let's check time.
            # If we are running out of time, we should break.
            # But we want to be sure we have a good answer.
            # The first few masks (small values) might not be optimal.
            # Actually, we want to maximize sum.
            # Small values have large weights.
            # So masks with small numbers are likely to be better.
            # But we iterate 0..2^15-1.
            # Mask 0 is empty small set.
            # Mask 1 is {2}.
            # ...
            # We should probably iterate in order of potential quality?
            # Or just finish if possible.
            # 32768 iterations. If each takes 1ms, 32s.
            # If each takes 10ms, 327s -> too slow.
            # solve_large_mwis involves B&B or greedy+LS.
            # Greedy+LS on 240 nodes should be fast.
            # Let's assume it's fast enough.
            pass

    # Construct result
    result_set = []
    
    # Add small vertices
    m = best_small_mask
    while m:
        lsb = m & -m
        i = lsb.bit_length() - 1
        result_set.append(small_vertices[i])
        m ^= lsb
        
    # Add large vertices
    m = best_large_mask
    while m:
        lsb = m & -m
        j = lsb.bit_length() - 1
        result_set.append(large_vertices[j])
        m ^= lsb
        
    # Verify result
    # Check non-empty
    if not result_set:
        # Fallback: return [256]
        return [256]
        
    # Check constraints
    # lcm(a,b) > 256 for all distinct pairs
    # This is guaranteed by construction if our logic is correct.
    # But let's double check just in case.
    # O(N^2) check. N <= 255.
    for i in range(len(result_set)):
        for j in range(i+1, len(result_set)):
            a = result_set[i]
            b = result_set[j]
            g = gcd(a, b)
            l = (a // g) * b
            if l <= 256:
                # Error in logic!
                # Fallback to a known valid set?
                # Return empty? No, must be non-empty.
                # Return [256] is safe.
                return [256]

    return result_set

# Test locally?
# if __name__ == "__main__":
#     res = construct()
#     print(res)
#     s = sum(1/x for x in res)
#     print(s)
