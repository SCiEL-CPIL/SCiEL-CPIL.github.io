import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 128
    
    # Precompute all 4-term APs
    # ap_list[i] = list of (a, d) such that a+3d = i (0-indexed)
    # Actually, for incremental update, it's better to have:
    # For each AP (a, a+d, a+2d, a+3d), store the indices.
    # When color of index x changes, we need to update violations for all APs containing x.
    
    # Let's store:
    # ap_indices: list of tuples (i0, i1, i2, i3)
    # pos_to_aps: for each position p, list of indices into ap_indices where p appears
    
    ap_indices = []
    pos_to_aps = [[] for _ in range(n)]
    
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            i0 = a
            i1 = a + d
            i2 = a + 2 * d
            i3 = a + 3 * d
            if i3 < n:
                idx = len(ap_indices)
                ap_indices.append((i0, i1, i2, i3))
                pos_to_aps[i0].append(idx)
                pos_to_aps[i1].append(idx)
                pos_to_aps[i2].append(idx)
                pos_to_aps[i3].append(idx)
    
    num_aps = len(ap_indices)
    
    # Initial state: all distinct colors
    # colors[i] = i
    # But we want to minimize the number of distinct colors used.
    # We can represent the state as an array of color labels.
    # To make "merge" moves easy, we can use a mapping:
    # physical_color[i] = the actual color label assigned to position i.
    # However, for merging, it's easier to think in terms of "groups".
    # Let's stick to direct labels but use a "merge" operation:
    # Pick two colors c1, c2. Remap all c1 to c2 (or vice versa).
    
    # To efficiently track violations, we maintain:
    # ap_violations[ap_idx] = number of distinct colors in that AP (0 to 4)
    # A violation exists if ap_violations[ap_idx] < 3.
    # total_violations = sum(1 for v in ap_violations if v < 3)
    
    # Initial colors: 0..127
    colors = np.arange(n, dtype=np.int32)
    
    # Calculate initial ap_violations
    # For each AP, count distinct colors.
    # Since initial colors are all distinct, every AP has 4 distinct colors.
    # So ap_violations is all 4.
    # total_violations = 0.
    ap_violations = np.full(num_aps, 4, dtype=np.int32)
    total_violations = 0
    num_distinct_colors = n
    
    # We also need to know which colors are currently used.
    # used_colors = set(colors)
    # But for speed, let's just track the count of distinct colors in the `colors` array.
    # Actually, since we start with all distinct, and we only merge, the number of distinct colors decreases.
    # We can track `active_colors` as a set or just count unique values in `colors`.
    # Counting unique values in a 128-length array is fast enough (O(n)).
    
    best_colors = colors.copy()
    best_distinct = n
    best_violations = 0
    
    # Simulated Annealing
    # Temperature starts high, cools down.
    # Move: Pick two distinct colors c1, c2 present in the array.
    #       Randomly decide to merge c1 into c2 or c2 into c1.
    #       Apply the merge to `colors`.
    #       Update `ap_violations` and `total_violations` for affected APs.
    #       Calculate new distinct color count.
    #       Accept if new_distinct < old_distinct (always good) or
    #       if new_violations < old_violations (good) or
    #       with probability exp(-(delta_energy)/T).
    #       Energy = (violations * 100) + (distinct_colors * 1) ?
    #       We want to minimize violations first, then distinct colors.
    #       Actually, the task requires valid colorings (violations = 0).
    #       So we should prioritize keeping violations = 0 if possible.
    #       If we start with 0 violations, we should only accept moves that keep violations = 0.
    #       If we get stuck with violations > 0, we allow moves that increase distinct colors
    #       (by splitting? No, we only have merge moves in this design) or increase violations
    #       to escape local minima.
        
    # Wait, if I only use merge moves, I can never increase the number of distinct colors.
    # If I get stuck in a local minimum with violations > 0, I cannot split a color back.
    # I need "split" moves or "recolor" moves to escape.
    
    # Let's use a more flexible move set:
    # 1. Recolor a single position to a random existing color or a new color (if we allow increasing distinct count).
    # 2. Merge two colors.
    
    # Given the small n=128, let's try a simpler heuristic first:
    # Randomized Local Search with "Recolor" moves.
    # State: colors array.
    # Move: Pick a random position i, assign a random color c from 0..K_max (where K_max is current distinct + 1 or similar).
    # Objective: Minimize (violations, distinct_colors).
    
    # Let's restart with a better initial state or a different search.
    # The inspiration said: "Randomized local search with incremental violation tracking and color merging should find valid colorings with fewer colors more efficiently than the CSP approach".
    
    # Let's implement:
    # 1. Start with a valid coloring (all distinct).
    # 2. Loop:
    #    a. Try to merge two random colors.
    #    b. If merge reduces violations or keeps them 0 and reduces distinct count, accept.
    #    c. If merge increases violations, accept with probability based on T.
    #    d. If stuck (no merges possible or T low), try a "split" move? Or just restart with random init?
    
    # Actually, a simpler and very effective approach for small n:
    # Start with a random assignment using k colors (e.g., k=4).
    # Calculate violations.
    # While violations > 0:
    #    Find a violating AP.
    #    Recolor one of its positions to a random color from 0..k-1 (or introduce new color if k is small).
    # This is basically a repair heuristic.
    
    # Let's combine:
    # 1. Try to find a valid coloring with small k using repair.
    # 2. Then try to reduce k further by merging.
    
    def count_distinct(arr):
        return len(set(arr.tolist()))
    
    def calc_violations(colors_arr):
        # Returns list of distinct counts for each AP
        viol_counts = np.zeros(num_aps, dtype=np.int32)
        for idx, (i0, i1, i2, i3) in enumerate(ap_indices):
            c0, c1, c2, c3 = colors_arr[i0], colors_arr[i1], colors_arr[i2], colors_arr[i3]
            # Count distinct
            # Since n is small, set() is fast
            viol_counts[idx] = len({c0, c1, c2, c3})
        return viol_counts
    
    def get_violation_count(viol_counts):
        return np.sum(viol_counts < 3)
    
    # Strategy:
    # 1. Initialize with k=3 random colors.
    # 2. Repair violations by recoloring positions in violating APs.
    # 3. Once valid, try to merge colors to reduce k.
    
    best_result = None
    best_score = 128  # Max possible distinct colors
    
    # Try different starting k values
    for start_k in [3, 4, 5, 6]:
        if time.time() > deadline - 5:
            break
            
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
                
            # Initialize randomly with start_k colors
            colors = np.random.randint(0, start_k, size=n, dtype=np.int32)
            viol_counts = calc_violations(colors)
            total_viol = get_violation_count(viol_counts)
            current_distinct = count_distinct(colors)
            
            # Repair phase
            max_repair_steps = 5000
            for step in range(max_repair_steps):
                if time.time() > deadline - 5:
                    break
                if total_viol == 0:
                    break
                    
                # Pick a random violating AP
                viol_aps = np.where(viol_counts < 3)[0]
                if len(viol_aps) == 0:
                    break
                ap_idx = viol_aps[np.random.randint(len(viol_aps))]
                i0, i1, i2, i3 = ap_indices[ap_idx]
                
                # Pick a random position in the AP to recolor
                pos_idx = np.random.randint(4)
                if pos_idx == 0:
                    pos = i0
                elif pos_idx == 1:
                    pos = i1
                elif pos_idx == 2:
                    pos = i2
                else:
                    pos = i3
                
                old_color = colors[pos]
                
                # Try new colors
                # We can try all colors 0..start_k-1 or introduce a new color if stuck
                # Let's try existing colors first
                found_fix = False
                for new_c in range(start_k):
                    if new_c == old_color:
                        continue
                    
                    # Simulate change
                    colors[pos] = new_c
                    
                    # Update viol_counts for APs containing pos
                    # We need to re-calculate distinct count for APs containing pos
                    # For efficiency, we could pre-compute, but for n=128, re-calc is ok
                    # However, we need to update total_viol correctly.
                    
                    # Let's just recalculate the specific APs
                    # But wait, we need to update viol_counts array.
                    # Let's store the old viol counts for affected APs
                    affected_aps = pos_to_aps[pos]
                    old_viol_affected = [viol_counts[ai] for ai in affected_aps]
                    
                    # Calculate new viol counts for affected APs
                    new_viol_affected = []
                    for ai in affected_aps:
                        a0, a1, a2, a3 = ap_indices[ai]
                        new_viol_affected.append(len({colors[a0], colors[a1], colors[a2], colors[a3]}))
                    
                    # Update total_viol
                    # old_total = total_viol
                    # new_total = total_viol - sum(1 for v in old_viol_affected if v < 3) + sum(1 for v in new_viol_affected if v < 3)
                    
                    old_viol_sum = sum(1 for v in old_viol_affected if v < 3)
                    new_viol_sum = sum(1 for v in new_viol_affected if v < 3)
                    new_total_viol = total_viol - old_viol_sum + new_viol_sum
                    
                    if new_total_viol < total_viol:
                        # Accept
                        for j, ai in enumerate(affected_aps):
                            viol_counts[ai] = new_viol_affected[j]
                        total_viol = new_total_viol
                        found_fix = True
                        break
                    else:
                        # Revert
                        colors[pos] = old_color
                        # No need to revert viol_counts as we didn't update them
                        # But we did update colors[pos] back
                        pass
                
                if not found_fix:
                    # Try introducing a new color if we are stuck and start_k is small
                    # Or just accept a move that doesn't reduce violations but might help later?
                    # For now, let's try a random color from a larger range if start_k is small
                    if start_k < 8:
                        new_c = np.random.randint(0, 16)
                        if new_c != old_color:
                            colors[pos] = new_c
                            affected_aps = pos_to_aps[pos]
                            old_viol_affected = [viol_counts[ai] for ai in affected_aps]
                            new_viol_affected = []
                            for ai in affected_aps:
                                a0, a1, a2, a3 = ap_indices[ai]
                                new_viol_affected.append(len({colors[a0], colors[a1], colors[a2], colors[a3]}))
                            
                            old_viol_sum = sum(1 for v in old_viol_affected if v < 3)
                            new_viol_sum = sum(1 for v in new_viol_affected if v < 3)
                            new_total_viol = total_viol - old_viol_sum + new_viol_sum
                            
                            # Accept if it doesn't make things too much worse (simulated annealing-like)
                            # Or just accept if it reduces violations
                            if new_total_viol <= total_viol:
                                for j, ai in enumerate(affected_aps):
                                    viol_counts[ai] = new_viol_affected[j]
                                total_viol = new_total_viol
                                found_fix = True
                                # Note: this might increase distinct colors
                            
                            if not found_fix:
                                colors[pos] = old_color
                    else:
                        # Stuck
                        break
            
            if total_viol == 0:
                # Valid coloring found
                current_distinct = count_distinct(colors)
                if current_distinct < best_score:
                    best_score = current_distinct
                    best_result = colors.copy()
                
                # Try to reduce distinct colors further by merging
                # We can try to merge colors that appear in the same "context" or just random merges
                # Let's do a local search on the valid coloring to reduce distinct count
                # Move: Pick two colors c1, c2. Merge c1 into c2.
                # Accept if still valid.
                
                for merge_step in range(1000):
                    if time.time() > deadline - 5:
                        break
                    
                    # Get list of active colors
                    active = list(set(colors.tolist()))
                    if len(active) <= 1:
                        break
                    
                    # Pick two random distinct colors
                    c1, c2 = random.sample(active, 2)
                    
                    # Simulate merge c1 -> c2
                    # Find positions with c1
                    mask = (colors == c1)
                    if not np.any(mask):
                        continue
                    
                    # Save old colors
                    old_colors = colors[mask].copy()
                    
                    # Apply merge
                    colors[mask] = c2
                    
                    # Check validity
                    # We need to check APs containing any position in mask
                    # Let's just re-calculate violations for affected APs
                    affected_aps = set()
                    for p in np.where(mask)[0]:
                        for ai in pos_to_aps[p]:
                            affected_aps.add(ai)
                    
                    is_valid = True
                    for ai in affected_aps:
                        a0, a1, a2, a3 = ap_indices[ai]
                        if len({colors[a0], colors[a1], colors[a2], colors[a3]}) < 3:
                            is_valid = False
                            break
                    
                    if is_valid:
                        # Accept merge
                        # Update viol_counts? Not needed if we only care about validity
                        # But we need to update best_result if distinct count decreased
                        new_distinct = count_distinct(colors)
                        if new_distinct < best_score:
                            best_score = new_distinct
                            best_result = colors.copy()
                        # Update active list implicitly by next loop
                    else:
                        # Revert
                        colors[mask] = old_colors
                    
                    if len(set(colors.tolist())) == 1:
                        break
                
                if len(set(colors.tolist())) == 1:
                    break # Can't do better than 1, but task requires >=3 colors on APs, so 1 color is invalid.
                # Actually, if distinct < 3, it's likely invalid.
                # The merge check ensures validity.
    
    if best_result is not None:
        return best_result.tolist()
    
    # Fallback: all distinct
    return list(range(n))

# Test locally
if __name__ == "__main__":
    res = construct()
    print(f"Result: {res}")
    print(f"Distinct: {len(set(res))}")
    
    # Verify
    n = 128
    valid = True
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            i0, i1, i2, i3 = a, a+d, a+2*d, a+3*d
            if i3 < n:
                if len({res[i0], res[i1], res[i2], res[i3]}) < 3:
                    valid = False
                    print(f"Invalid AP: {i0}, {i1}, {i2}, {i3} -> {res[i0]}, {res[i1]}, {res[i2]}, {res[i3]}")
                    break
        if not valid:
            break
    print(f"Valid: {valid}")
