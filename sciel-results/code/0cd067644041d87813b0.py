import time
import math

def construct():
    start_time = time.time()
    DEADLINE = start_time + 78.0  # safety margin before 90s kill

    N = 256
    nums = list(range(2, 257))
    n = len(nums)  # 255 nodes, index 0..254

    # Precompute conflict bitmasks
    # conflict_mask[i] = bitmask of all j where lcm(nums[i], nums[j]) <= 256
    conflict_mask = [0] * n
    for i in range(n):
        a = nums[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            b = nums[j]
            g = math.gcd(a, b)
            if a * b <= 256 * g:
                mask |= (1 << j)
        conflict_mask[i] = mask

    # Weights
    weights = [1.0 / nums[i] for i in range(n)]

    # Check deadline
    if time.time() > DEADLINE:
        return [nums[i] for i in range(n) if nums[i] > 128]

    # Phase 1: Greedy initial solution
    # Process in order of decreasing weight (smallest numbers first)
    order = sorted(range(n), key=lambda i: -weights[i])
    
    selected = 0  # bitmask of selected nodes
    for i in order:
        if (conflict_mask[i] & selected) == 0:
            selected |= (1 << i)

    best_selected = selected
    best_weight = sum(weights[i] for i in range(n) if (selected >> i) & 1)

    # Phase 2: Local search with add/swap moves
    def get_selected_set(mask):
        return [i for i in range(n) if (mask >> i) & 1]

    improved = True
    while improved and time.time() < DEADLINE - 5:
        improved = False
        selected = best_selected

        # Try adding each non-selected node
        for i in range(n):
            if not ((selected >> i) & 1):
                if (conflict_mask[i] & selected) == 0:
                    new_mask = selected | (1 << i)
                    new_weight = best_weight + weights[i]
                    if new_weight > best_weight:
                        best_selected = new_mask
                        best_weight = new_weight
                        selected = new_mask
                        improved = True

        # Try swap: remove one selected, add one non-selected
        sel_list = get_selected_set(selected)
        for i in sel_list:
            if time.time() > DEADLINE - 5:
                break
            # Remove i
            reduced = selected & ~(1 << i)
            # Try adding each non-selected j
            for j in range(n):
                if ((selected >> j) & 1) or j == i:
                    continue
                if (conflict_mask[j] & reduced) == 0:
                    new_weight = best_weight - weights[i] + weights[j]
                    if new_weight > best_weight:
                        best_selected = reduced | (1 << j)
                        best_weight = new_weight
                        selected = best_selected
                        sel_list = get_selected_set(selected)
                        improved = True
                        break  # restart inner loop with updated selection

        # Try remove one, add two
        if time.time() < DEADLINE - 10:
            sel_list = get_selected_set(selected)
            for i in sel_list:
                if time.time() > DEADLINE - 10:
                    break
                reduced = selected & ~(1 << i)
                # Find candidates that can be added to reduced
                candidates = []
                for j in range(n):
                    if ((selected >> j) & 1) or j == i:
                        continue
                    if (conflict_mask[j] & reduced) == 0:
                        candidates.append(j)
                # Try pairs from candidates
                for a_idx in range(len(candidates)):
                    j = candidates[a_idx]
                    for b_idx in range(a_idx + 1, len(candidates)):
                        k = candidates[b_idx]
                        if (conflict_mask[j] & (1 << k)) or (conflict_mask[k] & (1 << j)):
                            continue
                        new_weight = best_weight - weights[i] + weights[j] + weights[k]
                        if new_weight > best_weight:
                            best_selected = reduced | (1 << j) | (1 << k)
                            best_weight = new_weight
                            selected = best_selected
                            sel_list = get_selected_set(selected)
                            improved = True
                            break
                    if improved:
                        break

    # Phase 3: Branch and bound on small numbers (≤ 64)
    # Numbers > 128 are always includable
    # Numbers 65-128: conflict with their multiples
    # Numbers ≤ 64: complex conflicts
    # We'll do B&B on the "hard" part: numbers ≤ 64
    # But first, determine which numbers > 64 to include based on conflicts with ≤ 64

    if time.time() < DEADLINE - 3:
        # Split into: small (≤64) and large (>64)
        small_indices = [i for i in range(n) if nums[i] <= 64]
        large_indices = [i for i in range(n) if nums[i] > 64]
        
        # For each subset of small, determine best large
        # This is too expensive for all subsets, so use the current best as guide
        
        # Alternative: do B&B on small nodes, and for each partial solution,
        # greedily add compatible large nodes
        
        # Reorder small nodes by weight
        small_order = sorted(small_indices, key=lambda i: -weights[i])
        n_small = len(small_order)
        
        # Map small index to position in small_order
        small_pos = {idx: pos for pos, idx in enumerate(small_order)}
        
        # Precompute: for each small node, which large nodes does it conflict with?
        # conflict_large[i] = bitmask over large_indices of conflicts
        large_pos = {idx: pos for pos, idx in enumerate(large_indices)}
        n_large = len(large_indices)
        
        # For B&B, we need conflict within small and between small and large
        # small_conflict[i] = bitmask of small nodes conflicting with i (within small)
        small_conflict = [0] * n_small
        for pos_i, idx_i in enumerate(small_order):
            mask = 0
            for pos_j, idx_j in enumerate(small_order):
                if pos_i == pos_j:
                    continue
                if (conflict_mask[idx_i] >> idx_j) & 1:
                    mask |= (1 << pos_j)
            small_conflict[pos_i] = mask
        
        # large_conflict_from_small[pos_i] = bitmask of large nodes conflicting with small_order[pos_i]
        large_conflict_from_small = [0] * n_small
        for pos_i, idx_i in enumerate(small_order):
            mask = 0
            for pos_j, idx_j in enumerate(large_indices):
                if (conflict_mask[idx_i] >> idx_j) & 1:
                    mask |= (1 << pos_j)
            large_conflict_from_small[pos_i] = mask
        
        # large_conflict[j] = bitmask of large nodes conflicting with large_indices[j]
        large_conflict = [0] * n_large
        for pos_j, idx_j in enumerate(large_indices):
            mask = 0
            for pos_k, idx_k in enumerate(large_indices):
                if pos_j == pos_k:
                    continue
                if (conflict_mask[idx_j] >> idx_k) & 1:
                    mask |= (1 << pos_k)
            large_conflict[pos_j] = mask
        
        # large_weights
        large_weights = [weights[idx] for idx in large_indices]
        
        # Greedy large given a set of blocked large nodes
        def greedy_large(blocked_mask):
            # blocked_mask: bitmask of large nodes that are blocked
            available = 0
            for pos_j in range(n_large):
                if not ((blocked_mask >> pos_j) & 1):
                    available |= (1 << pos_j)
            
            # Greedy: process large in decreasing weight order
            large_order = sorted(range(n_large), key=lambda j: -large_weights[j])
            sel = 0
            for j in large_order:
                if not ((available >> j) & 1):
                    continue
                if (large_conflict[j] & sel) == 0:
                    sel |= (1 << j)
            # Compute weight
            w = 0.0
            for j in range(n_large):
                if (sel >> j) & 1:
                    w += large_weights[j]
            return sel, w
        
        # B&B on small nodes
        best_bb_weight = 0.0
        best_bb_small = 0  # bitmask over small_order positions
        best_bb_large = 0  # bitmask over large_indices positions
        
        def bb(pos, small_sel_mask, small_weight, blocked_large_mask):
            nonlocal best_bb_weight, best_bb_small, best_bb_large
            if time.time() > DEADLINE - 1:
                return
            
            # Upper bound: current small weight + remaining small weights + best possible large
            ub_small = small_weight
            for p in range(pos, n_small):
                ub_small += weights[small_order[p]]
            # Upper bound for large: all unblocked large
            ub_large = 0.0
            for j in range(n_large):
                if not ((blocked_large_mask >> j) & 1):
                    ub_large += large_weights[j]
            
            if ub_small + ub_large <= best_bb_weight:
                return
            
            if pos == n_small:
                # Greedy large
                large_sel, large_w = greedy_large(blocked_large_mask)
                total = small_weight + large_w
                if total > best_bb_weight:
                    best_bb_weight = total
                    best_bb_small = small_sel_mask
                    best_bb_large = large_sel
                return
            
            idx = small_order[pos]
            w = weights[idx]
            
            # Option 1: include idx
            new_blocked = blocked_large_mask | large_conflict_from_small[pos]
            new_small_mask = small_sel_mask | (1 << pos)
            # Also need to block small nodes that conflict with idx
            # (handled by checking in recursion)
            bb(pos + 1, new_small_mask, small_weight + w, new_blocked)
            
            # Option 2: exclude idx
            bb(pos + 1, small_sel_mask, small_weight, blocked_large_mask)
        
        # Start B&B
        bb(0, 0, 0.0, 0)
        
        # Convert B&B result to full selection
        if best_bb_weight > best_weight:
            new_selected = 0
            for pos_i in range(n_small):
                if (best_bb_small >> pos_i) & 1:
                    new_selected |= (1 << small_order[pos_i])
            for pos_j in range(n_large):
                if (best_bb_large >> pos_j) & 1:
                    new_selected |= (1 << large_indices[pos_j])
            best_selected = new_selected
            best_weight = best_bb_weight

    # Final local search polish
    if time.time() < DEADLINE - 1:
        improved = True
        while improved and time.time() < DEADLINE - 1:
            improved = False
            selected = best_selected
            # Add moves
            for i in range(n):
                if not ((selected >> i) & 1):
                    if (conflict_mask[i] & selected) == 0:
                        new_weight = best_weight + weights[i]
                        if new_weight > best_weight:
                            best_selected = selected | (1 << i)
                            best_weight = new_weight
                            selected = best_selected
                            improved = True
            # Swap moves
            sel_list = [i for i in range(n) if (best_selected >> i) & 1]
            for i in sel_list:
                if time.time() > DEADLINE - 1:
                    break
                reduced = best_selected & ~(1 << i)
                for j in range(n):
                    if ((best_selected >> j) & 1) or j == i:
                        continue
                    if (conflict_mask[j] & reduced) == 0:
                        new_weight = best_weight - weights[i] + weights[j]
                        if new_weight > best_weight:
                            best_selected = reduced | (1 << j)
                            best_weight = new_weight
                            selected = best_selected
                            sel_list = [i for i in range(n) if (best_selected >> i) & 1]
                            improved = True
                            break

    # Convert to result
    result = [nums[i] for i in range(n) if (best_selected >> i) & 1]
    return result
