import time
import random
import itertools

P = {'n': 12, 'm': 40}

def _count_comparable_pairs(sets):
    m = len(sets)
    cnt = 0
    for i in range(m):
        si = sets[i]
        si_set = set(si) if si else None
        for j in range(i + 1, m):
            sj = sets[j]
            if not si and not sj:
                continue
            if not si:
                cnt += 1
            elif not sj:
                cnt += 1
            else:
                if len(si) < len(sj):
                    is_sub = True
                    s_j_set = set(sj)
                    for x in si:
                        if x not in s_j_set:
                            is_sub = False
                            break
                    if is_sub:
                        cnt += 1
                elif len(sj) < len(si):
                    is_sub = True
                    s_i_set = si_set
                    for x in sj:
                        if x not in s_i_set:
                            is_sub = False
                            break
                    if is_sub:
                        cnt += 1
    return cnt

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    m = P['m']
    total_masks = 1 << n
    
    # Precompute comparability for all masks
    # comp[i] is a bitmask of masks comparable to i
    comp = [0] * total_masks
    for i in range(total_masks):
        ci = 0
        for j in range(total_masks):
            if i == j:
                continue
            if (i & j) == i or (i & j) == j:
                ci |= (1 << j)
        comp[i] = ci
    
    def score_of_selection(sel):
        cnt = 0
        for i in range(m):
            for j in range(i + 1, m):
                if (sel[i] & sel[j]) == sel[i] or (sel[i] & sel[j]) == sel[j]:
                    cnt += 1
        return cnt

    def generate_chains():
        """Generate 3 disjoint chains of length 13 and 1 singleton."""
        used = set()
        chains = []
        
        # Chain 1: Standard chain
        c1 = []
        for k in range(n + 1):
            mask = (1 << k) - 1
            c1.append(mask)
        chains.append(c1)
        used.update(c1)
        
        # Chain 2: Try to find a disjoint chain of length 13
        # We need 13 distinct masks not in used, forming a chain.
        # A simple heuristic: shift the bits.
        # C1: 0, 1, 3, 7, ... (binary 0, 1, 11, 111...)
        # Let's try starting with {1} and adding bits in a different order.
        # But {1} is in C1? No, C1 has 0, 1, 3...
        # Wait, (1<<1)-1 = 1. So {1} is in C1.
        # (1<<0)-1 = 0.
        # So C1 is {0}, {0,1}, {0,1,2}...
        # We need masks not in C1.
        # {1,2} is not in C1.
        # Let's try to build C2 by adding bits 1, 2, 3... to a base.
        # Base: {1}. But {1} is in C1.
        # So we must skip {1}.
        # Can we have a chain of length 13 without {1} or {0}?
        # Max chain length is 13.
        # If we exclude {0} and {1}, we lose 2 steps.
        # So we can only get length 11?
        # No, we can start from a set of size 1 that is not in C1?
        # C1 contains all sets of the form {0, ..., k-1}.
        # {1} is in C1. {2} is NOT in C1.
        # So we can start C2 with {2}.
        # Chain: {2}, {2,0}? No, {0,2} is not in C1.
        # {2,0} = {0,2}.
        # {2,0,1} = {0,1,2}. This IS in C1.
        # So we can't use {0,1,2}.
        # This is tricky.
        
        # Alternative: Just use random search to find disjoint chains.
        # Or, just use the local search to fix a bad initial solution.
        # Let's initialize with a random selection and run local search.
        # To get a good start, we can use the "colex" order.
        # The first 40 sets in colex order form a family with many comparable pairs.
        
        # Colex order: Sort by largest element, then next largest, etc.
        # Actually, just sorting by integer value is not colex.
        # Let's generate masks in increasing order of size, then by value.
        # This tends to create clusters of comparable sets.
        
        masks_by_size = [[] for _ in range(n + 1)]
        for mask in range(total_masks):
            masks_by_size[mask.bit_count()].append(mask)
        
        # Shuffle within sizes to allow randomization, but keep size structure
        rng = random.Random(42)
        for size in range(n + 1):
            rng.shuffle(masks_by_size[size])
        
        initial_sel = []
        for size in range(n + 1):
            for mask in masks_by_size[size]:
                if len(initial_sel) < m:
                    initial_sel.append(mask)
                else:
                    break
            if len(initial_sel) >= m:
                break
        
        # Ensure we have exactly m distinct masks
        # The above loop might not fill m if n is small, but n=12, m=40 is fine.
        
        return initial_sel

    def local_search(selected, rng, time_limit):
        sel = list(selected)
        sel_set = set(sel)
        current_score = score_of_selection(sel)
        best_score = current_score
        best_sel = list(sel)
        
        # Precompute losses
        def compute_losses(sel_list):
            losses = [0] * len(sel_list)
            for i in range(len(sel_list)):
                for j in range(len(sel_list)):
                    if i == j:
                        continue
                    if (sel_list[i] & sel_list[j]) == sel_list[i] or (sel_list[i] & sel_list[j]) == sel_list[j]:
                        losses[i] += 1
            return losses
        
        losses = compute_losses(sel)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # 1-swap moves
            for idx in range(m):
                if time.time() >= time_limit:
                    return best_sel, best_score
                old = sel[idx]
                loss = losses[idx]
                # Try all non-selected masks
                # Optimization: Only try masks that are "close" in size?
                # For now, try all.
                for new in range(total_masks):
                    if new in sel_set:
                        continue
                    gain = 0
                    for j in range(m):
                        if j == idx:
                            continue
                        if (new & sel[j]) == new or (new & sel[j]) == sel[j]:
                            gain += 1
                    delta = gain - loss
                    if delta > 0:
                        sel_set.remove(old)
                        sel_set.add(new)
                        sel[idx] = new
                        current_score += delta
                        losses = compute_losses(sel)
                        if current_score > best_score:
                            best_score = current_score
                            best_sel = list(sel)
                        improved = True
                        break
        
        return best_sel, best_score

    best_overall_score = -1
    best_overall_sel = None
    
    rng = random.Random(42)
    
    # Initial solution
    init_sel = generate_chains()
    s1, sc1 = local_search(init_sel, rng, deadline - 5)
    if sc1 > best_overall_score:
        best_overall_score = sc1
        best_overall_sel = s1
    
    # More restarts
    restart_count = 0
    while time.time() < deadline - 3 and restart_count < 5:
        # Random initialization
        init = rng.sample(range(total_masks), m)
        s, sc = local_search(init, rng, deadline - 3)
        if sc > best_overall_score:
            best_overall_score = sc
            best_overall_sel = s
        restart_count += 1
    
    if best_overall_sel is None:
        best_overall_sel = list(range(m))
    
    # Convert to lists
    result = []
    for mask in best_overall_sel:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    final_result = []
    for sl in result:
        final_result.append([int(x) for x in sl])
    
    return final_result
