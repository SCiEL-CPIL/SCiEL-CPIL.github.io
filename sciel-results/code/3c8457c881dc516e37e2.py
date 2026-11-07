# EVOLVE-BLOCK-START
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
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = P['n']
    m = P['m']
    total_masks = 1 << n  # 4096
    
    # Precompute comparability matrix as bitsets for fast operations
    comp_bitset = [0] * total_masks
    for i in range(total_masks):
        ci = 0
        for j in range(total_masks):
            if i == j:
                continue
            if (i & j) == i or (i & j) == j:
                ci |= (1 << j)
        comp_bitset[i] = ci
    
    # Precompute list of masks by popcount
    masks_by_size = [[] for _ in range(n + 1)]
    for mask in range(total_masks):
        masks_by_size[mask.bit_count()].append(mask)
    
    def score_of_selection(sel):
        """Compute score for a selection (list of m mask IDs)."""
        cnt = 0
        for i in range(m):
            for j in range(i + 1, m):
                if (sel[i] & sel[j]) == sel[i] or (sel[i] & sel[j]) == sel[j]:
                    cnt += 1
        return cnt
    
    def greedy_initial(seed_rng):
        """Greedy construction starting from random seed."""
        selected = []
        selected_set = set()
        
        # Start with a few random masks
        for _ in range(3):
            k = seed_rng.randint(0, total_masks - 1)
            while k in selected_set:
                k = seed_rng.randint(0, total_masks - 1)
            selected.append(k)
            selected_set.add(k)
        
        # Greedily add masks that maximize comparability with current selection
        while len(selected) < m:
            best_mask = -1
            best_score = -1
            for k in range(total_masks):
                if k in selected_set:
                    continue
                # Count comparable with current selected
                s = 0
                for sel in selected:
                    if (k & sel) == k or (k & sel) == sel:
                        s += 1
                if s > best_score:
                    best_score = s
                    best_mask = k
            selected.append(best_mask)
            selected_set.add(best_mask)
        
        return selected
    
    def chain_initial(seed_rng):
        """Chain-based initialization: build a long chain of comparable sets."""
        selected = []
        selected_set = set()
        
        # Start with empty set
        selected.append(0)
        selected_set.add(0)
        
        # Build a chain by adding sets that contain all previous sets
        current = 0
        remaining = list(range(n))
        seed_rng.shuffle(remaining)
        
        # Add singletons in order
        for b in remaining:
            if len(selected) >= m:
                break
            current |= (1 << b)
            selected.append(current)
            selected_set.add(current)
        
        # If we still need more, add random supersets
        while len(selected) < m:
            k = seed_rng.randint(0, total_masks - 1)
            if k not in selected_set:
                selected.append(k)
                selected_set.add(k)
        
        return selected
    
    def local_search(selected, rng, time_limit):
        """Local search with 1-swap and 2-swap moves."""
        sel = list(selected)
        sel_set = set(sel)
        current_score = score_of_selection(sel)
        best_score = current_score
        best_sel = list(sel)
        
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
                        # Accept
                        sel_set.remove(old)
                        sel_set.add(new)
                        sel[idx] = new
                        current_score += delta
                        # Update losses
                        losses = compute_losses(sel)
                        if current_score > best_score:
                            best_score = current_score
                            best_sel = list(sel)
                        improved = True
                        break
        
        # 2-swap moves (swap two selected with two non-selected)
        for _ in range(30):  # Increased 2-swap attempts
            if time.time() >= time_limit:
                break
            idx1 = rng.randint(0, m - 1)
            idx2 = rng.randint(0, m - 1)
            if idx1 == idx2:
                continue
            old1 = sel[idx1]
            old2 = sel[idx2]
            # Pick two random non-selected
            candidates = [x for x in range(total_masks) if x not in sel_set]
            if len(candidates) < 2:
                break
            new1, new2 = rng.sample(candidates, 2)
            
            # Compute delta
            loss = 0
            for j in range(m):
                if j == idx1 or j == idx2:
                    continue
                if (old1 & sel[j]) == old1 or (old1 & sel[j]) == sel[j]:
                    loss += 1
                if (old2 & sel[j]) == old2 or (old2 & sel[j]) == old2:
                    loss += 1
            if (old1 & old2) == old1 or (old1 & old2) == old2:
                loss += 1  # old1-old2 pair
            
            gain = 0
            for j in range(m):
                if j == idx1 or j == idx2:
                    continue
                if (new1 & sel[j]) == new1 or (new1 & sel[j]) == sel[j]:
                    gain += 1
                if (new2 & sel[j]) == new2 or (new2 & sel[j]) == sel[j]:
                    gain += 1
            if (new1 & new2) == new1 or (new1 & new2) == new2:
                gain += 1
            
            delta = gain - loss
            if delta > 0:
                sel_set.remove(old1)
                sel_set.remove(old2)
                sel_set.add(new1)
                sel_set.add(new2)
                sel[idx1] = new1
                sel[idx2] = new2
                current_score += delta
                losses = compute_losses(sel)
                if current_score > best_score:
                    best_score = current_score
                    best_sel = list(sel)
                improved = True
        
        return best_sel, best_score
    
    # Multiple restarts with different initialization strategies
    best_overall_score = -1
    best_overall_sel = None
    
    rng = random.Random(42)
    
    # Strategy 1: Structured chain initialization
    chain_init = chain_initial(rng)
    s1, sc1 = local_search(chain_init, rng, deadline - 10)
    if sc1 > best_overall_score:
        best_overall_score = sc1
        best_overall_sel = s1
    
    # Strategy 2: Greedy initialization
    greedy_init = greedy_initial(rng)
    s2, sc2 = local_search(greedy_init, rng, deadline - 10)
    if sc2 > best_overall_score:
        best_overall_score = sc2
        best_overall_sel = s2
    
    # More restarts with alternating strategies
    restart_count = 0
    while time.time() < deadline - 3 and restart_count < 20:
        if restart_count % 2 == 0:
            init = greedy_initial(rng)
        else:
            init = chain_initial(rng)
        
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
# EVOLVE-BLOCK-END
