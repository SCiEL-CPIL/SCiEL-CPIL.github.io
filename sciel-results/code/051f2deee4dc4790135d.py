import math
import random
import time

def score(U):
    if not U:
        return 0
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return 1.0
    return 1 + math.log(ratio) / math.log(2 * max_u + 1)

def score_fast(U, sums_cache, diffs_cache):
    """Score using pre-computed sum/diff sets. Returns (score, sums_set, diffs_set)."""
    if not U:
        return 0, set(), set()
    if 0 not in U:
        return 0, set(), set()
    max_u = max(U)
    if max_u == 0:
        return 1.0, set(), set()
    
    if sums_cache is not None and diffs_cache is not None:
        sums = sums_cache
        diffs = diffs_cache
    else:
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
    
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return 1.0, sums, diffs
    return 1 + math.log(ratio) / math.log(2 * max_u + 1), sums, diffs

def incremental_score_add(U, new_elem, sums_cache, diffs_cache):
    """
    Compute score after adding new_elem to U, using existing sum/diff caches.
    Returns (new_score, new_sums, new_diffs).
    """
    new_sums = set(sums_cache)
    new_diffs = set(diffs_cache)
    
    for a in U:
        new_sums.add(a + new_elem)
        new_sums.add(new_elem + a)
        new_diffs.add(a - new_elem)
        new_diffs.add(new_elem - a)
    
    new_set = list(U) + [new_elem]
    new_set.sort()
    max_u = max(new_set)
    
    if max_u == 0:
        return 1.0, new_sums, new_diffs
    
    ratio = len(new_diffs) / len(new_sums)
    if ratio <= 1:
        return 1.0, new_sums, new_diffs
    return 1 + math.log(ratio) / math.log(2 * max_u + 1), new_sums, new_diffs

def incremental_score_remove(U, remove_elem, sums_cache, diffs_cache):
    """
    Compute score after removing remove_elem from U.
    Returns (new_score, new_sums, new_diffs) or None if can't determine incrementally.
    """
    # For removal, it's complex to do incrementally since we need to know
    # which sums/diffs were unique to this element. Fall back to full recompute.
    new_set = [x for x in U if x != remove_elem]
    if not new_set or 0 not in new_set:
        return None
    new_set.sort()
    s = score(new_set)
    # Recompute caches
    new_sums = set()
    new_diffs = set()
    for a in new_set:
        for b in new_set:
            new_sums.add(a + b)
            new_diffs.add(a - b)
    return s, new_sums, new_diffs

def incremental_score_shift(U, old_elem, new_elem, sums_cache, diffs_cache):
    """
    Compute score after replacing old_elem with new_elem in U.
    Returns (new_score, new_sums, new_diffs).
    """
    new_set = [x for x in U if x != old_elem]
    new_set.append(new_elem)
    new_set.sort()
    
    if not new_set or 0 not in new_set:
        return None
    
    # Full recompute for shift (complex to do incrementally)
    new_sums = set()
    new_diffs = set()
    for a in new_set:
        for b in new_set:
            new_sums.add(a + b)
            new_diffs.add(a - b)
    
    max_u = max(new_set)
    if max_u == 0:
        return 1.0, new_sums, new_diffs
    
    ratio = len(new_diffs) / len(new_sums)
    if ratio <= 1:
        return 1.0, new_sums, new_diffs
    return 1 + math.log(ratio) / math.log(2 * max_u + 1), new_sums, new_diffs

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with a good seed
    best_set = [0, 1, 3, 7]
    best_score = score(best_set)
    
    # Initialize caches
    current_set = list(best_set)
    current_sums = set()
    current_diffs = set()
    for a in current_set:
        for b in current_set:
            current_sums.add(a + b)
            current_diffs.add(a - b)
    current_score = best_score
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Reset to best and try a different strategy
            current_set = list(best_set)
            current_sums = set()
            current_diffs = set()
            for a in current_set:
                for b in current_set:
                    current_sums.add(a + b)
                    current_diffs.add(a - b)
            current_score = best_score
            no_improve_count = 0
            max_no_improve += 20
        
        # Generate neighbor with diff-guided bias
        neighbor = list(current_set)
        move_type = random.random()
        
        if move_type < 0.5:
            # Add: prioritize elements that create many new diffs relative to new sums
            max_val = max(neighbor)
            # Candidate range: up to max_val + 10 for exploration
            upper = max_val + 10
            candidates = [i for i in range(1, upper + 1) if i not in neighbor]
            if candidates:
                # Score each candidate by (new_diffs - new_sums) / log(2*new_max+1)
                best_candidate = None
                best_candidate_score = -float('inf')
                
                # Limit candidates to keep it fast
                if len(candidates) > 30:
                    # Sample a subset
                    random.shuffle(candidates)
                    candidates = candidates[:30]
                
                for cand in candidates:
                    new_sums_count = len(current_sums)
                    new_diffs_count = len(current_diffs)
                    
                    # Count new sums and diffs
                    new_sum_vals = set()
                    new_diff_vals = set()
                    for a in current_set:
                        s1 = a + cand
                        s2 = cand + a
                        d1 = a - cand
                        d2 = cand - a
                        if s1 not in current_sums:
                            new_sum_vals.add(s1)
                        if s2 not in current_sums:
                            new_sum_vals.add(s2)
                        if d1 not in current_diffs:
                            new_diff_vals.add(d1)
                        if d2 not in current_diffs:
                            new_diff_vals.add(d2)
                    
                    ns = len(current_sums) + len(new_sum_vals)
                    nd = len(current_diffs) + len(new_diff_vals)
                    
                    if ns == 0:
                        continue
                    ratio = nd / ns
                    if ratio <= 1:
                        continue
                    
                    new_max = max(max_val, cand)
                    s = 1 + math.log(ratio) / math.log(2 * new_max + 1)
                    
                    if s > best_candidate_score:
                        best_candidate_score = s
                        best_candidate = cand
                
                if best_candidate is not None:
                    neighbor.append(best_candidate)
                elif candidates:
                    neighbor.append(random.choice(candidates))
        elif move_type < 0.7:
            # Remove
            if len(neighbor) > 1:
                rem = random.choice([x for x in neighbor if x != 0])
                neighbor.remove(rem)
        else:
            # Shift
            if len(neighbor) > 1:
                idx = random.choice([i for i, x in enumerate(neighbor) if x != 0])
                old_val = neighbor[idx]
                new_val = random.randint(1, max(neighbor) + 10)
                if new_val not in neighbor:
                    neighbor[idx] = new_val
        
        neighbor.sort()
        
        # Compute new score
        if len(neighbor) == len(current_set) + 1:
            # It's an add
            added = [x for x in neighbor if x not in current_set]
            if len(added) == 1:
                result = incremental_score_add(current_set, added[0], current_sums, current_diffs)
                if result is not None:
                    new_score, current_sums, current_diffs = result
                else:
                    new_score = score(neighbor)
                    current_sums = set()
                    current_diffs = set()
                    for a in neighbor:
                        for b in neighbor:
                            current_sums.add(a + b)
                            current_diffs.add(a - b)
            else:
                new_score = score(neighbor)
                current_sums = set()
                current_diffs = set()
                for a in neighbor:
                    for b in neighbor:
                        current_sums.add(a + b)
                        current_diffs.add(a - b)
        elif len(neighbor) == len(current_set) - 1:
            # It's a remove
            result = incremental_score_remove(current_set, [x for x in current_set if x not in neighbor][0], current_sums, current_diffs)
            if result is not None:
                new_score, current_sums, current_diffs = result
            else:
                new_score = score(neighbor)
                current_sums = set()
                current_diffs = set()
                for a in neighbor:
                    for b in neighbor:
                        current_sums.add(a + b)
                        current_diffs.add(a - b)
        else:
            # Shift or other
            if len(neighbor) == len(current_set):
                removed = [x for x in current_set if x not in neighbor]
                added = [x for x in neighbor if x not in current_set]
                if len(removed) == 1 and len(added) == 1:
                    result = incremental_score_shift(current_set, removed[0], added[0], current_sums, current_diffs)
                    if result is not None:
                        new_score, current_sums, current_diffs = result
                    else:
                        new_score = score(neighbor)
                        current_sums = set()
                        current_diffs = set()
                        for a in neighbor:
                            for b in neighbor:
                                current_sums.add(a + b)
                                current_diffs.add(a - b)
                else:
                    new_score = score(neighbor)
                    current_sums = set()
                    current_diffs = set()
                    for a in neighbor:
                        for b in neighbor:
                            current_sums.add(a + b)
                            current_diffs.add(a - b)
            else:
                new_score = score(neighbor)
                current_sums = set()
                current_diffs = set()
                for a in neighbor:
                    for b in neighbor:
                        current_sums.add(a + b)
                        current_diffs.add(a - b)
        
        if new_score > current_score:
            current_set = neighbor
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_set = list(current_set)
                best_score = current_score
        else:
            no_improve_count += 1
            # Accept worse moves with decaying probability
            accept_prob = 0.05 + 0.05 * math.exp(-no_improve_count / 30.0)
            if random.random() < accept_prob:
                current_set = neighbor
                current_score = new_score

    return best_set
