import time
import math
import random

def _score_of_set(U):
    """
    Compute the score for a set U of nonnegative integers.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    """
    if not U or 0 not in U:
        return 0.0
    max_el = max(U)
    if max_el == 0:
        return 0.0
    
    # For small sets, direct set operations are fast and exact
    # If U is large, this might be slow, but we expect optimal U to be small
    sums = set()
    diffs = set()
    
    # Optimization: iterate only once
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
            
    sum_size = len(sums)
    diff_size = len(diffs)
    
    if sum_size == 0:
        return 0.0
        
    ratio = diff_size / sum_size
    if ratio <= 1.0:
        # If ratio <= 1, log is <= 0, score <= 1. 
        # We want to maximize, so we prefer ratio > 1.
        # But we still calculate to be safe.
        pass
        
    denom = math.log(2 * max_el + 1)
    if denom <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial candidates: Small Golomb rulers and structured sets
    # Golomb rulers: all differences distinct
    # Sidon sets: all sums distinct (usually implies larger sumset)
    # We want |U-U| > |U+U|, so we prefer sets where differences are diverse 
    # and sums are somewhat compact, or just generally "spread out" differences.
    
    candidates = [
        [0, 1, 3],
        [0, 1, 4],
        [0, 1, 3, 7],
        [0, 1, 4, 9],
        [0, 1, 4, 10],
        [0, 1, 4, 9, 15],
        [0, 1, 4, 10, 18],
        [0, 2, 5, 11, 17], # Golomb
        [0, 1, 3, 7, 12, 18], # Golomb
        [0, 1, 4, 10, 18, 23], # Golomb
    ]
    
    best_set = []
    best_score = 0.0
    
    for cand in candidates:
        s = _score_of_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
            
    # Local search
    current_set = set(best_set)
    current_score = best_score
    
    # We will search for improvements by modifying the set
    # Given the small expected size, we can afford to be thorough
    
    max_iterations = 10000
    no_improve_count = 0
    
    while time.time() < deadline and no_improve_count < 200:
        improved = False
        current_list = sorted(current_set)
        
        # 1. Try adding a new element
        if len(current_set) < 12: # Limit size to keep score high
            # Try adding elements in a reasonable range
            max_el = max(current_set) if current_set else 0
            # Search range: up to 2*max_el or so
            search_max = max(20, 2 * max_el)
            # Sample candidate positions
            candidates_to_add = [x for x in range(1, search_max + 1) if x not in current_set]
            random.shuffle(candidates_to_add)
            
            for new_el in candidates_to_add[:50]: # Try up to 50 random additions
                new_set = current_set | {new_el}
                s = _score_of_set(list(new_set))
                if s > current_score + 1e-9:
                    current_set = new_set
                    current_score = s
                    improved = True
                    break
        else:
            # If set is large, try removing
            removable = [x for x in current_set if x != 0]
            if removable:
                random.shuffle(removable)
                for x in removable[:20]:
                    new_set = current_set - {x}
                    s = _score_of_set(list(new_set))
                    if s > current_score + 1e-9:
                        current_set = new_set
                        current_score = s
                        improved = True
                        break

        # 2. Try swapping an element
        if not improved and len(current_set) > 2:
            removable = [x for x in current_set if x != 0]
            if removable:
                max_el = max(current_set)
                search_max = max(20, 2 * max_el)
                candidates_to_add = [x for x in range(1, search_max + 1) if x not in current_set]
                
                for x_rm in removable[:10]:
                    # Try replacing x_rm with a random other value
                    if not candidates_to_add: continue
                    for x_add in random.sample(candidates_to_add, min(10, len(candidates_to_add))):
                        new_set = (current_set - {x_rm}) | {x_add}
                        s = _score_of_set(list(new_set))
                        if s > current_score + 1e-9:
                            current_set = new_set
                            current_score = s
                            improved = True
                            break
                    if improved:
                        break

        # 3. Try local moves (shift an element by +/- 1)
        if not improved:
            movable = [x for x in current_set if x != 0]
            if movable:
                random.shuffle(movable)
                for x in movable[:20]:
                    for delta in [-2, -1, 1, 2]:
                        new_x = x + delta
                        if new_x > 0 and new_x not in current_set:
                            new_set = (current_set - {x}) | {new_x}
                            s = _score_of_set(list(new_set))
                            if s > current_score + 1e-9:
                                current_set = new_set
                                current_score = s
                                improved = True
                                break
                    if improved:
                        break

        if improved:
            no_improve_count = 0
            # Update global best
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
        else:
            no_improve_count += 1
            
            # If stuck, try a restart with a different structure occasionally
            if no_improve_count % 50 == 0 and no_improve_count < 200:
                # Generate a random Golomb-like set
                # Simple greedy Golomb ruler construction
                size = random.randint(3, 8)
                U = [0]
                for _ in range(size - 1):
                    # Find next smallest number that doesn't create duplicate differences
                    found = False
                    for cand in range(1, 100):
                        if cand in U:
                            continue
                        # Check differences
                        valid = True
                        for u in U:
                            d = cand - u
                            # If d or -d is already in the diff set (implicitly), it's a collision
                            # We just check if any pair in U has diff d
                            # This is O(N^2) per check, but N is small
                            for v in U:
                                if u - v == d or v - u == d:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            U.append(cand)
                            found = True
                            break
                    if not found:
                        break
                
                if len(U) >= 3:
                    s = _score_of_set(U)
                    if s > current_score:
                        current_set = set(U)
                        current_score = s
                        no_improve_count = 0

    # Final check
    if current_score > best_score:
        best_set = list(current_set)
        
    # Ensure 0 is in set and sorted
    best_set = sorted(set(best_set))
    if 0 not in best_set:
        best_set.insert(0, 0)
        
    return best_set
