import time
import random
from itertools import combinations

def _can_add(x, selected):
    """
    Check if x can be added to selected.
    selected is a list of distinct integers.
    Condition: 
    1. x does not divide sum of any 3 distinct elements in selected.
    2. No element y in selected divides sum of any 3 distinct elements in (selected + [x]) \ {y}.
       This implies y does not divide (x + a + b) for any a, b in selected \ {y}.
    """
    n = len(selected)
    if n == 0:
        return True
    
    # Check 1: x vs existing triplets
    if n >= 3:
        for t in combinations(selected, 3):
            if sum(t) % x == 0:
                return False
    
    # Check 2: existing y vs new triplets involving x
    # New triplets involving x are {x, a, b} where a, b in selected \ {y}
    # So for each y in selected, we check if y divides x + a + b for any a, b in selected \ {y}
    for y in selected:
        # We need pairs (a, b) from selected \ {y}
        # If n == 1, selected \ {y} is empty, no pairs.
        # If n == 2, selected \ {y} has 1 element, no pairs.
        # If n >= 3, we have pairs.
        if n >= 3:
            # Optimization: Pre-filter pairs?
            # For a fixed y and x, we check if (x + a + b) % y == 0
            # a, b are from selected, a != y, b != y, a != b.
            # Let's iterate over indices to avoid list creation overhead in inner loop
            sel_indices = range(n)
            y_idx = selected.index(y) # This is O(n), but n is small. 
                                      # Better to pass indices or use a set?
                                      # Let's assume selected is small enough.
            
            # To speed up, we can iterate a, b
            # a is in selected, a != y
            # b is in selected, b != y, b != a
            
            # We can precompute sums of pairs? No, x changes.
            # But y changes.
            
            # Simple loop
            for i in range(n):
                a = selected[i]
                if a == y:
                    continue
                for j in range(i + 1, n):
                    b = selected[j]
                    if b == y:
                        continue
                    # Check y | (x + a + b)
                    if (x + a + b) % y == 0:
                        return False
    return True

def _is_valid(selected):
    """Full validity check for a set."""
    n = len(selected)
    for i, x in enumerate(selected):
        others = [selected[j] for j in range(n) if j != i]
        if len(others) < 3:
            continue
        for t in combinations(others, 3):
            if sum(t) % x == 0:
                return False
    return True

def _greedy_initialization():
    """
    Greedily build a set from 192 down to 1.
    """
    selected = []
    # Iterate from largest to smallest
    for x in range(192, 0, -1):
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _local_search(initial_set, time_limit):
    """
    Improve the set using local search moves:
    1. Add 1: Try to add any candidate.
    2. Swap 1-for-2: Remove 1, Add 2. Increases size by 1.
    3. Swap 1-for-1: Remove 1, Add 1. Keeps size same, may escape local minima.
    """
    selected = list(initial_set)
    start_time = time.time()
    all_candidates = set(range(1, 193))
    
    # Helper to get candidates not in selected
    def get_candidates():
        return list(all_candidates - set(selected))
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try to add any candidate
        candidates = get_candidates()
        if candidates:
            random.shuffle(candidates)
            for c in candidates:
                if _can_add(c, selected):
                    selected.append(c)
                    improved = True
                    break # Restart loop to try adding more
            
        if not improved:
            # 2. Try 1-for-2 moves (Remove 1, Add 2)
            # This increases cardinality by 1
            # We need to find u in selected, v, w in candidates
            # such that (selected - {u}) + {v, w} is valid.
            
            # To make this efficient, we can try random swaps.
            # Or systematic?
            # Let's try random sampling of u, v, w.
            
            found_move = False
            num_attempts = 100
            for _ in range(num_attempts):
                if time.time() - start_time > time_limit:
                    break
                
                if len(selected) < 2: # Need at least 1 to remove
                    break
                    
                u_idx = random.randint(0, len(selected) - 1)
                u = selected[u_idx]
                
                candidates = get_candidates()
                if len(candidates) < 2:
                    break
                
                # Pick two distinct candidates
                v, w = random.sample(candidates, 2)
                
                # Construct new set
                new_selected = [x for x in selected if x != u]
                new_selected.append(v)
                new_selected.append(w)
                
                # Check validity of new_selected
                # We only need to check constraints involving the new elements v, w
                # and the fact that removing u might have "unblocked" something?
                # No, removing u removes constraints *imposed by* u.
                # But we need to ensure v and w don't violate constraints with existing elements,
                # and existing elements don't violate constraints with v, w.
                # Also, existing elements must not violate constraints with each other (they didn't before).
                # So we just need to check if v and w can be added to (selected - {u}).
                # But wait, adding v and w simultaneously might create a conflict between v and w?
                # No, the constraint is "x does not divide sum of 3 OTHERS".
                # v and w are distinct.
                # The check _can_add(v, S') checks v against S' and S' against v.
                # If we add v then w, we check w against (S' + {v}).
                # This is equivalent to checking the whole set.
                
                # Let's just use _can_add twice for simplicity, assuming order doesn't matter for final validity.
                # Is it possible that v can be added to S', and w can be added to S'+{v}, but the set is invalid?
                # No, if v is valid wrt S' and w is valid wrt S'+{v}, then:
                # 1. v doesn't divide sum of 3 from S'.
                # 2. No y in S' divides sum of 3 from (S' \ {y}) + {v}.
                # 3. w doesn't divide sum of 3 from (S' + {v}).
                # 4. No y in (S' + {v}) divides sum of 3 from ((S' + {v}) \ {y}) + {w}.
                #    This covers y in S' and y=v.
                #    If y in S', it checks sums involving w and 2 from (S' \ {y}) + {v}.
                #    This is correct.
                
                S_prime = [x for x in selected if x != u]
                if _can_add(v, S_prime):
                    S_prime_v = S_prime + [v]
                    if _can_add(w, S_prime_v):
                        # Valid move found
                        selected = S_prime_v + [w]
                        improved = True
                        found_move = True
                        break
            
            if improved:
                continue # Restart loop to try adding more
            
            # 3. Try 1-for-1 swaps (Remove 1, Add 1)
            # This doesn't increase size, but might change the set to allow future 1-for-2.
            # We should be careful not to loop forever.
            # Limit attempts.
            
            found_swap = False
            num_swap_attempts = 50
            for _ in range(num_swap_attempts):
                if time.time() - start_time > time_limit:
                    break
                
                if not selected:
                    break
                    
                u_idx = random.randint(0, len(selected) - 1)
                u = selected[u_idx]
                
                candidates = get_candidates()
                if not candidates:
                    break
                
                v = random.choice(candidates)
                
                S_prime = [x for x in selected if x != u]
                if _can_add(v, S_prime):
                    selected = S_prime + [v]
                    found_swap = True
                    break
            
            if found_swap:
                # We changed the set. Continue loop.
                # To avoid getting stuck in a cycle, we could track visited states, 
                # but with 192 elements, state space is huge. 
                # Randomization usually helps.
                continue
            else:
                # No moves found. 
                # We can try to be more systematic or give up on this branch.
                # For now, just break to avoid infinite loop if no random move works.
                # Or we can try to remove the "most restrictive" element.
                # Let's just break for now and rely on multiple restarts in construct().
                break
        
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s safety margin
    
    best_set = []
    
    # Strategy 1: Greedy from largest to smallest
    greedy_set = _greedy_initialization()
    best_set = greedy_set
    
    # Strategy 2: Local Search on Greedy
    if time.time() < deadline - 10:
        ls_set = _local_search(greedy_set, time_limit=10)
        if len(ls_set) > len(best_set):
            best_set = ls_set
    
    # Strategy 3: Multiple Randomized Greedy + Local Search
    # Try different orderings or starting points
    for i in range(20):
        if time.time() > deadline - 10:
            break
        
        random.seed(i * 100 + 1)
        
        # Randomized Greedy: 
        # Pick a random subset of "large" numbers to start?
        # Or just shuffle the range and pick?
        # Shuffling 192 numbers and picking greedily is O(192 * Check).
        
        # Let's try a "Biased Greedy": 
        # Start with a few random large numbers, then fill greedily.
        
        # Or simpler: Just run Local Search on a slightly perturbed greedy set.
        
        # Perturb Greedy Set: Remove 1-3 random elements from greedy_set, then run Local Search.
        if len(best_set) > 3:
            perturbed = list(best_set)
            # Remove 1-2 random elements
            num_removals = random.randint(1, 2)
            indices = random.sample(range(len(perturbed)), num_removals)
            for idx in sorted(indices, reverse=True):
                perturbed.pop(idx)
            
            # Run local search with limited time
            time_per_run = (deadline - time.time() - 5) / (20 - i)
            if time_per_run < 1:
                time_per_run = 1
            
            ls_set = _local_search(perturbed, time_limit=time_per_run)
            if len(ls_set) > len(best_set):
                best_set = ls_set
    
    # Final Polish: Try to add any remaining elements
    if time.time() < deadline - 2:
        candidates = [x for x in range(192, 0, -1) if x not in best_set]
        for c in candidates:
            if _can_add(c, best_set):
                best_set.append(c)
    
    return best_set
