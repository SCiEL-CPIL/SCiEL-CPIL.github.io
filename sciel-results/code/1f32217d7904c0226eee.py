import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    best_family = None
    best_score = -1
    
    # Try different universe sizes k
    for k in [6, 7, 8]:
        if time.time() > deadline:
            break
            
        n_masks = 1 << k
        all_masks = list(range(n_masks))
        
        # Run multiple restarts of Greedy + Local Search
        for restart in range(20): # 20 restarts per k is fast
            if time.time() > deadline:
                break
                
            # 1. Greedy Construction
            family = []
            pool = set(all_masks)
            
            # Optional: Randomize the initial order or pick randomly from top candidates
            # To add diversity, we can pick the first mask randomly from the top 5 candidates
            for _ in range(40):
                if not pool:
                    break
                best_gain = -1
                best_candidates = []
                
                for m in pool:
                    gain = 0
                    for s in family:
                        if (m | s) == m or (m | s) == s:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [m]
                    elif gain == best_gain:
                        best_candidates.append(m)
                
                # Pick randomly from best candidates to diversify restarts
                if best_candidates:
                    chosen = random.choice(best_candidates)
                else:
                    chosen = pool.pop()
                
                family.append(chosen)
                pool.remove(chosen)
            
            # 2. Local Search (Swap)
            current_score = _score(family)
            improved = True
            while improved and time.time() < deadline:
                improved = False
                family_set = set(family)
                # Create a list of available masks not in family
                # To speed up, we can just iterate all_masks and check membership
                # But checking membership in a set is O(1)
                
                for i in range(len(family)):
                    old_m = family[i]
                    # Calculate comparables of old_m with rest of family
                    rest = [s for j, s in enumerate(family) if j != i]
                    comp_old = sum(1 for s in rest if (old_m | s) == old_m or (old_m | s) == s)
                    
                    for new_m in all_masks:
                        if new_m in family_set:
                            continue
                        
                        # Calculate comparables of new_m with rest of family
                        comp_new = sum(1 for s in rest if (new_m | s) == new_m or (new_m | s) == s)
                        
                        if comp_new > comp_old:
                            # Swap
                            family[i] = new_m
                            family_set.remove(old_m)
                            family_set.add(new_m)
                            current_score += (comp_new - comp_old)
                            improved = True
                            break # Restart the scan for this i or whole loop? 
                            # Breaking inner loop and restarting outer is standard for simple local search
                            # But we need to update 'rest' and 'comp_old' if we continue? 
                            # It's easier to just restart the whole 'while improved' loop or 
                            # just mark improved and break out to re-evaluate from scratch.
                            # Given the speed, breaking out and restarting the while loop is safe.
                            break
                    
                    if improved:
                        break
            
            if current_score > best_score:
                best_score = current_score
                best_family = family[:]

    # Convert best_family to list of lists
    if best_family is None:
        # Fallback to seed behavior
        return [[b for b in range(12) if (mask >> b) & 1] for mask in range(40)]
    
    result = []
    for mask in best_family:
        result.append([b for b in range(12) if (mask >> b) & 1])
        
    return result

def _score(family):
    count = 0
    n = len(family)
    for i in range(n):
        for j in range(i + 1, n):
            a = family[i]
            b = family[j]
            if (a | b) == a or (a | b) == b:
                count += 1
    return count
