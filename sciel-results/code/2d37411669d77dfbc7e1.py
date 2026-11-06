import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    N = 1000
    best_set = []
    best_size = 0
    
    def is_valid_addition(x, chosen, counts):
        if counts.get(2*x, 0) >= 2:
            return False
        for y in chosen:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        return True

    def get_tight_count(x, chosen, counts):
        """Return tightness score for adding x, or None if invalid."""
        tight = 0
        if counts.get(2*x, 0) >= 2:
            return None
        if counts.get(2*x, 0) == 1:
            tight += 1
        for y in chosen:
            s = x + y
            c = counts.get(s, 0)
            if c >= 2:
                return None
            if c == 1:
                tight += 1
        return tight

    def run_greedy(noise=0.0, time_limit=None):
        if time_limit is None:
            time_limit = deadline
        
        chosen = []
        counts = defaultdict(int)
        candidates = set(range(1, N + 1))
        
        while candidates:
            if time.time() > time_limit:
                break
            
            best_x = None
            best_score = float('inf')
            
            for x in candidates:
                score = get_tight_count(x, chosen, counts)
                if score is None:
                    continue
                total = score + (random.random() * noise if noise > 0 else 0)
                if total < best_score:
                    best_score = total
                    best_x = x
            
            if best_x is None:
                break
                
            counts[2*best_x] = counts.get(2*best_x, 0) + 1
            for y in chosen:
                s = best_x + y
                counts[s] = counts.get(s, 0) + 1
            chosen.append(best_x)
            candidates.remove(best_x)
            
        return chosen, counts

    def kick_and_refill(chosen, counts, kick_size, time_limit):
        """Remove kick_size random elements, then greedily refill."""
        if len(chosen) <= kick_size:
            return chosen, counts
        
        # Pick random elements to remove
        to_remove = random.sample(chosen, kick_size)
        remove_set = set(to_remove)
        
        # Remove them
        new_chosen = [x for x in chosen if x not in remove_set]
        
        # Rebuild counts
        new_counts = defaultdict(int)
        for x in new_chosen:
            new_counts[2*x] = new_counts.get(2*x, 0) + 1
        for i in range(len(new_chosen)):
            for j in range(i+1, len(new_chosen)):
                s = new_chosen[i] + new_chosen[j]
                new_counts[s] = new_counts.get(s, 0) + 1
        
        # Now greedily add back elements
        candidates = set(range(1, N + 1))
        for x in new_chosen:
            candidates.discard(x)
        
        while candidates:
            if time.time() > time_limit:
                break
            
            best_x = None
            best_score = float('inf')
            
            for x in candidates:
                score = get_tight_count(x, new_chosen, new_counts)
                if score is None:
                    continue
                total = score + random.random() * 0.5  # Small noise
                if total < best_score:
                    best_score = total
                    best_x = x
            
            if best_x is None:
                break
                
            new_counts[2*best_x] = new_counts.get(2*best_x, 0) + 1
            for y in new_chosen:
                s = best_x + y
                new_counts[s] = new_counts.get(s, 0) + 1
            new_chosen.append(best_x)
            candidates.remove(best_x)
            
        return new_chosen, new_counts

    def local_search(initial_set, time_limit):
        chosen = list(initial_set)
        
        # Build counts
        counts = defaultdict(int)
        for x in chosen:
            counts[2*x] = counts.get(2*x, 0) + 1
        for i in range(len(chosen)):
            for j in range(i+1, len(chosen)):
                s = chosen[i] + chosen[j]
                counts[s] = counts.get(s, 0) + 1
                
        candidates = set(range(1, N + 1))
        for x in chosen:
            candidates.discard(x)
            
        if not candidates:
            return chosen
            
        # Phase 1: Try to add elements
        while time.time() < time_limit:
            cand_list = list(candidates)
            random.shuffle(cand_list)
            
            added_any = False
            for x in cand_list:
                if time.time() > time_limit:
                    break
                if is_valid_addition(x, chosen, counts):
                    counts[2*x] = counts.get(2*x, 0) + 1
                    for y in chosen:
                        s = x + y
                        counts[s] = counts.get(s, 0) + 1
                    chosen.append(x)
                    candidates.discard(x)
                    added_any = True
                    break
            
            if added_any:
                continue
            
            # Phase 2: Try swaps
            if len(candidates) == 0:
                break
                
            swap_success = False
            # Try multiple swap attempts
            for attempt in range(10):
                if time.time() > time_limit:
                    break
                    
                x_to_try = random.choice(list(candidates))
                chosen_list = list(chosen)
                random.shuffle(chosen_list)
                
                for y in chosen_list[:20]:  # Limit swaps per x
                    # Check if removing y and adding x is valid
                    old_counts = dict(counts)
                    
                    # Remove y from counts
                    counts[2*y] -= 1
                    if counts[2*y] == 0:
                        del counts[2*y]
                    for z in chosen:
                        if z != y:
                            s = y + z
                            counts[s] -= 1
                            if counts[s] == 0:
                                del counts[s]
                    
                    # Check if x can be added to (chosen - {y})
                    reduced_chosen = [z for z in chosen if z != y]
                    if is_valid_addition(x, reduced_chosen, counts):
                        # Add x
                        counts[2*x] = counts.get(2*x, 0) + 1
                        for z in reduced_chosen:
                            s = x + z
                            counts[s] = counts.get(s, 0) + 1
                            
                        # Update sets
                        chosen.remove(y)
                        chosen.append(x)
                        candidates.discard(x)
                        candidates.add(y)
                        
                        swap_success = True
                        break
                    else:
                        # Revert
                        counts.clear()
                        counts.update(old_counts)
                
                if swap_success:
                    break
            
            if not swap_success:
                # Phase 3: Kick - remove some elements and refill
                if time.time() > time_limit:
                    break
                    
                kick_size = random.choice([1, 2, 3])
                chosen, counts = kick_and_refill(chosen, counts, kick_size, time_limit)
                candidates = set(range(1, N + 1))
                for x in chosen:
                    candidates.discard(x)
                    
                # Continue loop to try adding more
        
        return chosen

    # Main loop
    round_num = 0
    while time.time() < deadline:
        round_num += 1
        remaining = deadline - time.time()
        if remaining < 2:
            break
        
        # Vary noise for diversity
        noise = random.uniform(0, 3.0)
        
        # Greedy phase: allocate ~20% of remaining time
        greedy_time = min(remaining * 0.2, 8.0)
        baseline, _ = run_greedy(noise=noise, time_limit=time.time() + greedy_time)
        
        if len(baseline) > best_size:
            best_size = len(baseline)
            best_set = list(baseline)
        
        # Local search phase: allocate ~60% of remaining time
        ls_time = min(remaining * 0.6, 30.0)
        start_ls_set = best_set if len(best_set) > len(baseline) else baseline
        refined = local_search(start_ls_set, time.time() + ls_time)
        
        if len(refined) > best_size:
            best_size = len(refined)
            best_set = list(refined)
        
        # If we have good time left, do another greedy with different noise
        if time.time() < deadline - 10:
            noise2 = random.uniform(0, 5.0)
            baseline2, _ = run_greedy(noise=noise2, time_limit=min(deadline - 5, time.time() + 5))
            if len(baseline2) > best_size:
                best_size = len(baseline2)
                best_set = list(baseline2)
            
            # Quick local search on the new baseline
            if time.time() < deadline - 3:
                refined2 = local_search(best_set, time.time() + min(remaining * 0.1, 5.0))
                if len(refined2) > best_size:
                    best_size = len(refined2)
                    best_set = list(refined2)
            
    return best_set
