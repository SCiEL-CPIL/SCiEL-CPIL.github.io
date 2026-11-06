import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    N = 1000
    best_set = []
    best_size = 0
    MAX_SUM = 2 * N  # max possible sum is 2000

    def build_counts(chosen):
        """Build counts array from a list of chosen elements."""
        counts = [0] * (MAX_SUM + 1)
        n = len(chosen)
        for i in range(n):
            x = chosen[i]
            counts[2 * x] += 1
            for j in range(i + 1, n):
                counts[x + chosen[j]] += 1
        return counts

    def is_valid_addition(x, chosen_set, counts):
        if counts[2 * x] >= 2:
            return False
        for y in chosen_set:
            if counts[x + y] >= 2:
                return False
        return True

    def get_tight_count(x, chosen_list, counts):
        """Return tightness score for adding x, or None if invalid."""
        tight = 0
        if counts[2 * x] >= 2:
            return None
        if counts[2 * x] == 1:
            tight += 1
        for y in chosen_list:
            c = counts[x + y]
            if c >= 2:
                return None
            if c == 1:
                tight += 1
        return tight

    def run_greedy(noise=0.0, time_limit=None):
        if time_limit is None:
            time_limit = deadline
        
        chosen = []
        chosen_set = set()
        counts = [0] * (MAX_SUM + 1)
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
                
            counts[2 * best_x] += 1
            for y in chosen:
                counts[best_x + y] += 1
            chosen.append(best_x)
            chosen_set.add(best_x)
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
        new_chosen_set = set(new_chosen)
        
        # Rebuild counts
        new_counts = build_counts(new_chosen)
        
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
                
            new_counts[2 * best_x] += 1
            for y in new_chosen:
                new_counts[best_x + y] += 1
            new_chosen.append(best_x)
            new_chosen_set.add(best_x)
            candidates.remove(best_x)
            
        return new_chosen, new_counts

    def local_search(initial_set, time_limit):
        chosen = list(initial_set)
        chosen_set = set(chosen)
        
        # Build counts
        counts = build_counts(chosen)
        
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
                if is_valid_addition(x, chosen_set, counts):
                    counts[2 * x] += 1
                    for y in chosen:
                        counts[x + y] += 1
                    chosen.append(x)
                    chosen_set.add(x)
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
            for attempt in range(15):
                if time.time() > time_limit:
                    break
                    
                x_to_try = random.choice(list(candidates))
                chosen_list = list(chosen)
                random.shuffle(chosen_list)
                
                for y in chosen_list[:30]:  # Limit swaps per x
                    # Check if removing y and adding x is valid
                    # Temporarily remove y
                    counts[2 * y] -= 1
                    for z in chosen:
                        if z != y:
                            counts[y + z] -= 1
                    
                    reduced_chosen_set = set(z for z in chosen if z != y)
                    if is_valid_addition(x, reduced_chosen_set, counts):
                        # Add x
                        counts[2 * x] += 1
                        for z in reduced_chosen_set:
                            counts[x + z] += 1
                            
                        # Update sets
                        chosen.remove(y)
                        chosen_set.discard(y)
                        chosen.append(x)
                        chosen_set.add(x)
                        candidates.discard(x)
                        candidates.add(y)
                        
                        swap_success = True
                        break
                    else:
                        # Revert: re-add y
                        counts[2 * y] += 1
                        for z in reduced_chosen_set:
                            counts[y + z] += 1
                
                if swap_success:
                    break
            
            if not swap_success:
                # Phase 3: Kick - remove some elements and refill
                if time.time() > time_limit:
                    break
                    
                kick_size = random.choice([1, 2, 3])
                chosen, counts = kick_and_refill(chosen, counts, kick_size, time_limit)
                chosen_set = set(chosen)
                candidates = set(range(1, N + 1))
                for x in chosen:
                    candidates.discard(x)
                    
                # Continue loop to try adding more
        
        return chosen

    def final_polish(chosen, time_limit):
        """One last pass: try to add any remaining candidate."""
        chosen_set = set(chosen)
        counts = build_counts(chosen)
        candidates = set(range(1, N + 1))
        for x in chosen:
            candidates.discard(x)
        
        if not candidates:
            return chosen
        
        cand_list = list(candidates)
        random.shuffle(cand_list)
        
        for x in cand_list:
            if time.time() > time_limit:
                break
            if is_valid_addition(x, chosen_set, counts):
                counts[2 * x] += 1
                for y in chosen:
                    counts[x + y] += 1
                chosen.append(x)
                chosen_set.add(x)
        
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
            
    # Final polish
    if time.time() < deadline - 1:
        best_set = final_polish(best_set, deadline - 0.5)
        best_size = len(best_set)
    
    return best_set
