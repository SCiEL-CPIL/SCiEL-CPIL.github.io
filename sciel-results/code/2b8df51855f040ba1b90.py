import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    N = 1000
    max_rep = 2
    
    # 1. Greedy Initialization
    def greedy_init():
        chosen = []
        counts = {}
        for x in range(1, N + 1):
            new_sums = [x + y for y in chosen] + [2 * x]
            trial = dict(counts)
            valid = True
            for s in new_sums:
                trial[s] = trial.get(s, 0) + 1
                if trial[s] > max_rep:
                    valid = False
                    break
            if valid:
                chosen.append(x)
                counts = trial
        return chosen, counts

    best_set, best_counts = greedy_init()
    best_size = len(best_set)
    
    # Helper to evaluate "quality" of a set (lower is better for tightness)
    # We want to minimize the number of sums that are "full" (count == max_rep)
    # or maximize the "slack"
    def quality(counts):
        # Sum of (count - max_rep)^2 for counts > max_rep is 0 if valid.
        # So we minimize the number of sums with count == max_rep.
        return sum(1 for v in counts.values() if v == max_rep)

    current_set = best_set[:]
    current_counts = best_counts
    current_quality = quality(current_counts)
    
    # Precompute candidates
    all_numbers = set(range(1, N + 1))
    
    # Local Search Loop
    while time.time() < deadline:
        improved = False
        
        # 1. Try Insertions
        candidates = [x for x in all_numbers if x not in current_set]
        random.shuffle(candidates) # Add randomness to escape local traps
        
        best_insert_candidate = None
        best_insert_quality = float('inf')
        best_insert_counts = None
        
        for x in candidates:
            # Quick check: if x is very large, sums might be unique, but we need to check
            new_sums = [x + y for y in current_set] + [2 * x]
            trial = dict(current_counts)
            valid = True
            for s in new_sums:
                trial[s] = trial.get(s, 0) + 1
                if trial[s] > max_rep:
                    valid = False
                    break
            
            if valid:
                q = quality(trial)
                if q < best_insert_quality:
                    best_insert_quality = q
                    best_insert_candidate = x
                    best_insert_counts = trial
        
        if best_insert_candidate is not None:
            current_set.append(best_insert_candidate)
            current_set.sort()
            current_counts = best_insert_counts
            current_quality = best_insert_quality
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
                best_counts = current_counts
            improved = True
        
        # 2. Try Swaps
        if not improved:
            # If we can't insert, try swapping an existing element with a candidate
            # This allows us to replace a "bad" element with a "good" one
            # We limit the number of swaps tried per iteration to save time
            # Try a random subset of swaps
            
            swap_improved = False
            # Sort current set by "badness" (how many sums it participates in with count 2)
            # Or just shuffle to be fast
            shuffled_set = current_set[:]
            random.shuffle(shuffled_set)
            
            # Only try a few swaps per loop to keep it fast
            swaps_to_try = 20
            for _ in range(swaps_to_try):
                x = shuffled_set[_ % len(shuffled_set)]
                # Pick a random candidate
                if not candidates:
                    break
                y = random.choice(candidates)
                
                # Simulate swap: remove x, add y
                # First remove x from counts
                trial = dict(current_counts)
                for s in [x + z for z in current_set if z != x] + [2 * x]:
                    trial[s] -= 1
                    if trial[s] == 0:
                        del trial[s]
                
                # Then add y
                new_sums_y = [y + z for z in current_set if z != x] + [2 * y]
                valid = True
                for s in new_sums_y:
                    trial[s] = trial.get(s, 0) + 1
                    if trial[s] > max_rep:
                        valid = False
                        break
                
                if valid:
                    q = quality(trial)
                    # Accept if it improves quality or is equal (to allow exploration)
                    # To avoid cycles, we might need a tabu list, but simple randomization helps
                    if q <= current_quality:
                        # Perform swap
                        current_set.remove(x)
                        current_set.append(y)
                        current_set.sort()
                        current_counts = trial
                        current_quality = q
                        swap_improved = True
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = current_set[:]
                            best_counts = current_counts
                        break # Restart insertion/swap loop with new state
            
            if swap_improved:
                continue # Restart the while loop with the new state
        
        # 3. If no insertion and no swap, try a "kick" (remove an element)
        if not improved and not swap_improved:
            # Remove the element that is most "constrained"
            # Count how many times each element is involved in a sum with count == max_rep
            involvement = {x: 0 for x in current_set}
            for s, c in current_counts.items():
                if c == max_rep:
                    # Find pairs a,b in current_set such that a+b=s
                    # This is O(|S|^2) which is fine for |S|~60
                    for i in range(len(current_set)):
                        a = current_set[i]
                        b = s - a
                        if b >= a and b in current_set:
                            involvement[a] += 1
                            if b != a:
                                involvement[b] += 1
            
            # Find element with max involvement
            if involvement:
                worst_elem = max(involvement, key=involvement.get)
                # Remove it
                current_set.remove(worst_elem)
                # Update counts
                for s in [worst_elem + z for z in current_set] + [2 * worst_elem]:
                    current_counts[s] -= 1
                    if current_counts[s] == 0:
                        del current_counts[s]
                current_quality = quality(current_counts)
                # This effectively allows us to re-insert a different element in the next iteration
                
    return best_set
# EVOLVE-BLOCK-END
