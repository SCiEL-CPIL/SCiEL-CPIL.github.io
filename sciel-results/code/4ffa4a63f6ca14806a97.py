"""B2[2]-set construction for {1..1000} combining fast greedy, in-place local search, and conflict-aware filling."""
import time
import random

N = 1000
TIME_LIMIT = 115  # seconds; leaves margin before the 150 s kill

def _valid_insertion(counts, x, chosen_set):
    """
    Check if x can be inserted into chosen_set without violating B2[2] condition.
    Returns a list of (sum, delta) pairs if valid, else None.
    """
    # New sums are x + y for all y in chosen_set, and 2*x.
    # We need to check if any of these sums currently have count >= 2.
    # If count is 0 or 1, it becomes 1 or 2, which is valid.
    # If count is 2, it becomes 3, which is invalid.
    
    # Check 2*x first
    if counts.get(2 * x, 0) >= 2:
        return None
    
    # Check x + y
    for y in chosen_set:
        s = x + y
        if counts.get(s, 0) >= 2:
            return None
            
    # If we get here, it's valid. Construct the delta list.
    # Note: We don't need to deduplicate sums in the delta list if we just apply them,
    # but for rollback efficiency, it's better to know exactly which sums change and by how much.
    # However, since each pair (x,y) is unique, and x is new, all sums x+y are distinct from each other?
    # No, x+y1 could equal x+y2 only if y1==y2. So sums x+y are distinct.
    # 2*x could equal x+y if x==y, but y is in chosen_set and x is not (we check x in chosen_set before calling).
    # So 2*x is distinct from all x+y.
    # Thus, each sum gets exactly +1.
    
    deltas = []
    deltas.append((2 * x, 1))
    for y in chosen_set:
        deltas.append((x + y, 1))
        
    return deltas

def _apply_deltas(counts, deltas, sign=1):
    """Apply a list of (sum, delta) pairs to the counts dict."""
    for s, d in deltas:
        counts[s] = counts.get(s, 0) + d * sign
        if counts[s] == 0:
            # Optional: cleanup zero entries to keep dict small, but might be slower due to del overhead.
            # For now, keep them or let them be. 0 is fine for .get checks.
            pass

def _greedy_build(order, time_limit, start_time):
    """
    Greedy construction with fast insertion check.
    """
    chosen = []
    chosen_set = set()
    counts = {}
    
    for x in order:
        if (time.time() - start_time) > time_limit:
            break
        if x in chosen_set:
            continue
            
        deltas = _valid_insertion(counts, x, chosen_set)
        if deltas is not None:
            _apply_deltas(counts, deltas, 1)
            chosen.append(x)
            chosen_set.add(x)
            
    return chosen, counts

def _local_search(initial_set, initial_counts, time_limit, start_time, seed=42):
    """
    Local search using in-place swaps to avoid dict copies.
    """
    best = sorted(initial_set)
    best_set = set(best)
    counts = dict(initial_counts) # Copy once at start
    best_size = len(best)
    
    # Candidate pool: all numbers not in best
    candidates = [x for x in range(1, N + 1) if x not in best_set]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    
    # To allow swapping back and forth, we need to track which candidates are "active" for swapping in.
    # Actually, a simple swap move is: remove i from best, add j from candidates.
    # If we accept, i goes to candidates, j goes to best.
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Occasionally reshuffle candidates to avoid bias
        if rng.random() < 0.05:
            rng.shuffle(candidates)
            
        # Iterate through candidates to find a valid swap
        # We iterate over best elements to remove
        # To be efficient, we can pick a random element from best to remove, and try to find a candidate to add.
        # Or iterate through candidates and try to swap with a random element from best?
        # Standard approach: for each new_elem in candidates, try to swap with each old_elem in best.
        # But that's O(|C| * |B|) checks.
        
        # Let's try a random restart-like approach within the loop:
        # Pick a random candidate. Try to swap it with a random element in best.
        # If valid, accept.
        
        # However, systematic search is often better for local optima.
        # Let's stick to the structure but optimize the check.
        
        for new_elem in candidates:
            if (time.time() - start_time) > time_limit:
                return best, counts
            
            if new_elem in best_set:
                continue
                
            # Try to swap new_elem with each element in best
            # To speed up, we can randomize the order of best elements we try to remove
            indices = list(range(len(best)))
            rng.shuffle(indices)
            
            for idx in indices:
                old_elem = best[idx]
                
                # Calculate deltas for removing old_elem and adding new_elem
                # Remove old_elem:
                # Sums involving old_elem: 2*old_elem, and old_elem + y for y in best (y != old_elem)
                # Add new_elem:
                # Sums involving new_elem: 2*new_elem, and new_elem + y for y in best (y != old_elem)
                
                # We need to check if the resulting counts are valid.
                # Instead of applying and checking, we can calculate the "net" effect on specific sums.
                # But it's easier to just apply, check, and rollback if invalid.
                
                # Collect sums to be modified
                # Note: Some sums might be affected by both removal and addition.
                # e.g., old_elem + y1 == new_elem + y2.
                
                # Let's build a temporary delta dict for this specific swap
                swap_deltas = {}
                
                # Subtract old_elem contributions
                swap_deltas[2 * old_elem] = swap_deltas.get(2 * old_elem, 0) - 1
                for y in best:
                    if y == old_elem:
                        continue
                    s = old_elem + y
                    swap_deltas[s] = swap_deltas.get(s, 0) - 1
                    
                # Add new_elem contributions
                swap_deltas[2 * new_elem] = swap_deltas.get(2 * new_elem, 0) + 1
                for y in best:
                    if y == old_elem:
                        continue
                    s = new_elem + y
                    swap_deltas[s] = swap_deltas.get(s, 0) + 1

                # Check validity
                valid = True
                for s, delta in swap_deltas.items():
                    current_count = counts.get(s, 0)
                    new_count = current_count + delta
                    if new_count > 2:
                        valid = False
                        break
                    # Also new_count should be >= 0, which is guaranteed by the logic
                    # (we only subtract what exists, and add what didn't exist in that specific pair context? 
                    # Wait, if old_elem + y == new_elem + z, and both are in best, the net delta might be 0.
                    # If we subtract 1 and add 1, net is 0.
                    # If we subtract 1 and don't add, count decreases.
                    # Since old_elem was in best, its contributions were in counts.
                    # So current_count >= delta (if delta is negative).
                    # So new_count >= 0 is safe.)
                
                if valid:
                    # Apply the swap
                    for s, delta in swap_deltas.items():
                        counts[s] = counts.get(s, 0) + delta
                        if counts[s] == 0:
                            # Keep 0s for simplicity, .get handles it
                            pass
                    
                    # Update best list and sets
                    best[idx] = new_elem
                    best.sort() # Keep sorted for consistency, though not strictly needed for set logic
                    best_set.remove(old_elem)
                    best_set.add(new_elem)
                    
                    # Update candidates list
                    candidates.remove(new_elem)
                    candidates.append(old_elem)
                    
                    improved = True
                    break # Move to next new_elem
                
                if (time.time() - start_time) > time_limit:
                    break
            
            if (time.time() - start_time) > time_limit:
                break
    
    return best, counts

def _fill_remaining(best_set, counts, time_limit, start_time):
    """
    Try to add remaining elements in descending order (conflict-aware filling).
    """
    chosen = sorted(best_set)
    chosen_set = set(chosen)
    
    for x in range(N, 0, -1):
        if (time.time() - start_time) > time_limit:
            break
        if x in chosen_set:
            continue
        
        deltas = _valid_insertion(counts, x, chosen_set)
        if deltas is not None:
            _apply_deltas(counts, deltas, 1)
            chosen.append(x)
            chosen_set.add(x)
    
    return chosen, counts

def construct():
    start_time = time.time()
    
    best = []
    best_counts = {}
    best_size = 0
    
    # Step 1: Multiple greedy constructions with different orderings
    # Use fast insertion check
    
    # Try increasing order
    greedy1, counts1 = _greedy_build(list(range(1, N + 1)), TIME_LIMIT, start_time)
    if len(greedy1) > best_size:
        best = greedy1
        best_counts = counts1
        best_size = len(greedy1)
    
    # Try decreasing order
    if (time.time() - start_time) < TIME_LIMIT - 20:
        greedy2, counts2 = _greedy_build(list(range(N, 0, -1)), TIME_LIMIT, start_time)
        if len(greedy2) > best_size:
            best = greedy2
            best_counts = counts2
            best_size = len(greedy2)
    
    # Try randomized orders with large bias
    rng = random.Random(123)
    for _ in range(10):
        if (time.time() - start_time) > TIME_LIMIT - 30:
            break
        order = list(range(1, N + 1))
        # Mix of random and descending bias
        if rng.random() < 0.5:
            rng.shuffle(order)
            # Sort by a key that favors larger numbers but keeps some randomness
            # key = -x + noise
            order.sort(key=lambda x: (-x + rng.random() * 200))
        else:
            rng.shuffle(order)
            
        greedy_rand, counts_rand = _greedy_build(order, TIME_LIMIT, start_time)
        if len(greedy_rand) > best_size:
            best = greedy_rand
            best_counts = counts_rand
            best_size = len(greedy_rand)
            
    if (time.time() - start_time) > TIME_LIMIT - 15:
        return best
    
    # Step 2: Local search to improve via swaps
    # Use in-place swap logic for speed
    remaining_time = TIME_LIMIT - (time.time() - start_time) - 5
    if remaining_time > 10:
        improved, improved_counts = _local_search(best, best_counts, remaining_time, start_time, seed=42)
        if len(improved) > best_size:
            best = improved
            best_counts = improved_counts
            best_size = len(best)
    
    if (time.time() - start_time) > TIME_LIMIT - 10:
        return best
    
    # Step 3: Fill remaining with conflict-aware descending pass
    remaining_time = TIME_LIMIT - (time.time() - start_time) - 3
    if remaining_time > 5:
        filled, filled_counts = _fill_remaining(best, best_counts, remaining_time, start_time)
        if len(filled) > best_size:
            best = filled
            best_counts = filled_counts
            best_size = len(best)
    
    # Step 4: One more local search pass if time remains
    if (time.time() - start_time) < TIME_LIMIT - 5:
        remaining_time = TIME_LIMIT - (time.time() - start_time) - 2
        if remaining_time > 5:
            improved2, improved2_counts = _local_search(best, best_counts, remaining_time, start_time, seed=99)
            if len(improved2) > best_size:
                best = improved2
                best_counts = improved2_counts
                best_size = len(best)
    
    return best
