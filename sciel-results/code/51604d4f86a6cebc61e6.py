import random
import time

def construct():
    start = time.time()
    deadline = start + 85.0  # 5s safety margin before 90s kill

    def is_valid(sums):
        """Check if any two distinct sums divide each other."""
        if len(sums) < 2:
            return True
        sum_list = sorted(sums)
        max_sum = sum_list[-1]
        sums_set = sums
        for u in sum_list:
            if u * 2 > max_sum:
                break
            for v in range(u * 2, max_sum + 1, u):
                if v in sums_set:
                    return False
        return True

    def try_add_element(current_sums, c, current_set):
        """Try adding element c. Returns (new_sums, new_set) if valid, else (None, None)."""
        if c in current_set:
            return None, None
        new_sums = set(current_sums)
        for s in current_sums:
            new_sums.add(s + c)
        new_sums.add(c)
        
        # Check for duplicate sums (which would violate the distinct subset sum condition)
        if len(new_sums) != 2 * len(current_sums) + 1:
            return None, None
            
        if not is_valid(new_sums):
            return None, None
            
        new_set = current_set | {c}
        return new_sums, new_set

    best_set = set()
    best_size = 0

    # Strategy 1: Deterministic greedy with descending order (large numbers first)
    # Large numbers have fewer multiples within the range, reducing conflicts.
    # We try multiple ordering heuristics.
    
    orderings = [
        list(range(256, 0, -1)),  # Descending
        list(range(1, 257)),      # Ascending
        list(range(128, 257)),    # Large only ascending
        list(range(256, 127, -1)),# Large only descending
    ]

    for ordering in orderings:
        if time.time() > deadline - 5.0:
            break
        
        current_sums = set()
        current_set = set()
        
        for c in ordering:
            if len(current_set) >= 30:
                break
            new_sums, new_set = try_add_element(current_sums, c, current_set)
            if new_sums is not None:
                current_sums = new_sums
                current_set = new_set
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            if best_size >= 30:
                break

    # Strategy 2: Randomized restarts with greedy insertion
    # This explores different combinations that deterministic greedy might miss.
    
    for attempt in range(2000):
        if time.time() > deadline - 5.0:
            break
        
        # Bias towards larger numbers initially, but allow smaller ones later
        # We can shuffle the candidate list but keep a bias.
        # Simple approach: shuffle the full range, or pick a random subset of candidates to consider in a specific order.
        
        # Let's try shuffling the full range, but maybe weight larger numbers?
        # Actually, random shuffle of 1..256 is a good general search.
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        # To make it more effective, we can try to prioritize numbers that are "far" from multiples of existing sums.
        # But simple shuffle + greedy is a strong baseline.
        
        current_sums = set()
        current_set = set()
        
        for c in candidates:
            if len(current_set) >= 30:
                break
            new_sums, new_set = try_add_element(current_sums, c, current_set)
            if new_sums is not None:
                current_sums = new_sums
                current_set = new_set
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            if best_size >= 30:
                break

    # Strategy 3: Local Search / Refinement
    # Try to improve the best set by swapping elements or adding/removing.
    # If we have a set of size k, can we find a set of size k+1?
    
    if best_size > 0 and time.time() < deadline - 10.0:
        # We can try to extend the best set by removing one element and trying to add two?
        # Or just try random swaps.
        
        # Let's try a simple local search: 
        # Pick a random element from best_set, remove it, then try to greedily add back elements (including the removed one) in a random order.
        
        for _ in range(500):
            if time.time() > deadline - 5.0:
                break
            
            base_list = list(best_set)
            random.shuffle(base_list)
            
            current_sums = set()
            current_set = set()
            
            # Greedily add from the shuffled base list
            for c in base_list:
                new_sums, new_set = try_add_element(current_sums, c, current_set)
                if new_sums is not None:
                    current_sums = new_sums
                    current_set = new_set
            
            # Now try to add more elements from the rest of the range
            remaining = [c for c in range(1, 257) if c not in current_set]
            random.shuffle(remaining)
            
            for c in remaining:
                if len(current_set) >= 30:
                    break
                new_sums, new_set = try_add_element(current_sums, c, current_set)
                if new_sums is not None:
                    current_sums = new_sums
                    current_set = new_set
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
                if best_size >= 30:
                    break

    # Convert set to list of ints
    return [int(x) for x in best_set]
