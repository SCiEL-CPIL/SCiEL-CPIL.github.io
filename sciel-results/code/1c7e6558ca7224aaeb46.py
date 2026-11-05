import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    fixed = {1, 2, 5, 11}
    best_set = list(fixed)
    best_size = len(fixed)
    
    def get_sums(s):
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.add(s[i] + s[j])
        return sums

    def is_valid_addition(x, current_set, current_sums):
        # Check if x can be added to current_set
        # Sums involving x: x + y for y in current_set, and x + x
        for y in current_set:
            if x + y in current_sums:
                return False
        if x + x in current_sums:
            return False
        return True

    def add_to_set(x, current_set, current_sums):
        for y in current_set:
            current_sums.add(x + y)
        current_sums.add(x + x)
        current_set.append(x)
        current_set.sort()

    def remove_from_set(x, current_set, current_sums):
        # Remove sums involving x
        for y in current_set:
            if y != x:
                current_sums.discard(x + y)
        current_sums.discard(x + x)
        current_set.remove(x)

    def greedy_fill(current_set, current_sums, candidates_order=None):
        """Add all valid elements from candidates in the given order."""
        if candidates_order is None:
            candidates_order = [x for x in range(1, 285) if x not in current_set]
        for x in candidates_order:
            if time.time() > deadline:
                break
            if x not in current_set and is_valid_addition(x, current_set, current_sums):
                add_to_set(x, current_set, current_sums)
        return current_set, current_sums

    # Initial greedy fill with multiple random orders, keep best
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        current_set = list(fixed)
        current_sums = get_sums(current_set)
        candidates = [x for x in range(1, 285) if x not in fixed]
        random.shuffle(candidates)
        greedy_fill(current_set, current_sums, candidates)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            if best_size >= 20:
                return best_set

    # Local search with destroy and repair
    # Use the best set found so far as starting point
    current_set = list(best_set)
    current_sums = get_sums(current_set)
    
    attempts = 0
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 5:
        attempts += 1
        
        # Destroy: remove 2-5 random non-fixed elements for more aggressive search
        removable = [x for x in current_set if x not in fixed]
        if not removable:
            break
            
        # Scale removal count with set size
        if len(removable) >= 10:
            num_remove = random.randint(2, 5)
        elif len(removable) >= 5:
            num_remove = random.randint(1, 3)
        else:
            num_remove = random.randint(1, 2)
        num_remove = min(num_remove, len(removable))
        to_remove = random.sample(removable, num_remove)
        
        for x in to_remove:
            remove_from_set(x, current_set, current_sums)
            
        # Repair: try to add elements back, prioritizing removed ones first
        # Then try all other candidates
        removed_set = set(to_remove)
        all_candidates = [x for x in range(1, 285) if x not in current_set]
        
        # Separate removed and other candidates
        removed_candidates = [x for x in all_candidates if x in removed_set]
        other_candidates = [x for x in all_candidates if x not in removed_set]
        
        random.shuffle(removed_candidates)
        random.shuffle(other_candidates)
        
        # Try removed candidates first (to recover size)
        for x in removed_candidates:
            if time.time() > deadline - 5:
                break
            if is_valid_addition(x, current_set, current_sums):
                add_to_set(x, current_set, current_sums)
        
        # Then try other candidates (to potentially grow)
        for x in other_candidates:
            if time.time() > deadline - 5:
                break
            if is_valid_addition(x, current_set, current_sums):
                add_to_set(x, current_set, current_sums)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            no_improve_count = 0
            
            # If we hit the theoretical max (20), we can stop
            if best_size >= 20:
                break
        else:
            no_improve_count += 1
            # Reset to best if stuck for too long
            if no_improve_count >= max_no_improve:
                current_set = list(best_set)
                current_sums = get_sums(current_set)
                no_improve_count = 0
                max_no_improve += 10  # Increase threshold
    
    # Final polish: try to add any remaining valid elements to best_set
    if time.time() < deadline - 1:
        final_set = list(best_set)
        final_sums = get_sums(final_set)
        remaining = [x for x in range(1, 285) if x not in final_set]
        random.shuffle(remaining)
        for x in remaining:
            if time.time() > deadline - 1:
                break
            if is_valid_addition(x, final_set, final_sums):
                add_to_set(x, final_set, final_sums)
        
        if len(final_set) > best_size:
            best_size = len(final_set)
            best_set = list(final_set)
    
    return best_set
