import time
import random
import math

START_TIME = time.time()
TIME_LIMIT = 85.0

def compute_sums(nums):
    """Compute all non-empty subset sums."""
    sums = set()
    n = len(nums)
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += nums[i]
        sums.add(s)
    return sums

def is_valid(sums):
    """Check if the set of sums is valid (no duplicates, no divisibility)."""
    # Check duplicates
    if len(sums) != len(set(sums)):
        return False
    
    max_sum = max(sums)
    # Boolean array for fast lookup
    # Max possible sum is 30 * 256 = 7680
    # But for n<=12, max sum is smaller.
    # We can just use a set for lookup if max_sum is large, 
    # but boolean array is faster for small max_sum.
    # Given n<=12, max_sum <= 12*256 = 3072.
    # So boolean array of size 3100 is efficient.
    
    if max_sum > 10000:
        # Fallback to set if somehow large (shouldn't happen for n<=12)
        sum_set = set(sums)
        for s in sums:
            mult = 2
            while s * mult <= max_sum:
                if (s * mult) in sum_set:
                    return False
                mult += 1
        return True
    else:
        is_sum = [False] * (max_sum + 1)
        for s in sums:
            is_sum[s] = True
        
        for s in sums:
            mult = 2
            while s * mult <= max_sum:
                if is_sum[s * mult]:
                    return False
                mult += 1
        return True

def check_set_validity(nums):
    """Check if the set of numbers is valid."""
    if not nums:
        return True
    sums = compute_sums(nums)
    return is_valid(sums)

def construct():
    """
    Main entry point. Uses Simulated Annealing to find a valid set of maximum size.
    Max possible size is 12 due to pigeonhole principle on subset sums.
    """
    best_set = []
    best_size = 0
    
    pool = list(range(1, 257))
    
    # Run multiple SA restarts
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # Initialize with a random small set
        # Start with size 5-8
        init_size = 5 + (r % 4)
        current_set = random.sample(pool, init_size)
        
        # Ensure it's valid to start? 
        # If not, SA will try to fix it.
        # But it's better to start valid.
        # Let's try to build a valid initial set greedily.
        current_set = []
        remaining = random.sample(pool, 20)
        for elem in remaining:
            trial = current_set + [elem]
            if check_set_validity(trial):
                current_set = trial
                if len(current_set) >= 8:
                    break
        
        if not current_set:
            # Fallback
            current_set = [100, 101, 102]
            
        current_sums = compute_sums(current_set)
        current_valid = is_valid(current_sums)
        current_size = len(current_set)
        
        if current_valid and current_size > best_size:
            best_set = current_set[:]
            best_size = current_size
            
        # Simulated Annealing
        T = 100.0
        cooling = 0.99
        min_T = 0.1
        max_steps = 2000
        
        for step in range(max_steps):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            if T < min_T:
                break
                
            # Generate neighbor
            neighbor = current_set[:]
            
            # Move types:
            # 1. Add element (if size < 12)
            # 2. Remove element (if size > 1)
            # 3. Swap element
            
            r_move = random.random()
            
            if current_size < 12 and r_move < 0.4:
                # Add
                candidates = [x for x in pool if x not in current_set]
                if candidates:
                    new_elem = random.choice(candidates)
                    neighbor.append(new_elem)
            elif current_size > 1 and r_move < 0.7:
                # Remove
                idx = random.randint(0, len(neighbor) - 1)
                neighbor.pop(idx)
            else:
                # Swap
                if current_size > 0:
                    idx = random.randint(0, len(neighbor) - 1)
                    old_elem = neighbor[idx]
                    candidates = [x for x in pool if x not in current_set]
                    if candidates:
                        new_elem = random.choice(candidates)
                        neighbor[idx] = new_elem
            
            if neighbor == current_set:
                continue
                
            # Evaluate neighbor
            neighbor_sums = compute_sums(neighbor)
            neighbor_valid = is_valid(neighbor_sums)
            neighbor_size = len(neighbor)
            
            # Calculate delta energy
            # We want to maximize size, and minimize violations (invalidity)
            # Energy = -size * 10 + (0 if valid else 100)
            # Or simply:
            # If current is valid and neighbor is valid: prefer larger size
            # If current is invalid: prefer valid neighbor
            
            if current_valid and neighbor_valid:
                delta = current_size - neighbor_size # We want to minimize delta (increase size)
                # Accept if neighbor is larger (delta < 0) or with probability
                if delta <= 0:
                    accept = True
                else:
                    # delta > 0, neighbor is smaller
                    accept = random.random() < math.exp(-delta / T)
            elif current_valid and not neighbor_valid:
                # Moving from valid to invalid
                # Accept with low probability to escape local optima?
                # Usually we want to stay valid.
                # But if we are stuck at size 10, we might need to go invalid to get to 12?
                # No, going invalid is bad.
                # Let's accept with very low probability
                accept = random.random() < 0.01
            elif not current_valid and neighbor_valid:
                # Moving from invalid to valid
                accept = True
            else:
                # Both invalid
                # Prefer larger size even if invalid?
                delta = current_size - neighbor_size
                if delta <= 0:
                    accept = True
                else:
                    accept = random.random() < math.exp(-delta / T)
            
            if accept:
                current_set = neighbor
                current_sums = neighbor_sums
                current_valid = neighbor_valid
                current_size = neighbor_size
                
                if current_valid and current_size > best_size:
                    best_set = current_set[:]
                    best_size = current_size
                    if best_size == 12:
                        break # Max possible
            
            T *= cooling
            
        if best_size == 12:
            break
            
    # Final verification of best_set
    if best_set and not check_set_validity(best_set):
        # If best is invalid (shouldn't happen), find a valid fallback
        # Try small sets
        for s in range(1, 13):
            if check_set_validity(best_set[:s]):
                best_set = best_set[:s]
                break
        else:
            best_set = [100]
            
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-START
# (The above code replaces the evolve block)
# EVOLVE-BLOCK-END
