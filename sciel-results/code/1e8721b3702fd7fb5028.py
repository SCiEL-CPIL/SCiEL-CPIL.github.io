import time
import random

def _subset_sums(arr):
    """Generate all subset sums of arr using efficient set operations."""
    sums = {0}
    for x in arr:
        new_sums = {s + x for s in sums}
        sums |= new_sums
    return sums

def _greedy_construct(n, max_val, start_time, deadline):
    """
    Greedy construction with backtracking: iteratively select the smallest
    positive integer that doesn't create duplicate subset sums.
    """
    arr = []
    sums = {0}
    
    def backtrack(idx, current_max):
        if idx == n:
            return current_max, True
        
        # Try candidates from 1 upward
        # We want to minimize the largest element, so try small values first
        # but we need to ensure we can complete the sequence
        
        for candidate in range(1, max_val + 1):
            if candidate in arr:
                continue
            
            # Check if adding this candidate creates duplicate subset sums
            new_sums = {s + candidate for s in sums}
            if len(new_sums & sums) > 0:
                continue
            
            # Valid candidate - recurse
            arr.append(candidate)
            sums |= new_sums
            
            result, success = backtrack(idx + 1, max(current_max, candidate))
            
            if success:
                return result, True
            
            # Backtrack
            arr.pop()
            sums -= new_sums
        
        return 0, False
    
    result, success = backtrack(0, 0)
    if success:
        return arr
    return None

def _local_search(arr, start_time, deadline):
    """
    Local search to improve the solution by trying small perturbations.
    """
    best_arr = arr[:]
    best_val = max(arr)
    current_arr = arr[:]
    
    while time.time() < deadline:
        # Generate a random neighbor
        neighbor = current_arr[:]
        move_type = random.choice(['change', 'nudge', 'swap'])
        
        if move_type == 'change':
            idx = random.randint(0, len(neighbor) - 1)
            neighbor[idx] = random.randint(1, best_val)
        elif move_type == 'nudge':
            idx = random.randint(0, len(neighbor) - 1)
            delta = random.choice([-1, 1, -2, 2, -3, 3])
            new_val = neighbor[idx] + delta
            if new_val > 0:
                neighbor[idx] = new_val
        else:  # swap
            i, j = random.sample(range(len(neighbor)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        # Check validity
        sums = _subset_sums(neighbor)
        is_valid = (len(sums) == (1 << len(neighbor)))
        
        if is_valid:
            new_val = max(neighbor)
            if new_val < best_val:
                best_val = new_val
                best_arr = neighbor[:]
                current_arr = neighbor[:]
        else:
            # With some probability, accept invalid solutions to escape local optima
            if random.random() < 0.1:
                current_arr = neighbor[:]
    
    return best_arr

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 12
    max_allowed = 1048576
    
    # Start with powers of 2 as a valid baseline
    best_arr = [1 << i for i in range(n)]
    best_val = max(best_arr)
    
    # Try greedy construction with different starting points
    # The greedy approach tries to find the lexicographically smallest valid sequence
    
    # First, try a simple greedy: pick smallest unused number that works
    greedy_success = False
    for trial in range(50):
        if time.time() >= deadline:
            break
        
        # Randomize the order of candidate selection slightly
        arr = []
        sums = {0}
        used = set()
        
        success = True
        for idx in range(n):
            # Try candidates from 1 upward
            found = False
            for candidate in range(1, best_val + 1):
                if candidate in used:
                    continue
                
                # Check if adding this candidate creates duplicate subset sums
                new_sums = {s + candidate for s in sums}
                if len(new_sums & sums) > 0:
                    continue
                
                # Valid candidate
                arr.append(candidate)
                sums |= new_sums
                used.add(candidate)
                found = True
                break
            
            if not found:
                success = False
                break
        
        if success:
            new_val = max(arr)
            if new_val < best_val:
                best_val = new_val
                best_arr = arr[:]
                greedy_success = True
    
    # If greedy found a good solution, do local search to improve further
    if time.time() < deadline:
        improved = _local_search(best_arr, start_time, deadline)
        if max(improved) < best_val:
            best_val = max(improved)
            best_arr = improved
    
    return best_arr
