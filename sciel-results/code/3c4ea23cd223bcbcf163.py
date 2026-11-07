import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    def is_valid(S):
        if not S:
            return True
        n = len(S)
        
        # Generate all non-empty subset sums
        sums = {0}
        for x in S:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            sums = new_sums
        
        # Remove 0
        sums.discard(0)
        
        # Check for duplicates
        # If len(sums) < 2**n - 1, there were collisions
        if len(sums) < (1 << n) - 1:
            return False
        
        # Check divisibility
        sums_list = sorted(sums)
        max_sum = sums_list[-1]
        sums_set = set(sums_list)
        
        for u in sums_list:
            if u > max_sum // 2:
                continue
            m = 2 * u
            while m <= max_sum:
                if m in sums_set:
                    return False
                m += u
        return True

    def get_best_set():
        best_size = 0
        best_set = []
        
        # Try multiple random restarts
        for restart in range(300):
            if time.time() > deadline:
                break
            
            # Start with a small random set of larger numbers
            # Avoiding 1 and small numbers helps with divisibility
            start_size = 3
            curr = random.sample(range(50, 257), start_size)
            curr.sort()
            
            if not is_valid(curr):
                # Try smaller numbers if large ones fail? Or just restart
                curr = random.sample(range(20, 257), start_size)
                curr.sort()
                if not is_valid(curr):
                    curr = []
            
            improved = True
            while improved:
                improved = False
                if time.time() > deadline:
                    break
                
                # Try to add a new element
                added = False
                candidates = [i for i in range(2, 257) if i not in curr]
                # Prefer larger numbers to keep sums large and distinct
                candidates.sort(reverse=True)
                
                for c in candidates:
                    new_curr = curr + [c]
                    if is_valid(new_curr):
                        curr = new_curr
                        improved = True
                        added = True
                        break
                
                if added:
                    if len(curr) > best_size:
                        best_size = len(curr)
                        best_set = curr[:]
                    continue # Restart the while loop to try adding more
                
                # If no element could be added, try swapping
                if curr:
                    idx = random.randint(0, len(curr) - 1)
                    old_elem = curr[idx]
                    curr_without = curr[:idx] + curr[idx+1:]
                    
                    candidates = [i for i in range(2, 257) if i not in curr_without]
                    candidates.sort(reverse=True)
                    
                    for c in candidates:
                        new_curr = curr_without + [c]
                        new_curr.sort()
                        if is_valid(new_curr):
                            curr = new_curr
                            improved = True
                            break
                            
            if len(curr) > best_size:
                best_size = len(curr)
                best_set = curr[:]
                
        return best_set

    result = get_best_set()
    
    # Ensure types are correct
    return [int(x) for x in result]

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
