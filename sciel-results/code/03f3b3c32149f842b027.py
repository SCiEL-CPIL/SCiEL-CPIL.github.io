import time
import random

def construct():
    n = 512
    start_time = time.time()
    time_limit = 58.0  # Leave 2s buffer
    
    # Helper: Check if adding x to bitmask M creates a 4-AP
    def creates_4ap(M, x):
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            if ((M >> (x + d)) & 1) and ((M >> (x + 2 * d)) & 1) and ((M >> (x + 3 * d)) & 1):
                return True
        
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            if ((M >> (x - d)) & 1) and ((M >> (x + d)) & 1) and ((M >> (x + 2 * d)) & 1):
                return True
        
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            if ((M >> (x - 2 * d)) & 1) and ((M >> (x - d)) & 1) and ((M >> (x + d)) & 1):
                return True
        
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if ((M >> (x - 3 * d)) & 1) and ((M >> (x - 2 * d)) & 1) and ((M >> (x - d)) & 1):
                return True
        
        return False

    # Helper: Greedily fill the bitmask
    def greedy_fill(M, size, order=None):
        if order is None:
            order = list(range(1, n + 1))
        for x in order:
            if not (M >> x) & 1:
                if not creates_4ap(M, x):
                    M |= (1 << x)
                    size += 1
        return M, size

    best_M = 0
    best_size = 0
    
    random.seed(42)
    
    # Standard order
    current_M, current_size = greedy_fill(0, 0, list(range(1, n + 1)))
    if current_size > best_size:
        best_M = current_M
        best_size = current_size
        
    # Reverse order
    current_M, current_size = greedy_fill(0, 0, list(range(n, 0, -1)))
    if current_size > best_size:
        best_M = current_M
        best_size = current_size
        
    # Random order
    for _ in range(10):
        if time.time() - start_time > time_limit * 0.1: # 10% time for init
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        current_M, current_size = greedy_fill(0, 0, order)
        if current_size > best_size:
            best_M = current_M
            best_size = current_size

    # Local Search
    current_M = best_M
    current_size = best_size
    
    while time.time() - start_time < time_limit:
        added = False
        missing = [i for i in range(1, n + 1) if not (current_M >> i) & 1]
        
        if not missing:
            break
            
        random.shuffle(missing)
        
        for x in missing[:50]: # Check up to 50 random candidates
            if not creates_4ap(current_M, x):
                current_M |= (1 << x)
                current_size += 1
                added = True
                if current_size > best_size:
                    best_M = current_M
                    best_size = current_size
                break
        
        if not added:
            # Stuck. Perform a kick.
            # Remove 1-2 random elements
            k = random.randint(1, 2)
            present = [i for i in range(1, n + 1) if (current_M >> i) & 1]
            if len(present) > k:
                to_remove = random.sample(present, k)
                for x in to_remove:
                    current_M &= ~(1 << x)
                    current_size -= 1
                
                # Now try to add as many as possible
                # Greedy fill from current state
                missing = [i for i in range(1, n + 1) if not (current_M >> i) & 1]
                random.shuffle(missing)
                for x in missing:
                    if not creates_4ap(current_M, x):
                        current_M |= (1 << x)
                        current_size += 1
                
                if current_size > best_size:
                    best_M = current_M
                    best_size = current_size

    result = [i for i in range(1, n + 1) if (best_M >> i) & 1]
    return result
