import random
import time

def construct():
    n = 512
    
    # Use a bytearray for fast membership checks
    # S[i] = 1 if i is in the set, 0 otherwise
    # Index 0 is unused
    
    def creates_4ap_with_x(S, x):
        # S is a bytearray of length n+1
        # Check if adding x creates a 4-AP
        # Case 1: x, x+d, x+2d, x+3d
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            if S[x + d] and S[x + 2 * d] and S[x + 3 * d]:
                return True
        # Case 2: x-d, x, x+d, x+2d
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            if S[x - d] and S[x + d] and S[x + 2 * d]:
                return True
        # Case 3: x-2d, x-d, x, x+d
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            if S[x - 2 * d] and S[x - d] and S[x + d]:
                return True
        # Case 4: x-3d, x-2d, x-d, x
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if S[x - 3 * d] and S[x - 2 * d] and S[x - d]:
                return True
        return False

    best_set = bytearray(n + 1)
    best_size = 0
    
    start_time = time.time()
    time_limit = 55.0
    
    random.seed(42)
    
    # Phase 1: Randomized Greedy
    # Try many different random orders
    num_greedy_runs = 2000
    
    for _ in range(num_greedy_runs):
        if time.time() - start_time > time_limit / 2:
            break
            
        current_set = bytearray(n + 1)
        current_size = 0
        elements = list(range(1, n + 1))
        random.shuffle(elements)
        
        for x in elements:
            if not creates_4ap_with_x(current_set, x):
                current_set[x] = 1
                current_size += 1
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set[:]
            
    # Phase 2: Local Search on the best set
    # Try to add any missing element
    # If stuck, remove one element and try again
    
    current_set = best_set[:]
    current_size = best_size
    
    # Try to add any element not in set
    added = True
    while added:
        added = False
        for x in range(1, n + 1):
            if not current_set[x]:
                if not creates_4ap_with_x(current_set, x):
                    current_set[x] = 1
                    current_size += 1
                    added = True
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_set[:]

    # Iterative kick
    for _ in range(5000):
        if time.time() - start_time > time_limit:
            break
            
        # Try to add random elements
        made_progress = False
        for _ in range(10):
            candidates = [i for i in range(1, n + 1) if not current_set[i]]
            if not candidates:
                break
            x = random.choice(candidates)
            if not creates_4ap_with_x(current_set, x):
                current_set[x] = 1
                current_size += 1
                made_progress = True
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set[:]
                break
        
        if not made_progress:
            # Kick: remove 1-3 random elements
            k = random.randint(1, 3)
            if current_size > k:
                # Get list of elements in set
                in_set = [i for i in range(1, n + 1) if current_set[i]]
                to_remove = random.sample(in_set, k)
                for x in to_remove:
                    current_set[x] = 0
                    current_size -= 1
                
                # Refill greedily with random order
                elements = list(range(1, n + 1))
                random.shuffle(elements)
                for x in elements:
                    if not current_set[x]:
                        if not creates_4ap_with_x(current_set, x):
                            current_set[x] = 1
                            current_size += 1
                            
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set[:]

    # Convert bytearray to list of integers
    result = [i for i in range(1, n + 1) if best_set[i]]
    return result
