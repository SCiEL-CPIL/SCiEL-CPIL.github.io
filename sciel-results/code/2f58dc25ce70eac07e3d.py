import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    N = 1000
    max_sum = 2 * N
    
    # Data structures
    A = []
    in_A = set()
    sum_counts = [0] * (2 * N + 1)
    
    best_set = []
    best_size = 0
    
    def try_add(A, in_A, sum_counts, x):
        if x in in_A:
            return False
        if sum_counts[2*x] >= 2:
            return False
        for y in A:
            if sum_counts[x+y] >= 2:
                return False
        # Valid
        in_A.add(x)
        A.append(x)
        sum_counts[2*x] += 1
        for y in A[:-1]:
            sum_counts[x+y] += 1
        return True

    def remove_element(A, in_A, sum_counts, x):
        if x not in in_A:
            return False
        in_A.remove(x)
        A.remove(x)
        sum_counts[2*x] -= 1
        for y in A:
            sum_counts[x+y] -= 1
        return True

    def try_swap(A, in_A, sum_counts, y, x):
        if x in in_A or y not in in_A:
            return False
        
        # Check 2x
        s = 2 * x
        contrib_y = 1 if (s - y) in in_A and (s - y) != y else 0
        if sum_counts[s] - contrib_y + 1 > 2:
            return False
            
        # Check x+z for z in A, z != y
        for z in A:
            if z == y:
                continue
            s = x + z
            contrib_y = 1 if (s - y) in in_A and (s - y) != y else 0
            if sum_counts[s] - contrib_y + 1 > 2:
                return False
                
        # Perform swap
        remove_element(A, in_A, sum_counts, y)
        in_A.add(x)
        A.append(x)
        sum_counts[2*x] += 1
        for z in A[:-1]:
            sum_counts[x+z] += 1
        return True

    # Initialize with greedy
    for x in range(1, N + 1):
        if try_add(A, in_A, sum_counts, x):
            continue
    
    best_set = A[:]
    best_size = len(A)
    
    # Local search with simulated annealing
    # Temperature starts high and cools down
    T = 10.0
    cooling_rate = 0.995
    
    while time.time() < deadline:
        # 1. Try to add any element (always accept improvements)
        added = False
        # Build candidate list more efficiently
        candidates = [x for x in range(1, N + 1) if x not in in_A]
        random.shuffle(candidates)
        
        for x in candidates:
            if try_add(A, in_A, sum_counts, x):
                added = True
                if len(A) > best_size:
                    best_size = len(A)
                    best_set = A[:]
                break
        
        if added:
            continue
        
        # 2. Try swaps with simulated annealing
        swapped = False
        # Try multiple swap attempts per iteration
        for _ in range(50):
            if not A or not candidates:
                break
            y = random.choice(A)
            x = random.choice(candidates)
            if try_swap(A, in_A, sum_counts, y, x):
                swapped = True
                if len(A) > best_size:
                    best_size = len(A)
                    best_set = A[:]
                break
        
        if swapped:
            T *= cooling_rate
            continue
            
        # 3. If no swap, try a random move (accept with probability based on temperature)
        if A and candidates:
            y = random.choice(A)
            x = random.choice(candidates)
            # Calculate "cost" of this swap (how many constraints it would violate)
            # For simplicity, just accept with probability exp(-1/T)
            if random.random() < min(1.0, T / 10.0):
                if try_swap(A, in_A, sum_counts, y, x):
                    T *= cooling_rate
                    continue
        
        # 4. If stuck, remove a random element to break deadlock
        if A and random.random() < 0.3:
            y = random.choice(A)
            remove_element(A, in_A, sum_counts, y)
            T = max(T * 1.1, 1.0)  # Reheat slightly
        
        T *= cooling_rate
        T = max(T, 0.1)
    
    return best_set
