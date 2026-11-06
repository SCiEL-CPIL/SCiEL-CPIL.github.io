import time
import random

def construct():
    n = 512
    deadline = time.time() + 55  # Safety margin for 60s limit
    
    # Precompute conflicts
    # conflicts[x] is a list of bitmasks. If (S & mask) == mask, x cannot be added.
    # x is 1-indexed in problem, 0-indexed in bit operations.
    # Bit i corresponds to element i+1.
    
    conflicts = [[] for _ in range(n)]
    
    for x in range(1, n + 1):
        i = x - 1
        # Case 1: x is 1st term: x, x+d, x+2d, x+3d
        # Need x+d, x+2d, x+3d in S
        max_d1 = (n - x) // 3
        for d in range(1, max_d1 + 1):
            y1 = x + d
            y2 = x + 2 * d
            y3 = x + 3 * d
            mask = (1 << (y1 - 1)) | (1 << (y2 - 1)) | (1 << (y3 - 1))
            conflicts[i].append(mask)
            
        # Case 2: x is 2nd term: x-d, x, x+d, x+2d
        # Need x-d, x+d, x+2d in S
        max_d2 = min(x - 1, (n - x) // 2)
        for d in range(1, max_d2 + 1):
            y1 = x - d
            y2 = x + d
            y3 = x + 2 * d
            mask = (1 << (y1 - 1)) | (1 << (y2 - 1)) | (1 << (y3 - 1))
            conflicts[i].append(mask)
            
        # Case 3: x is 3rd term: x-2d, x-d, x, x+d
        # Need x-2d, x-d, x+d in S
        max_d3 = min((x - 1) // 2, n - x)
        for d in range(1, max_d3 + 1):
            y1 = x - 2 * d
            y2 = x - d
            y3 = x + d
            mask = (1 << (y1 - 1)) | (1 << (y2 - 1)) | (1 << (y3 - 1))
            conflicts[i].append(mask)
            
        # Case 4: x is 4th term: x-3d, x-2d, x-d, x
        # Need x-3d, x-2d, x-d in S
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            y1 = x - 3 * d
            y2 = x - 2 * d
            y3 = x - d
            mask = (1 << (y1 - 1)) | (1 << (y2 - 1)) | (1 << (y3 - 1))
            conflicts[i].append(mask)
            
    def can_add(x_idx, S):
        """Check if element at index x_idx (0-based) can be added to bitset S."""
        for mask in conflicts[x_idx]:
            if (S & mask) == mask:
                return False
        return True

    def is_valid(S):
        """Check if S contains no 4-AP."""
        for i in range(n):
            if S & (1 << i):
                if not can_add(i, S & ~(1 << i)):
                    return False
        return True

    def greedy_fill(S):
        """Fill S greedily with elements that can be added."""
        for i in range(n):
            if not (S & (1 << i)):
                if can_add(i, S):
                    S |= (1 << i)
        return S

    def simulated_annealing(S):
        """Simulated annealing to improve S."""
        best_S = S
        best_size = S.bit_count()
        T = 1.0  # Initial temperature
        cooling = 0.9995
        
        while time.time() < deadline and T > 0.01:
            # Generate a move: remove k random elements and refill
            set_bits = []
            temp_S = S
            for i in range(n):
                if temp_S & (1 << i):
                    set_bits.append(i)
                temp_S >>= 1
            
            if not set_bits:
                break
            
            # Choose k elements to remove (k=1,2,3 with some probability)
            k = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
            if len(set_bits) < k:
                k = len(set_bits)
            
            removed = random.sample(set_bits, k)
            S_new = S
            for i in removed:
                S_new &= ~(1 << i)
            
            # Refill greedily
            S_new = greedy_fill(S_new)
            
            new_size = S_new.bit_count()
            old_size = S.bit_count()
            
            # Accept if better, or with probability exp((new-old)/T)
            if new_size > old_size:
                S = S_new
                if new_size > best_size:
                    best_S = S
                    best_size = new_size
            else:
                # Accept worse moves with probability exp((new-old)/T)
                delta = new_size - old_size
                if delta < 0:
                    prob = 2.718281828 ** (delta / T)
                    if random.random() < prob:
                        S = S_new
            
            T *= cooling
        
        return best_S

    best_S = 0
    best_size = 0
    
    # Initial greedy construction
    order = list(range(n))
    random.shuffle(order)
    S = 0
    for i in order:
        if can_add(i, S):
            S |= (1 << i)
    S = greedy_fill(S)
    
    if S.bit_count() > best_size:
        best_S = S
        best_size = S.bit_count()
    
    # Simulated annealing to improve
    S = simulated_annealing(S)
    
    if S.bit_count() > best_size:
        best_S = S
        best_size = S.bit_count()
    
    # Try a few more restarts if time permits
    restarts = 0
    while time.time() < deadline - 5:
        order = list(range(n))
        random.shuffle(order)
        S = 0
        for i in order:
            if can_add(i, S):
                S |= (1 << i)
        S = greedy_fill(S)
        
        if S.bit_count() > best_size:
            best_S = S
            best_size = S.bit_count()
        
        # Quick SA from this starting point
        S = simulated_annealing(S)
        
        if S.bit_count() > best_size:
            best_S = S
            best_size = S.bit_count()
        
        restarts += 1
        if restarts >= 3:  # Limit restarts
            break
    
    # Convert bitset to sorted list
    result = []
    for i in range(n):
        if best_S & (1 << i):
            result.append(i + 1)
            
    return sorted(result)
