"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import lcm

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s kill (evaluator limit)
    N = 24
    universe = list(range(1, N + 1))
    
    # Precompute LCM of all numbers 1..24 for exact integer arithmetic
    total_lcm = 1
    for i in range(1, N + 1):
        total_lcm = lcm(total_lcm, i)
    
    # weight[i] = total_lcm // i
    weights = {i: total_lcm // i for i in universe}
    # Map index 0..23 to number 1..24
    # weights_list[i-1] = weight[i]
    w_list = [0] * N
    for i in universe:
        w_list[i-1] = weights[i]
        
    def is_valid_mask(mask):
        """Check if the set represented by mask is valid."""
        # Extract elements
        elems = []
        for i in range(N):
            if mask & (1 << i):
                elems.append(i + 1)
                
        n = len(elems)
        if n <= 1:
            return True
        
        for i, a in enumerate(elems):
            target = weights[a]
            others = [x for j, x in enumerate(elems) if j != i]
            m = len(others)
            
            if m == 0:
                continue
            
            # Quick check: if target > sum of all others, impossible
            if target > sum(weights[x] for x in others):
                continue
                
            # Meet in the middle
            half = m // 2
            left = others[:half]
            right = others[half:]
            
            # Build left sums
            left_sums = set()
            left_sums.add(0)
            for mask_l in range(1, 1 << len(left)):
                s_sum = 0
                for k in range(len(left)):
                    if mask_l & (1 << k):
                        s_sum += weights[left[k]]
                left_sums.add(s_sum)
            
            # Check if target can be formed by left_sum + right_sum
            for mask_r in range(1 << len(right)):
                r_sum = 0
                for k in range(len(right)):
                    if mask_r & (1 << k):
                        r_sum += weights[right[k]]
                if (target - r_sum) in left_sums:
                    return False
            
        return True

    def count_violations(mask):
        """Count the number of violations in the set represented by mask."""
        elems = []
        for i in range(N):
            if mask & (1 << i):
                elems.append(i + 1)
                
        n = len(elems)
        if n <= 1:
            return 0
        
        violations = 0
        for i, a in enumerate(elems):
            target = weights[a]
            others = [x for j, x in enumerate(elems) if j != i]
            m = len(others)
            
            if m == 0:
                continue
            
            if target > sum(weights[x] for x in others):
                continue
                
            half = m // 2
            left = others[:half]
            right = others[half:]
            
            left_sums = set()
            left_sums.add(0)
            for mask_l in range(1, 1 << len(left)):
                s_sum = 0
                for k in range(len(left)):
                    if mask_l & (1 << k):
                        s_sum += weights[left[k]]
                left_sums.add(s_sum)
            
            found = False
            for mask_r in range(1 << len(right)):
                r_sum = 0
                for k in range(len(right)):
                    if mask_r & (1 << k):
                        r_sum += weights[right[k]]
                if (target - r_sum) in left_sums:
                    found = True
                    break
            if found:
                violations += 1
                
        return violations

    def popcount(x):
        return bin(x).count("1")

    # Initial best from Greedy
    best_mask = 0
    best_size = 0
    
    def greedy_construct(order):
        current_mask = 0
        current_elems = set()
        for elem in order:
            idx = elem - 1
            if current_mask & (1 << idx):
                continue
            new_mask = current_mask | (1 << idx)
            if is_valid_mask(new_mask):
                current_mask = new_mask
                current_elems.add(elem)
        return current_mask, len(current_elems)

    # 1. Run several greedy strategies to find a good initial solution
    rng = random.Random(42)
    
    orders = [
        sorted(universe, reverse=True),
        sorted(universe),
    ]
    
    # Add some random orders
    for _ in range(10):
        o = universe[:]
        rng.shuffle(o)
        orders.append(o)
        
    for order in orders:
        mask, size = greedy_construct(order)
        if size > best_size:
            best_size = size
            best_mask = mask
            
    # 2. Simulated Annealing / Local Search
    # Objective: Maximize size, minimize violations.
    # Score = size * 100 - violations * 10
    
    def get_score(mask):
        size = popcount(mask)
        viol = count_violations(mask)
        return size * 100 - viol * 10

    current_mask = best_mask
    current_score = get_score(current_mask)
    best_mask_sa = best_mask
    best_score_sa = current_score
    
    temp = 100.0
    cooling = 0.99
    
    start_time = time.time()
    
    while time.time() < deadline - 2:
        # Generate neighbor: flip a random bit
        idx = rng.randint(0, N - 1)
        neighbor_mask = current_mask ^ (1 << idx)
        
        neighbor_score = get_score(neighbor_mask)
        
        delta = neighbor_score - current_score
        
        if delta > 0 or (rng.random() < pow(2.71828, delta / temp)):
            current_mask = neighbor_mask
            current_score = neighbor_score
            
            if current_score > best_score_sa:
                best_score_sa = current_score
                best_mask_sa = current_mask
                
                # Update global best if valid and larger
                # Check if valid
                if count_violations(current_mask) == 0:
                    size = popcount(current_mask)
                    if size > best_size:
                        best_size = size
                        best_mask = current_mask
        else:
            # If not accepted, maybe try a different move?
            pass
            
        temp *= cooling
        
    # Final check: if best_mask_sa is valid and better, update
    if count_violations(best_mask_sa) == 0:
        size = popcount(best_mask_sa)
        if size > best_size:
            best_size = size
            best_mask = best_mask_sa
            
    # 3. Local improvement on best_mask
    if time.time() < deadline - 1:
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(N):
                if best_mask & (1 << i):
                    continue
                new_mask = best_mask | (1 << i)
                if is_valid_mask(new_mask):
                    best_mask = new_mask
                    best_size = popcount(best_mask)
                    improved = True
                    break
                    
    # Convert mask to list
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)
            
    return result
# EVOLVE-BLOCK-END
