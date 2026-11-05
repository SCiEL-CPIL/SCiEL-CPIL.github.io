import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Precompute weights and prime factors
    N = 512
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)
        
    # Prime factors for each number
    prime_factors = [set() for _ in range(N)]
    for p in range(2, N):
        is_prime = True
        for d in range(2, int(p**0.5) + 1):
            if p % d == 0:
                is_prime = False
                break
        if is_prime:
            for multiple in range(p, N, p):
                prime_factors[multiple].add(p)
                
    # 2. Greedy Selection
    # Sort candidates by weight descending (i.e., a descending)
    candidates = list(range(511, 0, -1))
    
    selected = []
    selected_set = set()
    
    for a in candidates:
        # Check if a is coprime to all in selected
        compatible = True
        for b in selected:
            if gcd(a, b) > 1:
                compatible = False
                break
        if compatible:
            selected.append(a)
            selected_set.add(a)
            
    best_score = sum(weights[a] for a in selected)
    best_solution = list(selected)
    
    # 3. Local Search: Try to improve by swapping or adding
    # We will iterate a few times or until no improvement
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding any unselected number
        for a in range(511, 0, -1):
            if a in selected_set:
                continue
            if all(gcd(a, b) == 1 for b in best_solution):
                best_solution.append(a)
                selected_set.add(a)
                best_score += weights[a]
                improved = True
        
        # Try swapping: remove one, add one
        # To be efficient, only try swapping if the new number has higher weight
        # and is compatible with the rest.
        # This is O(N^2) per iteration, which is 511^2 ~ 260k ops, very fast.
        for i, a_out in enumerate(best_solution):
            remaining = [x for j, x in enumerate(best_solution) if j != i]
            # Find best a_in to replace a_out
            best_gain = 0
            best_a_in = None
            for a_in in range(511, 0, -1):
                if a_in in selected_set:
                    continue
                if any(gcd(a_in, b) > 1 for b in remaining):
                    continue
                gain = weights[a_in] - weights[a_out]
                if gain > best_gain:
                    best_gain = gain
                    best_a_in = a_in
            
            if best_a_in is not None and best_gain > 1e-12:
                # Perform swap
                best_solution[i] = best_a_in
                selected_set.remove(a_out)
                selected_set.add(best_a_in)
                best_score += best_gain
                improved = True
                # Restart the while loop to check for further additions
                break 

    return best_solution
