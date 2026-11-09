import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Precompute LCM of 1..30
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    
    # Integer weights: weight[i] = L // (i+1) for coefficient of 1/(i+1)
    weights = [L // (i + 1) for i in range(30)]
    
    def get_sum(c):
        """Compute integer sum of c[i] * weights[i]"""
        s = 0
        for i in range(30):
            s += c[i] * weights[i]
        return s
    
    def is_valid(c):
        """Check that sum is nonzero"""
        return get_sum(c) != 0
    
    def local_search(c, max_iter=100000):
        """Hill-climbing: try single-coordinate flips and swaps"""
        current_sum = get_sum(c)
        best_c = list(c)
        best_abs = abs(current_sum)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            # Try flipping each coordinate
            for i in range(30):
                # Try changing c[i] to each value in {-1, 0, 1} if different
                for new_val in (-1, 0, 1):
                    if new_val == c[i]:
                        continue
                    delta = (new_val - c[i]) * weights[i]
                    new_sum = current_sum + delta
                    if new_sum == 0:
                        continue
                    new_abs = abs(new_sum)
                    if new_abs < best_abs:
                        c[i] = new_val
                        current_sum = new_sum
                        best_abs = new_abs
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Try pairwise swaps (flip two coordinates)
                found_pair = False
                for i in range(30):
                    if found_pair:
                        break
                    for j in range(i + 1, 30):
                        for ni in (-1, 0, 1):
                            if ni == c[i]:
                                continue
                            for nj in (-1, 0, 1):
                                if nj == c[j]:
                                    continue
                                delta = (ni - c[i]) * weights[i] + (nj - c[j]) * weights[j]
                                new_sum = current_sum + delta
                                if new_sum == 0:
                                    continue
                                new_abs = abs(new_sum)
                                if new_abs < best_abs:
                                    c[i] = ni
                                    c[j] = nj
                                    current_sum = new_sum
                                    best_abs = new_abs
                                    found_pair = True
                                    break
                            if found_pair:
                                break
                        if found_pair:
                            break
                    if found_pair:
                        break
                if not found_pair:
                    break
            
            best_c = list(c)
        
        return best_c, best_abs
    
    # Best solution found overall
    best_solution = None
    best_abs_sum = float('inf')
    
    # Multiple random restarts
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Generate random initial configuration
        # Strategy: start with sparse random assignments
        c = [0] * 30
        # Assign random values to a subset of positions
        k = random.randint(2, 15)
        positions = random.sample(range(30), k)
        for pos in positions:
            c[pos] = random.choice([-1, 1])
        
        # Ensure nonzero sum; if zero, flip one element
        if get_sum(c) == 0:
            for i in range(30):
                if c[i] == 0:
                    c[i] = 1
                    break
                else:
                    c[i] = -c[i]
                    break
            if get_sum(c) == 0:
                c[0] = 1
                if get_sum(c) == 0:
                    c[1] = -1
        
        # Run local search
        c, abs_sum = local_search(c)
        
        if abs_sum < best_abs_sum:
            best_abs_sum = abs_sum
            best_solution = list(c)
        
        # Also try the seed solution
        if restart == 0:
            seed_c = [0] * 28 + [1, -1]
            seed_c, seed_abs = local_search(seed_c)
            if seed_abs < best_abs_sum:
                best_abs_sum = seed_abs
                best_solution = list(seed_c)
    
    # Return best found, or fallback to seed
    if best_solution is not None:
        # Verify it's valid
        if get_sum(best_solution) != 0:
            return best_solution
    
    # Fallback: seed solution
    return [0] * 28 + [1, -1]
# EVOLVE-BLOCK-END
