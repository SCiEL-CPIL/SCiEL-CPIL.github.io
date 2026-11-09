"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from math import gcd
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    
    def lcm(a, b):
        return a // gcd(a, b) * b
    
    def is_valid(subset):
        """Check if all reciprocal subset sums of subset are distinct using integer scaling."""
        n = len(subset)
        if n == 0:
            return True
        # Compute LCM of all elements in subset
        L = 1
        for x in subset:
            L = lcm(L, x)
        if L > 10**18:  # Too large, fall back to Fraction check
            from fractions import Fraction
            sums = [Fraction(0)] * (1 << n)
            for mask in range(1, 1 << n):
                lsb = mask & (-mask)
                bit_idx = lsb.bit_length() - 1
                prev = mask ^ lsb
                sums[mask] = sums[prev] + Fraction(1, subset[bit_idx])
            return len(set(sums)) == (1 << n)
        
        # Scale: 1/x becomes L/x which is an integer
        weights = [L // x for x in subset]
        
        # Compute all subset sums using dynamic programming
        sums = [0] * (1 << n)
        for mask in range(1, 1 << n):
            lsb = mask & (-mask)
            bit_idx = lsb.bit_length() - 1
            prev = mask ^ lsb
            sums[mask] = sums[prev] + weights[bit_idx]
        
        return len(set(sums)) == (1 << n)
    
    def min_subset_sum_diff(subset):
        """Compute the minimum absolute difference between any two distinct subset sums."""
        n = len(subset)
        if n == 0:
            return float('inf')
        L = 1
        for x in subset:
            L = lcm(L, x)
        if L > 10**18:
            return float('inf')  # Can't compute efficiently
        
        weights = [L // x for x in subset]
        sums = [0] * (1 << n)
        for mask in range(1, 1 << n):
            lsb = mask & (-mask)
            bit_idx = lsb.bit_length() - 1
            prev = mask ^ lsb
            sums[mask] = sums[prev] + weights[bit_idx]
        
        # Find minimum difference
        sorted_sums = sorted(sums)
        min_diff = float('inf')
        for i in range(1, len(sorted_sums)):
            diff = sorted_sums[i] - sorted_sums[i-1]
            if diff < min_diff:
                min_diff = diff
        return min_diff
    
    best = []
    
    # Try multiple greedy strategies with different orderings
    orderings = [
        list(range(1, 23)),       # ascending
        list(range(22, 0, -1)),   # descending
        sorted(range(1, 23), key=lambda x: 1/x),  # by reciprocal ascending
        sorted(range(1, 23), key=lambda x: -1/x), # by reciprocal descending
        # Prime-heavy sets tend to work well
        [2, 3, 5, 7, 11, 13, 17, 19, 4, 6, 9, 10, 14, 15, 16, 18, 20, 21, 22],
        # Mix of small and large
        [1, 2, 4, 8, 16, 3, 6, 12, 5, 10, 20, 7, 14, 21, 9, 18, 11, 22, 13, 15, 17, 19],
    ]
    
    # Add more random orderings
    random.seed(42)
    for _ in range(200):
        perm = list(range(1, 23))
        random.shuffle(perm)
        orderings.append(perm)
    
    for order in orderings:
        if time.time() > deadline:
            break
        result = []
        for candidate in order:
            trial = result + [candidate]
            n = len(trial)
            if n <= 18:  # 2^18 = 262144, manageable with integers
                if is_valid(trial):
                    result = trial
        if len(result) > len(best):
            best = result
    
    # Simulated annealing to improve
    if time.time() < deadline - 10:
        random.seed(123)
        all_elements = set(range(1, 23))
        
        # Start from best greedy result
        current = list(best) if best else [1, 2, 3, 5, 7, 11, 13, 17, 19]
        current_set = set(current)
        
        # Initial temperature based on set size
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999
        
        # We want to maximize cardinality, so track best size found
        best_sa = current.copy()
        best_size = len(current)
        
        # Precompute LCM for quick checks
        def get_lcm(subset):
            L = 1
            for x in subset:
                L = lcm(L, x)
            return L
        
        iterations = 0
        max_iterations = 50000
        no_improve_count = 0
        
        while time.time() < deadline - 5 and iterations < max_iterations:
            iterations += 1
            
            # Generate a move: swap an element in/out
            if len(current) < 22 and random.random() < 0.3:
                # Try adding an element
                available = all_elements - current_set
                if not available:
                    continue
                elem_to_add = random.choice(list(available))
                trial = current + [elem_to_add]
                trial_set = current_set | {elem_to_add}
            elif len(current) > 1 and random.random() < 0.3:
                # Try removing an element
                elem_to_remove = random.choice(current)
                trial = [x for x in current if x != elem_to_remove]
                trial_set = current_set - {elem_to_remove}
            else:
                # Try swapping: remove one, add another
                if len(current) == 0 or not (all_elements - current_set):
                    continue
                elem_to_remove = random.choice(current)
                available = (all_elements - current_set)
                if not available:
                    continue
                elem_to_add = random.choice(list(available))
                trial = [x for x in current if x != elem_to_remove] + [elem_to_add]
                trial_set = (current_set - {elem_to_remove}) | {elem_to_add}
            
            # Check validity
            n = len(trial)
            if n > 18:
                continue
            
            if is_valid(trial):
                # Valid move - accept if it improves size or with some probability
                new_size = len(trial)
                if new_size > best_size:
                    best_sa = trial.copy()
                    best_size = new_size
                    no_improve_count = 0
                    T = max(T, 0.5)  # Reheat
                elif new_size >= len(current):
                    # Same or larger size, accept
                    current = trial
                    current_set = trial_set
                else:
                    # Smaller size, accept with probability
                    delta = len(current) - new_size
                    prob = min(1.0, T / delta)
                    if random.random() < prob:
                        current = trial
                        current_set = trial_set
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        if len(best_sa) > len(best):
            best = best_sa
    
    if best:
        return best
    
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
# EVOLVE-BLOCK-END
