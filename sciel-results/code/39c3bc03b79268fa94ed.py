"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math

# Precompute prime factors for numbers up to 20000
MAX_SUM = 20000
_prime_factors = [set() for _ in range(MAX_SUM + 1)]
for i in range(2, MAX_SUM + 1):
    if not _prime_factors[i]:
        for j in range(i, MAX_SUM + 1, i):
            _prime_factors[j].add(i)

def _get_prime_factors(n):
    return _prime_factors[n]

def _count_distinct_primes(numbers):
    """Count distinct prime factors across all pairwise sums."""
    all_primes = set()
    n = len(numbers)
    for i in range(n):
        for j in range(i + 1, n):
            s = numbers[i] + numbers[j]
            all_primes |= _get_prime_factors(s)
    return len(all_primes)

def construct():
    N = 12
    MAX_VAL = 10000
    
    def _init_solution(strategy):
        if strategy == 0:
            return random.sample(range(1, MAX_VAL + 1), N)
        elif strategy == 1:
            # Multiples of 6
            candidates = [i for i in range(6, MAX_VAL + 1, 6)]
            if len(candidates) >= N:
                return random.sample(candidates, N)
            return random.sample(range(1, MAX_VAL + 1), N)
        else:
            # Small numbers
            return random.sample(range(1, 100), N)

    best_solution = None
    best_score = float('inf')
    
    # Run multiple restarts with Simulated Annealing
    for restart in range(5):
        # Initialize
        if restart == 0:
            nums = _init_solution(1)
        elif restart == 1:
            nums = _init_solution(0)
        else:
            nums = _init_solution(2)
            
        if len(set(nums)) != N:
            nums = random.sample(range(1, MAX_VAL + 1), N)
            
        current_score = _count_distinct_primes(nums)
        if current_score < best_score:
            best_score = current_score
            best_solution = nums[:]
            
        # Simulated Annealing
        T = 10.0
        cooling = 0.995
        min_T = 0.1
        max_steps = 5000
        
        for step in range(max_steps):
            if T < min_T:
                break
                
            # Pick a random index to modify
            idx = random.randint(0, N - 1)
            old_val = nums[idx]
            
            # Generate new value
            new_val = random.randint(1, MAX_VAL)
            
            # Ensure distinctness
            if new_val != old_val:
                exists_elsewhere = False
                for k in range(N):
                    if k != idx and nums[k] == new_val:
                        exists_elsewhere = True
                        break
                if exists_elsewhere:
                    continue
            
            nums[idx] = new_val
            new_score = _count_distinct_primes(nums)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_solution = nums[:]
            else:
                nums[idx] = old_val
                
            T *= cooling
            
    return best_solution

# EVOLVE-BLOCK-END
