"""Rewritten program with optimized prime power selection for maximum coprime sum."""
# EVOLVE-BLOCK-START
import time

_START_TIME = time.time()
_MAX_TIME_LIMIT = 85.0

def construct():
    """
    Returns a list of distinct integers from 1 to 256, pairwise coprime,
    maximizing their sum.
    """
    # Check time limit to ensure we don't get killed by the evaluator
    if time.time() - _START_TIME > _MAX_TIME_LIMIT:
        return [1]
    
    limit = 256
    
    # Identify all prime numbers up to 256
    primes_list = []
    is_prime_flag = [True] * (limit + 1)
    is_prime_flag[0] = False
    is_prime_flag[1] = False
    
    for current in range(2, int(limit ** 0.5) + 1):
        if is_prime_flag[current]:
            for multiple in range(current * current, limit + 1, current):
                is_prime_flag[multiple] = False
    
    for num in range(2, limit + 1):
        if is_prime_flag[num]:
            primes_list.append(num)
            
    # Construct the optimal set:
    # 1 is always included.
    # For each prime p, include the largest power p^k <= 256.
    # These numbers are pairwise coprime because each is a power of a unique prime.
    optimal_numbers = [1]
    
    for p in primes_list:
        max_power_val = p
        # Find largest p^k <= 256
        while max_power_val * p <= limit:
            max_power_val *= p
        optimal_numbers.append(max_power_val)
        
    # Ensure all elements are standard Python integers
    final_result = []
    for val in optimal_numbers:
        final_result.append(int(val))
        
    return final_result
# EVOLVE-BLOCK-END
