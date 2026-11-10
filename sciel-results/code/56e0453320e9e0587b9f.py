import time
from math import factorial

TARGET = factorial(12)
DEADLINE = 80.0 # seconds

# Precompute valid divisors of TARGET that are > 12
def get_divisors(n):
    divs = []
    # Prime factorization of 12!
    # 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    exps = [10, 5, 2, 1, 1]
    
    def gen(idx, cur):
        if idx == len(primes):
            if cur > 12:
                divs.append(cur)
            return
        for e in range(exps[idx] + 1):
            gen(idx + 1, cur * (primes[idx] ** e))
            
    gen(0, 1)
    divs.sort()
    return divs

VALID_DIVS = get_divisors(TARGET)

best_max = TARGET
best_factors = []
start_time = time.time()

def min_prod_distinct(start_val, count):
    """
    Calculate the minimum product of 'count' distinct integers 
    strictly greater than start_val.
    i.e., (start_val+1) * (start_val+2) * ... * (start_val+count)
    Returns infinity if it overflows or exceeds a large limit (though Python ints don't overflow, 
    we can stop if it exceeds TARGET).
    """
    prod = 1
    for i in range(1, count + 1):
        prod *= (start_val + i)
        if prod > TARGET:
            return TARGET + 1
    return prod

def dfs(rem_prod, last_val, count, current_factors):
    global best_max, best_factors, start_time
    
    if time.time() - start_time > DEADLINE:
        return

    if rem_prod == 1:
        # Found a valid decomposition
        # last_val is the max factor (since we picked in increasing order)
        if last_val < best_max:
            best_max = last_val
            best_factors = current_factors[:]
        return

    if count == 12:
        return

    # Pruning: 
    # If we have 'slots' left, the minimum possible product of the remaining factors
    # must be <= rem_prod.
    # The next factor will be > last_val.
    # Let's say we pick 'x'. Then we have 12 - count - 1 factors left.
    # The minimum product of those is min_prod_distinct(x, 12 - count - 1).
    # So we need x * min_prod_distinct(x, 12 - count - 1) <= rem_prod.
    
    # We iterate through possible next factors x.
    # x must be a divisor of rem_prod.
    # x > last_val.
    # x < best_max (to improve solution).
    
    for x in VALID_DIVS:
        if x <= last_val:
            continue
        if x >= best_max:
            break # Since VALID_DIVS is sorted, all subsequent x are >= best_max
        
        if rem_prod % x != 0:
            continue
            
        # Check if we can fill the remaining slots
        slots_left = 12 - count - 1
        if slots_left > 0:
            min_next_prod = min_prod_distinct(x, slots_left)
            if x * min_next_prod > rem_prod:
                # Even with the smallest possible distinct factors, we can't reach rem_prod
                # Actually, we need x * (product of remaining) = rem_prod.
                # So product of remaining = rem_prod / x.
                # If min_next_prod > rem_prod / x, it's impossible.
                if min_next_prod > rem_prod // x:
                    continue
                    # Note: if rem_prod // x is not integer, we already checked divisibility.
        
        # Recursive step
        current_factors.append(x)
        dfs(rem_prod // x, x, count + 1, current_factors)
        current_factors.pop()

def construct():
    global best_max, best_factors, start_time
    start_time = time.time()
    best_max = TARGET
    best_factors = []
    
    # Start DFS
    # rem_prod = TARGET
    # last_val = 12 (factors must be > 12)
    # count = 0
    # current_factors = []
    
    dfs(TARGET, 12, 0, [])
    
    if best_factors:
        return best_factors
    else:
        # Fallback: trivial solution
        return [TARGET]
