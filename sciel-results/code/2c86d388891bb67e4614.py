"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s kill
    
    N = 256
    
    # Precompute prime factors for 1..N
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not prime_factors[i]:
            for j in range(i, N + 1, i):
                prime_factors[j].add(i)
    # prime_factors[1] remains empty
    
    # Get all numbers 1..N, sorted by value descending
    numbers = list(range(N, 0, -1))
    
    # Precompute compatibility: for each number, which other numbers conflict with it
    # conflict_mask[i] is a bitmask of numbers that conflict with numbers[i]
    # But 256 numbers is too many for bitmask in Python efficiently, let's use sets
    
    # For each number, precompute the set of prime factors
    # Two numbers are compatible if their prime factor sets are disjoint
    
    best_sum = 0
    best_set = []
    
    # Recursive branch and bound
    # State: index in numbers list, current selected set (as list), current sum, blocked primes (set)
    # We process numbers in order (descending value)
    
    # To speed up, precompute for each number its prime factor set
    pf = prime_factors  # shorthand
    
    # Upper bound function: given current sum and blocked primes, 
    # estimate max additional sum we can get
    def upper_bound(idx, current_sum, blocked):
        """
        Upper bound: current_sum + sum of values of numbers from idx..end 
        that are compatible with blocked primes.
        This is a loose bound but fast.
        """
        ub = current_sum
        for i in range(idx, len(numbers)):
            n = numbers[i]
            if n == 1:
                if 1 not in blocked:  # 1 has no prime factors, always compatible
                    ub += 1
            else:
                # Check if all prime factors of n are not in blocked
                if pf[n].isdisjoint(blocked):
                    ub += n
        return ub
    
    # Better upper bound: for each unblocked prime, take the best number that only uses that prime
    # Actually, let's use a simpler but tighter bound
    
    def tight_upper_bound(idx, current_sum, blocked):
        """
        For each prime not in blocked, find the best number >= idx that has this prime
        and all its other primes also not in blocked. Take max over such numbers per prime,
        but each number can only be counted once. This is complex.
        
        Simpler: greedy - go through remaining numbers, if compatible with blocked, add to bound.
        But also ensure pairwise compatibility among added numbers (greedy).
        """
        ub = current_sum
        # Greedy: add numbers in order if compatible with blocked AND with previously added in this bound
        temp_blocked = set(blocked)
        for i in range(idx, len(numbers)):
            n = numbers[i]
            if n == 1:
                ub += 1
                # 1 doesn't block any primes
            else:
                if pf[n].isdisjoint(temp_blocked):
                    ub += n
                    temp_blocked.update(pf[n])
        return ub
    
    # DFS with branch and bound
    # We use a stack-based approach to avoid recursion depth issues
    # State: (idx, selected_list, current_sum, blocked_primes_set)
    
    # Actually, let's use recursion with sys.setrecursionlimit
    import sys
    sys.setrecursionlimit(10000)
    
    def dfs(idx, selected, current_sum, blocked):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
        
        # Update best
        if current_sum > best_sum:
            best_sum = current_sum
            best_set = list(selected)
        
        if idx >= len(numbers):
            return
        
        # Upper bound check
        ub = tight_upper_bound(idx, current_sum, blocked)
        if ub <= best_sum:
            return
        
        n = numbers[idx]
        
        # Option 1: skip n
        dfs(idx + 1, selected, current_sum, blocked)
        
        # Option 2: include n (if compatible)
        if n == 1:
            # 1 is always compatible
            selected.append(1)
            dfs(idx + 1, selected, current_sum + 1, blocked)
            selected.pop()
        else:
            if pf[n].isdisjoint(blocked):
                selected.append(n)
                new_blocked = blocked | pf[n]
                dfs(idx + 1, selected, current_sum + n, new_blocked)
                selected.pop()
    
    # Start DFS
    dfs(0, [], 0, set())
    
    # Convert to list of ints
    result = list(best_set)
    # Ensure all are Python ints
    result = [int(x) for x in result]
    
    return result
# EVOLVE-BLOCK-END
