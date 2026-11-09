"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    def time_exceeded():
        return time.time() - start_time > deadline

    def is_smooth(n, prime_set):
        if n <= 1:
            return True
        for p in prime_set:
            while n % p == 0:
                n //= p
        return n == 1

    def find_solution(prime_set, target_size=12, limit=10000):
        # Precompute smooth flags for all possible sums (2 to 2*limit)
        smooth_flags = [False] * (2 * limit + 1)
        for s in range(2, 2 * limit + 1):
            if is_smooth(s, prime_set):
                smooth_flags[s] = True

        def backtrack(current_set, start_val):
            if len(current_set) == target_size:
                return current_set[:]
            
            if time_exceeded():
                return None

            # Optimization: We can skip x if it's too large?
            # Not really, but small x are better.
            
            for x in range(start_val, limit + 1):
                # Check compatibility with all existing elements
                valid = True
                for y in current_set:
                    if not smooth_flags[x + y]:
                        valid = False
                        break
                
                if valid:
                    res = backtrack(current_set + [x], x + 1)
                    if res is not None:
                        return res
                        
            return None
        
        return backtrack([], 1)

    # Candidate prime sets, ordered by size then by "likelihood" (smaller primes)
    prime_sets = [
        {2},
        {3},
        {2, 3},
        {5},
        {2, 5},
        {3, 5},
        {2, 3, 5},
        {7},
        {2, 7},
        {3, 7},
        {5, 7},
        {2, 3, 7},
        {2, 5, 7},
        {3, 5, 7},
        {2, 3, 5, 7},
        {11},
        {2, 11},
        {3, 11},
        {5, 11},
        {7, 11},
        {2, 3, 11},
        {2, 5, 11},
        {2, 7, 11},
        {3, 5, 11},
        {3, 7, 11},
        {5, 7, 11},
        {2, 3, 5, 11}
    ]
    
    for ps in prime_sets:
        if time_exceeded():
            break
        
        sol = find_solution(ps)
        
        if sol is not None and len(sol) == 12:
            # Verify
            if len(set(sol)) == 12 and all(1 <= x <= 10000 for x in sol):
                return sol
                
    # Fallback: Return a known valid set, even if not optimal
    # range(1, 13) is valid but might have many primes.
    # Let's try to return something slightly better if we have time?
    # No, just return the fallback to ensure validity.
    return list(range(1, 13))
# EVOLVE-BLOCK-END
