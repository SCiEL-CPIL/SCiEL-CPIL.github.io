"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}

def construct():
    # Primes <= 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Determine bits needed for each prime's exponent in the maximum possible product (40!)
    # Max exponent of p in 40! is sum_{k>=1} floor(40/p^k)
    max_exps = []
    for p in primes:
        exp = 0
        power = p
        while power <= 40:
            exp += 40 // power
            power *= p
        max_exps.append(exp)
        
    # Calculate bits needed
    bits = [max(1, math.ceil(math.log2(e + 1))) if e > 0 else 1 for e in max_exps]
    
    # Calculate shifts
    shifts = []
    current_shift = 0
    for b in bits:
        shifts.append(current_shift)
        current_shift += b
        
    # Map each number 2..40 to a packed integer
    num_to_val = {}
    for n in range(2, 41):
        val = 0
        temp = n
        for i, p in enumerate(primes):
            if temp % p == 0:
                cnt = 0
                while temp % p == 0:
                    temp //= p
                    cnt += 1
                val |= (cnt << shifts[i])
        num_to_val[n] = val
        
    candidates = list(range(2, 41))
    # Sort candidates by value to have a deterministic order
    # Maybe sort by number of prime factors or magnitude?
    # Let's try sorting by the packed value or just numerical order.
    # Numerical order is fine.
    
    best_set = []
    
    # Backtracking search
    def search(current_set, current_sums, start_idx):
        nonlocal best_set
        
        # Prune: if we can't beat best, return
        if len(current_set) + (len(candidates) - start_idx) <= len(best_set):
            return
            
        # Update best
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
        # Try adding each candidate from start_idx onwards
        for i in range(start_idx, len(candidates)):
            c = candidates[i]
            v = num_to_val[c]
            
            # Check if adding v creates collisions
            # New sums are current_sums and {s + v for s in current_sums}
            # If any s + v is in current_sums, or s1 + v == s2 + v (impossible if s1!=s2)
            # Actually, we just need to check if {s + v for s in current_sums} intersects current_sums
            # Or more simply, if the size of the union is 2 * len(current_sums)
            
            # Optimization: check intersection
            # new_sums = set()
            # collision = False
            # for s in current_sums:
            #     ns = s + v
            #     if ns in current_sums:
            #         collision = True
            #         break
            #     new_sums.add(ns)
            
            # Even faster:
            # If v is 0, collision (not possible here)
            # We can check if any s in current_sums is such that s+v in current_sums.
            
            collision = False
            for s in current_sums:
                if (s + v) in current_sums:
                    collision = True
                    break
            
            if not collision:
                # Construct new sums
                new_sums = set(current_sums)
                for s in current_sums:
                    new_sums.add(s + v)
                
                search(current_set + [c], new_sums, i + 1)
                
    # Start search
    search([], {0}, 0)
    
    # Convert to list of ints
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
