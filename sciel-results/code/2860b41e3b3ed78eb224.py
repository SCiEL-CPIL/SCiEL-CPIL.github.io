# EVOLVE-BLOCK-START
import math
from functools import lru_cache

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)

def _lcm(a, b):
    return a * b // math.gcd(a, b)

# Compute LCM of all divisors
LCM = 1
for d in DIVISORS:
    LCM = _lcm(LCM, d)

def construct():
    """
    Find 10 distinct positive integers <= 1000000, each divisible by its corresponding
    divisor in DIVISORS, minimizing max - min.
    """
    
    # Search for minimum R
    # R must be at least 9
    for R in range(9, 100):
        # Precompute for each S_mod, the bitsets of valid offsets for each divisor
        # This is expensive if done naively for all S_mod.
        # But LCM is 30030. R is small.
        # We can iterate S_mod and check feasibility.
        
        for S_mod in range(LCM):
            # Check feasibility for this S_mod
            # valid_sets[i] is a list of k in [0, R] such that (S_mod + k) % DIVISORS[i] == 0
            
            valid_sets = []
            feasible = True
            for i, d in enumerate(DIVISORS):
                remainder = (-S_mod) % d
                ks = []
                k = remainder
                if k > R:
                    feasible = False
                    break
                while k <= R:
                    ks.append(k)
                    k += d
                if not ks:
                    feasible = False
                    break
                valid_sets.append(ks)
            
            if not feasible:
                continue
                
            # Use backtracking with bitsets for speed
            # Order by smallest set size
            order = sorted(range(N), key=lambda i: len(valid_sets[i]))
            
            # Convert valid_sets to bitsets for fast intersection?
            # Or just use sets. R is small, so sets are small.
            
            used_mask = 0
            assignment = [0] * N
            
            def backtrack(idx):
                nonlocal used_mask
                if idx == N:
                    return True
                i = order[idx]
                for k in valid_sets[i]:
                    bit = 1 << k
                    if not (used_mask & bit):
                        used_mask |= bit
                        assignment[i] = k
                        if backtrack(idx + 1):
                            return True
                        used_mask &= ~bit
                return False
            
            if backtrack(0):
                # Found solution
                S = S_mod
                if S < 1:
                    # S must be >= 1
                    # S = S_mod + t*LCM
                    # We want smallest S >= 1
                    if S_mod == 0:
                        S = LCM # S=0 is not positive? "positive Python integers"
                        # Usually 1..1000000. 0 is not positive.
                        # If S_mod is 0, S can be LCM.
                        # If S_mod > 0, S = S_mod.
                        pass
                    else:
                        S = S_mod
                        
                    # Adjust S to be >= 1
                    if S < 1:
                        S = LCM # Since S_mod < 1 implies S_mod=0, next is LCM
                    
                # Check S + R <= 1000000
                if S + R > 1000000:
                    # Try next S? No, S is determined by S_mod for minimal S.
                    # If minimal S is too big, no solution for this S_mod.
                    continue
                
                result = [S + assignment[i] for i in range(N)]
                # Verify
                if all(isinstance(x, int) for x in result) and len(set(result)) == N \
                   and all(1 <= x <= 1000000 for x in result) \
                   and all(result[i] % DIVISORS[i] == 0 for i in range(N)):
                    return list(result)
    
    # Fallback
    return DIVISORS[:]

# EVOLVE-BLOCK-END
