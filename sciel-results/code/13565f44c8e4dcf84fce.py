"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys

def construct():
    n = 192
    k = 3
    
    # Initialize the set of all candidates
    S = set(range(1, n + 1))
    
    # Precompute mod_maps for each a in 1..n
    # mod_maps[a][r] is a set of elements in S such that element % a == r
    # We only need this for a in S, but S changes. 
    # However, we can just maintain it for all a in 1..n and update as S changes.
    # Actually, we only check a in S.
    
    mod_maps = {}
    for a in range(1, n + 1):
        mod_maps[a] = [set() for _ in range(a)]
        for x in S:
            mod_maps[a][x % a].add(x)
    
    # Helper to remove an element from S and mod_maps
    def remove_element(v):
        S.discard(v)
        for a in range(1, n + 1):
            mod_maps[a][v % a].discard(v)
            
    # Helper to find a violation
    # Returns (a, x, y, z) or None
    def find_violation():
        # We iterate through 'a' in S. 
        # To find violations quickly, we might want to check small 'a' first 
        # because they are more likely to divide sums.
        # Let's iterate through sorted(S) to prioritize small divisors.
        for a in sorted(S):
            others = [v for v in S if v != a]
            # We need x, y, z in others, distinct, such that (x+y+z) % a == 0
            # Iterate x, y from others
            # To optimize, we can use the mod_map for 'a'
            # But we must ensure x, y, z are distinct and != a.
            
            # Iterate pairs (x, y) from others
            # This is O(N^2) per a.
            # We can break early if found.
            for i in range(len(others)):
                x = others[i]
                for j in range(i + 1, len(others)):
                    y = others[j]
                    # We need z in others, z != x, z != y, such that (x+y+z) % a == 0
                    # z % a == (-(x+y)) % a
                    r = (-(x + y)) % a
                    candidates = mod_maps[a][r]
                    # candidates contains all v in S such that v % a == r
                    # We need z in candidates such that z != x and z != y and z != a
                    # Note: a % a == 0, so if r == 0, a is in candidates. 
                    # But we excluded a from 'others', so x,y != a.
                    # We just need to check if there is a z in candidates \ {x, y, a}
                    
                    # Quick check: size of candidates
                    # If len(candidates) == 0, continue
                    # If len(candidates) == 1, check if the one element is x, y, or a
                    # If len(candidates) >= 2, it's likely we can find one, unless {x,y,a} covers all.
                    
                    if not candidates:
                        continue
                    
                    # We need to find z in candidates such that z not in {x, y, a}
                    # Since candidates is a set, we can iterate if small, or check logic.
                    # Max size of candidates is N/a. For a=1, size 192.
                    # For a=1, r=0, candidates=S. We need z != x, y. 
                    # Since |S| >= 4 (if we found x,y), there is always a z.
                    
                    # Efficient check:
                    # Let forbidden = {x, y, a}
                    # We need candidates - forbidden to be non-empty.
                    
                    # Optimization: 
                    # If len(candidates) > len(forbidden), then definitely non-empty.
                    # len(forbidden) <= 3.
                    # So if len(candidates) >= 4, we are good.
                    if len(candidates) >= 4:
                        # We need to actually pick a z to return.
                        # Pick any z in candidates not in forbidden
                        # We can just pick one and verify?
                        # For the purpose of finding *a* violation, we just need to know it exists.
                        # But we need to return the specific tuple.
                        # Let's find one.
                        for z in candidates:
                            if z != x and z != y and z != a:
                                return (a, x, y, z)
                    else:
                        # len(candidates) < 4
                        # Check explicitly
                        for z in candidates:
                            if z != x and z != y and z != a:
                                return (a, x, y, z)
                            
        return None

    # Main loop
    while True:
        violation = find_violation()
        if violation is None:
            break
        a, x, y, z = violation
        # Heuristic: Remove the smallest element to maximize the size of the remaining set
        # Small numbers are more likely to cause other violations.
        candidates_to_remove = [a, x, y, z]
        # Remove the smallest
        v_to_remove = min(candidates_to_remove)
        remove_element(v_to_remove)
        
    return list(S)

# EVOLVE-BLOCK-END
