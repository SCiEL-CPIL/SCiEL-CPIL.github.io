"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Helper to check if a coloring is valid
    def is_valid(n, colors):
        # colors is a list of 0s and 1s for all triples in lex order
        # Red K4^(3): 4 vertices, all 4 triples red (0)
        # Blue K5^(3): 5 vertices, all 10 triples blue (1)
        
        # Map triple (a,b,c) to index
        # Index calculation for lex order of combinations(range(n), 3)
        # This is slow in pure python if done naively. 
        # Precompute indices for speed? 
        # For n up to 40, C(40,3) = 9880. 
        # Checking all C(40,4)=91390 subsets for red and C(40,5)=658008 for blue is too slow in pure python if done repeatedly.
        
        # Optimization: Use numpy or precomputed structures?
        # Let's use a faster validity check.
        # We can represent the coloring as a 3D array or use bitsets.
        # Given the time limit, we need a fast check.
        
        # Let's stick to a reasonably optimized python check first, 
        # but we might need to limit n during search if it's too slow.
        
        # Actually, for n=34, C(34,5) is ~278k. Checking 10 triples for each is ~2.7M ops. 
        # Doing this in pure python every iteration of SA is too slow.
        
        # Strategy: 
        # 1. Start with a known valid small n (e.g., n=9 from seed).
        # 2. Try to extend n by adding a vertex. 
        # 3. When extending, we only need to check new triples involving the new vertex.
        # 4. Use local search to fix conflicts.
        
        # However, implementing a full incremental SA with fast validity checks in Python within 110s is tricky.
        # Let's try a simpler approach: 
        # Use a heuristic construction that is known to work for larger n?
        # The seed uses (a+b+c)%3. 
        # Known constructions for R(4,5;3) often use finite fields or specific combinatorial designs.
        # A simple algebraic construction might exist for n=34?
        
        # Let's try a different algebraic construction first, as it's instant.
        # If we can find a valid n=34 algebraic coloring, we are done.
        
        # Common construction: Color (a,b,c) red if a+b+c is in a specific set?
        # Or based on GF(2) or GF(3)?
        
        # Let's try to search for a valid coloring using a fast bitset-based checker in NumPy.
        
        n_target = 34
        # Initialize with random coloring for n_target
        # C(34,3) = 5984
        colors = np.random.randint(0, 2, size=(n_target, n_target, n_target))
        
        # This 3D array is wasteful. Let's use a flat array and index mapping.
        # But for speed in NumPy, a 3D boolean mask might be faster for checking K4 and K5?
        
        # Let's define a function to check validity using NumPy vectorization.
        # Red K4: For every 4-tuple (a,b,c,d), colors[a,b,c] AND colors[a,b,d] AND colors[a,c,d] AND colors[b,c,d] must be False (i.e., at least one is Blue/1).
        # Wait, "No 4 vertices may have all four triples red". 
        # So for any 4 distinct vertices, it is NOT the case that all 4 triples are 0.
        # This means for any 4 vertices, at least one triple is 1.
        
        # Blue K5: For every 5-tuple, it is NOT the case that all 10 triples are 1.
        # So for any 5 vertices, at least one triple is 0.
        
        # Let's try to construct a valid coloring using a greedy algorithm with backtracking or simple local search.
        # Given the complexity, maybe a simpler heuristic:
        # Try to extend the n=9 solution?
        
        # Let's try a specific known construction if I can recall one.
        # Actually, let's just do a random restart local search with a fast validity check.
        
        def fast_validity_check(n, colors_flat):
            # colors_flat: array of length C(n,3)
            # This is still hard to vectorize easily without precomputed indices.
            
            # Let's use a different approach: 
            # Start with n=10, try to grow.
            # Or just try random colorings for n=34 and check validity.
            # Probability of random coloring being valid is low.
            
            pass
        
        # Let's try a deterministic construction based on modular arithmetic which is fast to generate and check.
        # The seed used (a+b+c)%3. 
        # Let's try (a+b+c)%k for different k or different functions.
        
        # Try to find a valid n using a simple loop over potential algebraic rules.
        # Rule: color = f(a,b,c) % 2
        # Candidates for f:
        # 1. a+b+c
        # 2. a*b + b*c + c*a
        # 3. (a+b+c) % 5 < 2 ?
        
        # Let's test a few simple rules for n=34.
        rules = [
            lambda a,b,c: (a+b+c) % 2,
            lambda a,b,c: (a*b + b*c + c*a) % 2,
            lambda a,b,c: (a+b+c) % 3 < 1, # Red if 0
            lambda a,b,c: (a+b+c) % 4 < 2,
            lambda a,b,c: (a*b*c) % 2,
            lambda a,b,c: (a+b) % 2,
            lambda a,b,c: (b+c) % 2,
            lambda a,b,c: (a+c) % 2,
        ]
        
        best_n = 9
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
        
        def check_validity(n, colors):
            # colors is list of 0/1
            # Convert to numpy for speed? 
            # For n=34, C(34,4) is ~59k, C(34,5) is ~278k.
            # Pure python check:
            # For each 4-comb: if all 4 triples are 0 -> invalid
            # For each 5-comb: if all 10 triples are 1 -> invalid
            
            # Precompute index map for triples
            idx_map = {}
            k = 0
            for a in range(n):
                for b in range(a+1, n):
                    for c in range(b+1, n):
                        idx_map[(a,b,c)] = k
                        k += 1
            
            # Check Red K4
            for quad in combinations(range(n), 4):
                a,b,c,d = quad
                if (colors[idx_map[(a,b,c)]] == 0 and 
                    colors[idx_map[(a,b,d)]] == 0 and 
                    colors[idx_map[(a,c,d)]] == 0 and 
                    colors[idx_map[(b,c,d)]] == 0):
                    return False
            
            # Check Blue K5
            for quint in combinations(range(n), 5):
                a,b,c,d,e = quint
                # 10 triples
                triples = [
                    (a,b,c), (a,b,d), (a,b,e),
                    (a,c,d), (a,c,e), (a,d,e),
                    (b,c,d), (b,c,e), (b,d,e),
                    (c,d,e)
                ]
                if all(colors[idx_map[t]] == 1 for t in triples):
                    return False
            
            return True

        # Test rules for n=34
        for rule in rules:
            colors_test = [int(rule(a,b,c)) for a,b,c in combinations(range(34), 3)]
            if check_validity(34, colors_test):
                return {'n': 34, 'colors': colors_test}
        
        # If simple rules fail, try local search on n=34
        # But local search in Python might be too slow for 110s.
        
        # Let's try to extend the seed's n=9 solution?
        # Or just return the best valid n found by testing small n with random colors?
        
        # Actually, let's try a few more rules for n=34.
        more_rules = [
            lambda a,b,c: (a+b+c) % 5 < 2,
            lambda a,b,c: (a+b+c) % 5 == 0,
            lambda a,b,c: (a*b + b*c + c*a + a+b+c) % 2,
            lambda a,b,c: (a+b+c) % 7 < 3,
            lambda a,b,c: (a+b+c) % 7 == 0,
        ]
        
        for rule in more_rules:
            colors_test = [int(rule(a,b,c)) for a,b,c in combinations(range(34), 3)]
            if check_validity(34, colors_test):
                return {'n': 34, 'colors': colors_test}
            
            if time.time() > deadline - 20:
                break
        
        # Fallback: Return the seed result or a slightly improved one if found
        # Let's try to find a valid n > 9 using random search for small n?
        # No, let's just return the seed if we can't find 34 quickly.
        # But wait, the task says "Maximize n". 
        # If I can't find 34, I should return the best I can.
        
        # Let's try one more thing: 
        # Is there a known simple construction for R(4,5;3)?
        # The bound is 34. 
        # Maybe a construction based on Z_34?
        
        # Let's try (a+b+c) % 34 < 17 ?
        rule_large = lambda a,b,c: (a+b+c) % 34 < 17
        colors_test = [int(rule_large(a,b,c)) for a,b,c in combinations(range(34), 3)]
        if check_validity(34, colors_test):
            return {'n': 34, 'colors': colors_test}
            
        # If all else fails, return seed
        return {'n': 9, 'colors': best_colors}

# EVOLVE-BLOCK-END
