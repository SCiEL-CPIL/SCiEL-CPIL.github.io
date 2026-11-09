"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    """
    Construct a valid two-coloring of triples on n vertices with no red K4^(3)
    and no blue K5^(3), maximizing n.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    # We aim for n=34. If we find a valid one, return it.
    # If not, try to find the largest n < 34 that is valid.
    
    def _validate_fast(colors, n):
        """
        Check if a coloring of triples on n vertices is valid.
        colors: list of 0/1 in lexicographic order of triples (a,b,c), a<b<c.
        Returns True if no red K4^(3) and no blue K5^(3).
        """
        # Build a 3D lookup structure for O(1) access
        # Since n is small (<=40), we can use a dictionary or a flat array.
        # A flat array indexed by (a, b, c) is tricky without a mapping.
        # Let's use a set of red triples and blue triples for simplicity, 
        # but optimized by precomputing indices.
        
        # Actually, for speed, let's build a 3D numpy array or similar.
        # n is up to 40. 40^3 = 64000. That's small.
        # Let's create a 3D array of shape (n, n, n) filled with -1.
        # Then fill it with the colors.
        
        lookup = np.full((n, n, n), -1, dtype=np.int8)
        
        idx = 0
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    lookup[a, b, c] = colors[idx]
                    lookup[b, a, c] = colors[idx] # Symmetry not strictly needed if we always access sorted, but safe
                    lookup[b, c, a] = colors[idx]
                    lookup[c, a, b] = colors[idx]
                    lookup[c, b, a] = colors[idx]
                    lookup[a, c, b] = colors[idx]
                    idx += 1
        
        # Check Red K4: 4 vertices, all 4 triples red (0)
        # Iterate over all 4-sets
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            # Triples: (a,b,c), (a,b,d), (a,c,d), (b,c,d)
            if (lookup[a, b, c] == 0 and 
                lookup[a, b, d] == 0 and 
                lookup[a, c, d] == 0 and 
                lookup[b, c, d] == 0):
                return False
        
        # Check Blue K5: 5 vertices, all 10 triples blue (1)
        for quint in combinations(range(n), 5):
            v0, v1, v2, v3, v4 = quint
            # Triples:
            # (v0,v1,v2), (v0,v1,v3), (v0,v1,v4)
            # (v0,v2,v3), (v0,v2,v4), (v0,v3,v4)
            # (v1,v2,v3), (v1,v2,v4), (v1,v3,v4)
            # (v2,v3,v4)
            
            # Quick check: if any is not blue, skip
            # To speed up, we can check the first few
            if lookup[v0, v1, v2] != 1: continue
            if lookup[v0, v1, v3] != 1: continue
            if lookup[v0, v1, v4] != 1: continue
            if lookup[v0, v2, v3] != 1: continue
            if lookup[v0, v2, v4] != 1: continue
            if lookup[v0, v3, v4] != 1: continue
            if lookup[v1, v2, v3] != 1: continue
            if lookup[v1, v2, v4] != 1: continue
            if lookup[v1, v3, v4] != 1: continue
            if lookup[v2, v3, v4] != 1: continue
            
            # All 10 are blue
            return False
            
        return True

    def _generate_candidate(n, seed_idx=0):
        """
        Generate a candidate coloring for n vertices.
        Uses a combination of algebraic structures and randomness.
        """
        random.seed(seed_idx)
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        colors = [0] * num_triples
        
        # Strategy 1: Modular arithmetic base
        # Try to find a modulus m such that sum % m == 0 works well,
        # or use it as a starting point.
        
        # Let's try a few different initialization strategies
        strategy = seed_idx % 5
        
        if strategy == 0:
            # Sum mod m
            m = n
            for i, (a, b, c) in enumerate(triples):
                colors[i] = 0 if (a + b + c) % m == 0 else 1
        elif strategy == 1:
            # Random
            colors = [random.randint(0, 1) for _ in range(num_triples)]
        elif strategy == 2:
            # Product mod m
            m = n
            for i, (a, b, c) in enumerate(triples):
                colors[i] = 0 if (a * b * c) % m == 0 else 1
        elif strategy == 3:
            # Difference based
            for i, (a, b, c) in enumerate(triples):
                # Check if (b-a) == (c-b) i.e. arithmetic progression
                if b - a == c - b:
                    colors[i] = 0
                else:
                    colors[i] = 1
        else:
            # XOR based (bitwise)
            for i, (a, b, c) in enumerate(triples):
                colors[i] = (a ^ b ^ c) & 1

        return colors

    def _local_search(colors, n, max_iters=10000, time_limit=5.0):
        """
        Perform local search to reduce violations.
        Returns (colors, violations).
        """
        start_ls = time.time()
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        
        # Precompute for each triple index, the list of 4-sets it belongs to
        # and the list of 5-sets it belongs to?
        # That's expensive.
        
        # Instead, let's use a "violation count" approach.
        # We maintain a count of red triples in each 4-set and blue triples in each 5-set.
        
        # Map 4-set to an index
        # Map 5-set to an index
        
        # For n=34:
        # C(34,4) = 46376
        # C(34,5) = 278256
        # This is manageable.
        
        # Let's map 4-sets to indices 0..C(n,4)-1
        # and 5-sets to indices 0..C(n,5)-1
        
        # To do this efficiently, we can precompute the indices.
        # Or we can just use a dictionary.
        
        # Let's try a simpler approach first:
        # Calculate total violations.
        # Pick a random violating 4-set or 5-set.
        # Pick a random triple in it.
        # Flip it.
        # Recalculate violations locally?
        
        # Actually, full recalculation is O(C(n,4) + C(n,5)) which is ~300k ops.
        # In Python, 300k ops might take ~0.1-0.5s.
        # So we can do ~10-50 iterations per second.
        # That's too slow for 10000 iterations.
        
        # We need incremental updates.
        
        # Let's build the incremental structure.
        
        # 1. Map each 4-set to a unique ID
        # 2. Map each 5-set to a unique ID
        # 3. For each triple, store which 4-sets and 5-sets it belongs to.
        
        # This precomputation takes time, but is worth it.
        
        # However, implementing this correctly and efficiently in Python within the time limit
        # is risky.
        
        # Alternative: Use a SAT solver approach? No, we only have stdlib/np/scipy.
        
        # Let's try a different local search:
        # 1. Calculate all violations (list of violating 4-sets and 5-sets).
        # 2. If none, done.
        # 3. Pick a random violating set.
        # 4. Pick a random triple in it.
        # 5. Flip the triple.
        # 6. Go to 1.
        
        # To make step 1 faster, we can optimize _validate_fast.
        # But even if _validate_fast takes 0.5s, we can only do 100 iterations in 50s.
        
        # Let's try to optimize _validate_fast further.
        
        # Actually, the bottleneck is the loops in Python.
        # Can we vectorize?
        
        # For Red K4:
        # We have a 3D array L.
        # We want to find a,b,c,d such that L[a,b,c]==0, L[a,b,d]==0, L[a,c,d]==0, L[b,c,d]==0.
        
        # This is hard to vectorize directly.
        
        # Let's stick to the incremental approach if possible, or just do many random restarts
        # with a simpler local search that doesn't require full validation every step.
        
        # Heuristic Local Search:
        # 1. Start with a candidate.
        # 2. Calculate violations.
        # 3. If 0, return.
        # 4. Pick a random violating 4-set or 5-set.
        # 5. Pick a random triple in it.
        # 6. Flip it.
        # 7. Repeat.
        
        # To avoid full validation every step, we can just check if the *current* violation is fixed.
        # But other violations might appear.
        
        # Let's try this:
        
        current_colors = colors[:]
        
        for i in range(max_iters):
            if time.time() > start_ls + time_limit:
                break
            
            # Check validity
            if _validate_fast(current_colors, n):
                return current_colors, 0
            
            # Find a violation
            # We need to find one quickly.
            # Let's modify _validate_fast to return the first violation found.
            
            # Let's write a helper to find one violation
            lookup = np.full((n, n, n), -1, dtype=np.int8)
            idx = 0
            for a in range(n):
                for b in range(a + 1, n):
                    for c in range(b + 1, n):
                        lookup[a, b, c] = current_colors[idx]
                        idx += 1
            
            violation_found = False
            # Check Red K4
            for quad in combinations(range(n), 4):
                a, b, c, d = quad
                if (lookup[a, b, c] == 0 and 
                    lookup[a, b, d] == 0 and 
                    lookup[a, c, d] == 0 and 
                    lookup[b, c, d] == 0):
                    # Found a red K4
                    # Pick a random triple from the 4
                    # Triples: (a,b,c), (a,b,d), (a,c,d), (b,c,d)
                    # Find their indices in the list
                    # We need a map from triple to index
                    # Let's build it once outside?
                    
                    # For now, let's just flip (a,b,c)
                    # Find index of (a,b,c)
                    # This is O(n^3) to find. Too slow.
                    
                    # We need the index map.
                    break
            
            if not violation_found:
                # Check Blue K5
                for quint in combinations(range(n), 5):
                    v0, v1, v2, v3, v4 = quint
                    if (lookup[v0, v1, v2] == 1 and 
                        lookup[v0, v1, v3] == 1 and 
                        lookup[v0, v1, v4] == 1 and 
                        lookup[v0, v2, v3] == 1 and 
                        lookup[v0, v2, v4] == 1 and 
                        lookup[v0, v3, v4] == 1 and 
                        lookup[v1, v2, v3] == 1 and 
                        lookup[v1, v2, v4] == 1 and 
                        lookup[v1, v3, v4] == 1 and 
                        lookup[v2, v3, v4] == 1):
                        # Found a blue K5
                        # Flip (v0,v1,v2)
                        break
            
            # If we didn't find a violation in the quick check (unlikely if _validate_fast said invalid),
            # or if we did, we need to flip.
            
            # This approach is flawed because finding the index of the triple is expensive.
            
            # Let's precompute the index map.
            
        return current_colors, -1 # -1 means not solved

    # Precompute index map for triples
    # We'll do this for each n we try.
    
    best_n = 0
    best_colors = None
    
    # Try n=34 first
    for n in range(34, 10, -1):
        if time.time() > deadline:
            break
            
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Try multiple initializations
        for seed in range(50):
            if time.time() > deadline:
                break
                
            colors = _generate_candidate(n, seed)
            
            # Check if valid
            if _validate_fast(colors, n):
                best_n = n
                best_colors = colors
                break
            
            # If not, try local search
            # We need a faster local search.
            # Let's implement the incremental one properly.
            
            # Precompute: for each triple index, list of 4-sets (as tuples) and 5-sets it belongs to?
            # No, that's too much memory/time.
            
            # Let's try a "greedy" approach:
            # 1. Count violations.
            # 2. If 0, done.
            # 3. Pick a random triple.
            # 4. Flip it.
            # 5. If violations decreased, keep. Else revert.
            
            # This requires counting violations efficiently.
            
            # Let's try to optimize _validate_fast to count violations, not just check.
            
            # For now, let's just try more seeds.
            
            # If we fail to find a valid coloring for n=34 after 50 seeds,
            # we might need a better search.
            
            # Let's add a simple local search that works on the list directly.
            
            # Improved Local Search:
            # 1. Calculate violation count.
            # 2. If 0, done.
            # 3. Pick a random triple.
            # 4. Flip it.
            # 5. Calculate new violation count.
            # 6. If improved, keep. Else revert.
            
            # To make this fast, we need to calculate violation count quickly.
            # _validate_fast is O(n^4) roughly.
            
            # Let's just do it and see if it fits in time.
            
            def count_violations(colors, n, triple_to_idx, triples):
                lookup = np.full((n, n, n), -1, dtype=np.int8)
                idx = 0
                for a in range(n):
                    for b in range(a + 1, n):
                        for c in range(b + 1, n):
                            lookup[a, b, c] = colors[idx]
                            idx += 1
                
                count = 0
                for quad in combinations(range(n), 4):
                    a, b, c, d = quad
                    if (lookup[a, b, c] == 0 and 
                        lookup[a, b, d] == 0 and 
                        lookup[a, c, d] == 0 and 
                        lookup[b, c, d] == 0):
                        count += 1
                
                for quint in combinations(range(n), 5):
                    v0, v1, v2, v3, v4 = quint
                    if (lookup[v0, v1, v2] == 1 and 
                        lookup[v0, v1, v3] == 1 and 
                        lookup[v0, v1, v4] == 1 and 
                        lookup[v0, v2, v3] == 1 and 
                        lookup[v0, v2, v4] == 1 and 
                        lookup[v0, v3, v4] == 1 and 
                        lookup[v1, v2, v3] == 1 and 
                        lookup[v1, v2, v4] == 1 and 
                        lookup[v1, v3, v4] == 1 and 
                        lookup[v2, v3, v4] == 1):
                        count += 1
                        
                return count

            # This is too slow for local search.
            
            # Let's rely on the fact that maybe a good seed will work.
            # Or maybe we can use a known construction.
            
            # Known: The published construction has 34 points.
            # It is often based on cyclic groups or similar.
            
            # Let's try: vertices 0..n-1.
            # Color (a,b,c) red if a+b+c is a quadratic residue mod p?
            # Or if a+b+c is in a specific set.
            
            # Let's try a few specific algebraic patterns for n=34.
            
            if n == 34:
                # Try pattern: red if (a+b+c) % 34 is in [0, 16] (half the residues)
                # This is just random-ish.
                
                # Try: red if (a+b+c) % 34 < 17
                colors_alt = []
                for a in range(n):
                    for b in range(a + 1, n):
                        for c in range(b + 1, n):
                            colors_alt.append(0 if (a + b + c) % n < n // 2 else 1)
                
                if _validate_fast(colors_alt, n):
                    best_n = n
                    best_colors = colors_alt
                    break

    if best_colors is not None:
        return {'n': best_n, 'colors': best_colors}
    
    # Fallback: return a small valid coloring
    # n=9 with mod 3 is known to be valid?
    # Let's check n=9.
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 
              for a in range(n) for b in range(a+1, n) for c in range(b+1, n)]
    
    if _validate_fast(colors, n):
        return {'n': n, 'colors': colors}
    
    # If even that fails (unlikely), return n=1
    return {'n': 1, 'colors': [0]}

# EVOLVE-BLOCK-END
