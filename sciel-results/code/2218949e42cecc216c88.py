import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # 1. Construct E8 + A3 set (252 vectors)
    # E8 minimal vectors (240) in first 8 dims
    e8_vectors = []
    # 112 vectors with two +/-1
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * 11
                v[i] = si
                v[j] = sj
                e8_vectors.append(v)
    # 128 vectors with four +/-1? No, E8 roots are 240.
    # The other 128 are of the form (1/2, ..., 1/2) with even number of minuses?
    # Wait, E8 is integral. The 240 roots are:
    # 112 with two +/-1.
    # 128 with four +/-1? No, norm squared 4.
    # Let's check E8 lattice definition.
    # E8 = { x in Z^8 | sum x_i = 0 mod 2 } U { x in (Z+1/2)^8 | sum x_i = 0 mod 2 }
    # Minimal vectors (norm squared 2):
    # From first part: two +/-1, rest 0. Sum is even.
    # Count: C(8,2)*4 = 112.
    # From second part: half-integers.
    # x_i = +/- 1/2. Sum x_i = integer.
    # Norm squared = 8 * (1/4) = 2.
    # So all 2^8 = 256 half-integer vectors?
    # Condition: sum x_i = 0 mod 2?
    # Sum of 8 half-integers is integer.
    # If even number of minuses, sum is even?
    # Let's check.
    # (1/2, ..., 1/2) sum = 4. Even.
    # (-1/2, 1/2, ..., 1/2) sum = 3. Odd.
    # We need sum to be even integer?
    # E8 condition: x in (Z+1/2)^8, sum x_i = 0 mod 2.
    # So we take the 128 vectors with even number of -1/2.
    # These are not integer vectors.
    # The task requires INTEGER vectors.
    # So we CANNOT use the half-integer E8 vectors.
    # We can only use the 112 integer E8 vectors.
    
    # So my E8 count was wrong for integer constraint.
    # Integer E8 minimal vectors: 112.
    
    # Let's re-evaluate the 252 construction.
    # If we only use integer vectors from E8, we have 112.
    # Plus 12 from A3.
    # Total 124.
    # This is worse than 220.
    
    # So the 2-non-zero construction (220) is better than the integer E8 subset.
    
    # What about other integer lattices?
    # The D8 lattice?
    # D8 = { x in Z^8 | sum x_i = 0 mod 2 }.
    # Minimal vectors: two +/-1.
    # Count 112.
    
    # It seems 220 is a strong baseline for integer vectors.
    
    # Let's check if we can improve 220 with 3-non-zeros.
    # Greedy search for 3-non-zeros.
    
    vectors_2 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vectors_2.append(v)
    
    best_set = vectors_2
    best_k = len(vectors_2)
    
    # Generate 3-non-zero vectors
    vectors_3 = []
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    vectors_3.append(v)
    
    # Greedy Independent Set on 3-non-zero conflict graph
    # Conflict if dist sq < 3
    # We want max set S such that for all u, v in S, ||u-v||^2 >= 3
    
    # Precompute norms? All are sqrt(3).
    # Dist sq < 3 means dist sq = 2 (since min non-zero dist sq is 2 for integer vectors).
    # Dist sq = 2 happens if they differ in exactly one coordinate by 2?
    # Or two coordinates by 1?
    # For 3-non-zero vectors:
    # If they share 2 indices with same signs, dist sq = 2.
    
    # Build adjacency list or just check on the fly.
    # N = 1320.
    # Greedy:
    # 1. Start with all vectors.
    # 2. Pick vector with min degree (fewest conflicts).
    # 3. Add to set.
    # 4. Remove it and its neighbors from available.
    
    available = list(range(len(vectors_3)))
    # To speed up, we can just use a boolean array.
    # But N is small.
    
    # We need to calculate degrees.
    # Degree of v is number of w such that dist_sq(v, w) < 3.
    
    # We can precompute the conflict graph?
    # 1320^2 = 1.7M pairs. Fast.
    
    if time.time() < deadline - 20:
        # Build conflict graph
        conflicts = [set() for _ in range(len(vectors_3))]
        for i in range(len(vectors_3)):
            vi = vectors_3[i]
            for j in range(i + 1, len(vectors_3)):
                vj = vectors_3[j]
                # Calculate dist sq
                # Only check if they share indices?
                # Just do it simply.
                d2 = 0
                for x in range(D):
                    diff = vi[x] - vj[x]
                    d2 += diff * diff
                    if d2 >= 3:
                        break
                if d2 < 3:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
        
        # Greedy
        selected = []
        removed = [False] * len(vectors_3)
        
        # Sort by degree? Or dynamic?
        # Dynamic: pick min degree in remaining graph.
        
        remaining = set(range(len(vectors_3)))
        
        while remaining and time.time() < deadline - 5:
            # Find node with min degree in remaining subgraph
            min_node = None
            min_deg = float('inf')
            for node in remaining:
                deg = len(conflicts[node] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_node = node
            
            if min_node is None:
                break
                
            selected.append(min_node)
            # Remove min_node and its neighbors
            to_remove = set([min_node])
            to_remove.update(conflicts[min_node])
            remaining -= to_remove
            
            if len(selected) > best_k:
                best_k = len(selected)
                # Update best_set? 
                # We need the actual vectors.
                # We can reconstruct at the end.
                
        # Reconstruct best_set if 3-non-zero is better
        if best_k > len(vectors_2):
            # Note: best_k is updated inside loop, but we didn't store the set.
            # Let's re-run or store.
            # For simplicity, if the loop finished, selected has the set.
            # But we need to check if it beat 220.
            if len(selected) > len(vectors_2):
                best_set = [vectors_3[i] for i in selected]
    
    return np.array(best_set, dtype=np.int64)

# EVOLVE-BLOCK-END
