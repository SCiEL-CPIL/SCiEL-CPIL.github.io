"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Must finish within 120s, keep safety margin
    
    # Base vectors from the seed program: all pairs with +/-1
    base_vectors = []
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    base_vectors.append(np.array(v, dtype=np.int64))
    
    vectors = np.array(base_vectors, dtype=np.int64)
    
    # Try to add more vectors with higher norms
    # We can try vectors with 3 non-zero components
    # But we need to ensure min pairwise distance >= max norm
    
    # Let's try a greedy approach: start with base, then try to add more
    # We'll try random vectors with small integer components
    
    # First, compute current max norm squared
    norms_sq = np.sum(vectors * vectors, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # The condition is: min pairwise dist >= max norm
    # For vectors u, v: ||u-v|| >= max_norm
    # ||u-v||^2 >= max_norm^2
    
    # Let's try to add vectors with 3 non-zero components
    # These will have norm sqrt(3) if components are +/-1
    # But we need to check if they can coexist with the base vectors
    
    # Actually, let's think differently. The kissing number in 11D is 558.
    # The seed gives 11*10 = 110 vectors (each pair with 4 sign combos = 4*55=220? No)
    # Wait: combinations(range(11),2) = 55 pairs, each with 4 sign combos = 220 vectors
    
    # Let me recount: 55 pairs * 4 signs = 212? No, 55*4 = 220
    
    # The kissing number for d=11 is known to be 558.
    # We need a much better construction.
    
    # Let's try using the Leech lattice or similar constructions.
    # But we need integer vectors.
    
    # Alternative: use vectors with more non-zero components
    # Let's try a greedy algorithm to add vectors
    
    # For now, let's try to improve by adding vectors with 3 non-zero components
    # where the components are chosen to maintain the distance constraint
    
    # Let's implement a greedy addition
    current_vectors = [v.copy() for v in vectors]
    max_norm = np.sqrt(max_norm_sq)
    
    # Try adding vectors with 3 non-zero components
    # We'll use a simple heuristic: try random sign patterns for triples
    added = 0
    attempts = 0
    max_attempts = 10000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        # Pick 3 random distinct indices
        idx = random.sample(range(D), 3)
        # Pick random signs
        signs = [random.choice([1, -1]) for _ in range(3)]
        v = np.zeros(D, dtype=np.int64)
        for i, s in zip(idx, signs):
            v[i] = s
            
        # Check if this vector can be added
        # Condition: ||v - u|| >= max_norm for all u in current_vectors
        # And ||v|| <= max_norm (actually, we need min dist >= max norm of ALL vectors)
        # Wait, the condition is: min pairwise distance >= maximum vector norm
        # So if we add v, the new max norm might increase
        # We need: for all u in current_vectors, ||v-u|| >= max(max_norm, ||v||)
        
        v_norm_sq = np.sum(v * v)
        v_norm = np.sqrt(v_norm_sq)
        new_max_norm = max(max_norm, v_norm)
        
        # Check all existing vectors
        can_add = True
        for u in current_vectors:
            diff = v - u
            dist_sq = np.sum(diff * diff)
            if dist_sq < new_max_norm ** 2 - 1e-9:
                can_add = False
                break
        
        if can_add:
            current_vectors.append(v)
            added += 1
            # Update max norm if needed
            if v_norm > max_norm:
                max_norm = v_norm
    
    # Return the result
    if len(current_vectors) > 0:
        return np.array(current_vectors, dtype=np.int64)
    else:
        return vectors
# EVOLVE-BLOCK-END
