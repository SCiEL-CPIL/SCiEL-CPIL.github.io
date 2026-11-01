"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Strategy:
    - We consider vectors with squared norm 4 (i.e., norm 2). These are vectors
      with exactly two non-zero entries, each being +/-1, OR one entry being +/-2
      and rest zero. Actually, for integer vectors with norm^2 = 4:
        - Type A: two entries are +/-1, rest 0 (norm^2 = 1+1=2) -- wait, that's norm^2=2
        - Let's reconsider. We want vectors such that min pairwise distance >= max norm.
        
    Let me re-read the task: "Construct nonzero integer vectors in R^11 whose minimum 
    pairwise distance is at least the maximum vector norm."
    
    So if we have vectors v_1, ..., v_k, we need:
    min_{i != j} ||v_i - v_j|| >= max_i ||v_i||
    
    If we normalize each vector to norm 2, this becomes a kissing configuration.
    
    The current seed uses vectors with two non-zero entries of +/-1, giving norm^2 = 2, norm = sqrt(2).
    For two such vectors, the distance depends on how many coordinates they share.
    
    Let me think about this more carefully. If we use vectors with norm^2 = 4 (norm = 2),
    then we need min pairwise distance >= 2.
    
    Vectors with norm^2 = 4 in Z^11:
    - One entry is +/-2, rest 0: there are 2*11 = 22 such vectors
    - Two entries are +/-1 and +/-1 (i.e., (1,1), (1,-1), (-1,1), (-1,-1)): 
      For each pair of coordinates (i,j), there are 4 sign combinations.
      Number of pairs: C(11,2) = 55, so 55*4 = 220 such vectors
    - One entry is +/-1 and another is... no, we need sum of squares = 4.
      Possibilities: 4+0+...+0, or 1+1+1+1 (four 1's), or 1+1+2? No, 1+1+2=4 but 2 is not a square.
      Wait: 1^2 + 1^2 + 1^2 + 1^2 = 4. So also vectors with four entries of +/-1.
      Number: C(11,4) * 2^4 = 330 * 16 = 5280
    - Or 2^2 = 4: one entry +/-2.
    
    So total candidates with norm^2 = 4: 22 + 220 + 5280 = 5522.
    
    For two vectors u, v with ||u|| = ||v|| = 2, we need ||u-v|| >= 2.
    ||u-v||^2 = ||u||^2 + ||v||^2 - 2*u.v = 8 - 2*u.v >= 4
    => u.v <= 2
    
    So we need an independent set in the graph where edges connect pairs with u.v > 2.
    
    Let's implement a greedy approach with random restarts.
    """
    deadline = time.time() + 140  # safety margin
    
    # Generate all candidate vectors with norm^2 = 4
    candidates = []
    
    # Type 1: one entry is +/-2
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            candidates.append(v)
    
    # Type 2: two entries are +/-1
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # Type 3: four entries are +/-1
    # This might be too many, let's be selective or use a subset
    # C(11,4) = 330, 2^4 = 16, total 5280. Let's include them but be careful with time.
    
    # Actually, let's first try with just types 1 and 2 (242 vectors) and see if we can get a good result.
    # Then we can add type 3 if time permits.
    
    n = len(candidates)
    cand_array = np.array(candidates, dtype=np.int64)
    
    # Precompute dot products
    # dot[i,j] = cand_array[i] . cand_array[j]
    # We need to find pairs where dot > 2 (conflict)
    
    # For efficiency, let's use a bitset or just a matrix
    # n is at most 242 for types 1+2, or 5522 for all. Let's start with types 1+2.
    
    # Actually, let me include all three types but be smart about it.
    # Let's add type 3 vectors.
    
    # For now, let's work with types 1 and 2 to keep it simple and fast.
    # We can always add more later.
    
    # Compute dot product matrix
    dot = cand_array @ cand_array.T
    
    # Conflict: dot[i,j] > 2 (for i != j)
    # We want an independent set: no two selected vectors conflict.
    
    # Greedy with random restarts
    best_set = []
    best_size = 0
    
    # Shuffle candidates for diversity
    np.random.seed(42)
    
    for restart in range(500):
        if time.time() > deadline - 10:
            break
        
        # Random order
        order = np.random.permutation(n)
        
        # Greedy: pick vectors in order if they don't conflict with already picked
        selected = []
        selected_set = set()
        
        for idx in order:
            if time.time() > deadline - 5:
                break
            
            # Check if idx conflicts with any selected
            conflict = False
            for s in selected:
                if dot[idx, s] > 2:
                    conflict = True
                    break
            
            if not conflict:
                selected.append(idx)
                selected_set.add(idx)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
    
    # Local search: try to add more vectors or swap
    # For now, just return the best found
    if best_set:
        return cand_array[best_set].astype(np.int64)
    else:
        # Fallback: return the seed-like result
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = [0] * D
                        v[i], v[j] = si, sj
                        vectors.append(v)
        return np.array(vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
