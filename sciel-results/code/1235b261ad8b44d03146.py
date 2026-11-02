import numpy as np
import itertools
import time
from collections import defaultdict

def construct():
    """
    Constructs a set of nonzero integer vectors in R^11 such that the minimum 
    pairwise distance is at least the maximum vector norm.
    
    Strategy:
    1. Generate all weight-4 vectors (norm 2). There are C(11,4)*2^4 = 5280.
    2. Build a conflict graph where edges connect vectors with dot product > 2.
       (Condition: ||u-v||^2 >= 4 => <u,v> <= 2 for weight-4 vectors).
       Conflicts occur when two weight-4 vectors share 3 indices and have the same signs on those 3.
    3. Use a fast greedy Maximum Independent Set (MIS) heuristic on the conflict graph.
       - Use bitsets (Python integers) for fast neighbor removal.
       - Try multiple orderings (random shuffles) to find a larger independent set.
    4. Attempt to add compatible weight-2 vectors (norm sqrt(2)).
       - A weight-2 vector u is compatible with a weight-4 vector v if <u,v> <= 1.
       - Since max norm becomes 2 (from weight-4), the condition is ||u-v||^2 >= 4.
       - 2 + 4 - 2<u,v> >= 4 => <u,v> <= 1.
       - We check this for all selected weight-4 vectors.
    5. Return the combined set.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for safety

    D = 11
    
    # --- Step 1: Generate all weight-4 vectors ---
    # Represent vector by (support_tuple, signs_tuple)
    # support: tuple of 4 indices in [0, 10]
    # signs: tuple of 4 values in {+1, -1}
    
    w4_vectors = []
    vec_to_idx = {}
    
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            idx = len(w4_vectors)
            w4_vectors.append((support, signs))
            vec_to_idx[(support, signs)] = idx
            
    n_w4 = len(w4_vectors) # 5280

    # --- Step 2: Build Conflict Graph using Bitsets ---
    # Conflict if share 3 indices and same signs on those 3.
    # We can precompute for each vector the bitset of its neighbors.
    
    # Map from (3-support, 3-signs) to list of vector indices
    triplet_map = defaultdict(list)
    
    for idx, (support, signs) in enumerate(w4_vectors):
        # Generate all 4 triplets
        for k in range(4):
            # Triplet excluding index k in the support/signs
            trip_sup = tuple(support[j] for j in range(4) if j != k)
            trip_sig = tuple(signs[j] for j in range(4) if j != k)
            key = (trip_sup, trip_sig)
            triplet_map[key].append(idx)
            
    # Initialize neighbor bitsets
    # Python integers are arbitrary precision, so we can use them as bitsets.
    # Bit i is 1 if vector i is a neighbor.
    neighbor_bits = [0] * n_w4
    
    for idx, (support, signs) in enumerate(w4_vectors):
        for k in range(4):
            trip_sup = tuple(support[j] for j in range(4) if j != k)
            trip_sig = tuple(signs[j] for j in range(4) if j != k)
            key = (trip_sup, trip_sig)
            # All vectors in this triplet share these 3 indices and signs.
            # They are conflicts with idx.
            for other_idx in triplet_map[key]:
                if other_idx != idx:
                    neighbor_bits[idx] |= (1 << other_idx)
                    
    # --- Step 3: Greedy MIS with Bitsets ---
    # We want to select a set of indices such that no two are connected in the conflict graph.
    # Greedy:
    # 1. Maintain a bitset of available candidates.
    # 2. Pick a candidate (e.g., lowest index, or random, or min-degree in remaining).
    # 3. Add to solution.
    # 4. Remove it and all its neighbors from available candidates.
    
    # To speed up, we can precompute degrees and try sorting by degree.
    # However, the degree in the *remaining* graph changes.
    # A simple heuristic: Random restarts with random ordering.
    # Or: Sort by initial degree (ascending) and try that.
    # Let's try a few strategies.
    
    best_set = set()
    best_size = 0
    
    # Strategy 1: Sort by initial degree (ascending)
    degrees = [bin(neighbor_bits[i]).count('1') for i in range(n_w4)]
    order1 = sorted(range(n_w4), key=lambda i: degrees[i])
    
    def greedy_mis(order):
        available = (1 << n_w4) - 1  # All bits set
        selected = []
        # We need to iterate through the order, but if a vertex is removed, we skip it.
        # However, the "remove neighbors" step modifies 'available'.
        # So we just iterate through the fixed order, and check if the bit is still set in 'available'.
        for idx in order:
            if time.time() > deadline:
                break
            if (available >> idx) & 1:
                # Select idx
                selected.append(idx)
                # Remove idx and neighbors
                # neighbors = neighbor_bits[idx]
                # available &= ~(1 << idx | neighbor_bits[idx])
                mask = (1 << idx) | neighbor_bits[idx]
                available &= ~mask
        return selected

    # Run Strategy 1
    sel1 = greedy_mis(order1)
    if len(sel1) > best_size:
        best_size = len(sel1)
        best_set = set(sel1)
        
    # Strategy 2: Random restarts
    # We have about 115s budget.
    # Let's see how long one run takes.
    t_start = time.time()
    sel2 = greedy_mis(order1) # Just to measure
    t_elapsed = time.time() - t_start
    
    # If it's fast enough, do random restarts.
    # The greedy loop is O(n * cost_of_bit_ops). n=5280. Bit ops are fast.
    # Let's try 50 random restarts if time permits.
    
    np_random = np.random.RandomState(42)
    
    for restart in range(100):
        if time.time() > deadline - 5:
            break
            
        # Random order
        order2 = np_random.permutation(n_w4)
        sel2 = greedy_mis(order2)
        if len(sel2) > best_size:
            best_size = len(sel2)
            best_set = set(sel2)

    # --- Step 4: Add Compatible Weight-2 Vectors ---
    # Generate all weight-2 vectors
    w2_vectors = []
    for support in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for i, s in zip(support, signs):
                v[i] = s
            w2_vectors.append(v)
            
    compatible_w2 = []
    # For each w2 vector, check if it is compatible with ALL selected w4 vectors.
    # Condition: <u, v> <= 1 for all v in best_set.
    
    # To speed up, we can precompute for each w4 vector in best_set, which w2 vectors it conflicts with?
    # Or just check directly.
    # Number of w2 vectors is C(11,2)*4 = 55*4 = 220.
    # Number of w4 vectors in best_set could be ~1000-2000.
    # 220 * 2000 = 440,000 checks. Each check is small.
    
    # Let's represent w4 vectors as lists for easier dot product.
    selected_w4_list = []
    for idx in best_set:
        support, signs = w4_vectors[idx]
        v = [0] * D
        for i, s in zip(support, signs):
            v[i] = s
        selected_w4_list.append(v)
        
    for u in w2_vectors:
        ok = True
        for v in selected_w4_list:
            # Calculate dot product
            # u has 2 non-zeros, v has 4.
            # We can just sum u[i]*v[i] for all i.
            # Or optimize:
            # Find indices where u is non-zero.
            # u_supp = [i for i in range(D) if u[i] != 0]
            # dot = sum(u[i]*v[i] for i in u_supp)
            
            # Simpler:
            dot = 0
            for i in range(D):
                if u[i] != 0:
                    dot += u[i] * v[i]
            if dot > 1:
                ok = False
                break
        if ok:
            compatible_w2.append(u)
            
    # --- Step 5: Construct Result ---
    result_vectors = []
    for v in selected_w4_list:
        result_vectors.append(v)
    for v in compatible_w2:
        result_vectors.append(v)
        
    if not result_vectors:
        # Fallback: just weight-2
        for v in w2_vectors:
            result_vectors.append(v)

    result = np.array(result_vectors, dtype=np.int64)
    return result
