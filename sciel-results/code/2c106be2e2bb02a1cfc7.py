"""Program for kissing number in dimension 11 using greedy integer vector search."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools

def construct():
    # We will search for integer vectors with norm squared 4 (norm 2).
    # This allows for a much denser packing than norm squared 2 (norm sqrt(2)).
    # Constraint: min pairwise dist >= max norm.
    # If all norms are 2, dist^2 >= 4 => 4+4-2dot >= 4 => dot <= 2.
    
    vectors = []
    
    # Type A: One entry +/-2, rest 0.
    # There are 2*D = 22 such vectors.
    # They are mutually compatible (dot 0 or +/-4? No, (2,0) and (-2,0) dot -4. 4+4-2(-4)=16>=4. OK.
    # (2,0) and (0,2) dot 0. OK.
    # So we can include all 22.
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            vectors.append(v)
            
    vecs_arr = np.array(vectors, dtype=np.int64)
    
    # Type B: Four entries +/-1, rest 0.
    # Count: C(11,4) * 2^4 = 330 * 16 = 5280.
    # We will greedily add these if they satisfy dot <= 2 with all existing vectors.
    
    # Pre-generate all Type B vectors
    type_b_vectors = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(support, signs):
                v[idx] = s
            type_b_vectors.append(v)
            
    tb_arr = np.array(type_b_vectors, dtype=np.int64)
    
    # Greedy addition
    # To speed up, we can check against the current set.
    # The set will grow.
    # Initial size 22.
    
    # Optimization: 
    # If we add vectors, the constraint is dot <= 2.
    # We can process in batches or one by one.
    # One by one is safer for correctness but might be slow if K is large.
    # Let's try one by one with NumPy dot product.
    
    current_vecs = vecs_arr
    for v in tb_arr:
        # Calculate dot products of v with all current_vecs
        # v is shape (D,)
        # current_vecs is shape (N, D)
        dots = current_vecs @ v
        if np.all(dots <= 2):
            current_vecs = np.vstack([current_vecs, v.reshape(1, -1)])
            
    return current_vecs
# EVOLVE-BLOCK-END
