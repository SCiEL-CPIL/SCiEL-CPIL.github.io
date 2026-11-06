import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    # 1. Precompute Q3 masks
    # Q3 is K4,4 minus a perfect matching.
    # Vertices 0-3 in A, 4-7 in B.
    # Edges are (i, 4+j) for i,j in 0..3.
    # Mask bit (i*4 + j) represents edge (i, 4+j).
    
    q3_masks = []
    for perm in range(24): # Permutations of 0,1,2,3
        # Generate permutation
        p = list(range(4))
        # Simple perm generation
        # Actually itertools.permutations is fine
        pass
        
    from itertools import permutations
    perms = list(permutations(range(4)))
    for p in perms:
        mask = 0
        for i in range(4):
            for j in range(4):
                if i != p[j]: # Edge exists if not in matching
                    mask |= (1 << (i*4 + j))
        q3_masks.append(mask)
        
    # 2. Precompute has_q3 lookup table
    # Size 2^16 = 65536
    has_q3 = [False] * 65536
    for mask in range(65536):
        # Check if mask contains any q3_mask
        # Optimization: if popcount < 12, skip
        if mask.bit_count() < 12:
            continue
        for qm in q3_masks:
            if (mask & qm) == qm:
                has_q3[mask] = True
                break
                
    # 3. Function to check if a bipartite graph (64-bit mask) is Q3-free
    # Bipartite mask: bit (i*8 + j) represents edge (i, j) for i in 0..7 (A), j in 0..7 (B)
    # We need to check all subsets A' of size 4 from A, and B' of size 4 from B.
    # There are C(8,4)=70 subsets for A and 70 for B. 70*70=4900 pairs.
    # For each pair, extract 16-bit mask and check has_q3.
    
    # Precompute index mappings for subsets
    # Map subset of 4 indices to a list of 4 indices
    subsets_4 = []
    from itertools import combinations
    for combo in combinations(range(8), 4):
        subsets_4.append(combo)
        
    # Precompute the bit positions for each subset
    # For a subset (i1, i2, i3, i4), we need to extract bits from the 64-bit mask.
    # This is done by shifting and masking.
    
    def is_valid_bipartite(b_mask):
        for a_sub in subsets_4:
            # Extract 4 bits for A? No, we need to form the 16-bit mask for the subgraph.
            # The 16-bit mask has bits (k*4 + l) where k is index in a_sub, l is index in b_sub.
            # This requires gathering bits.
            # To speed this up, we can precompute for each a_sub a list of shifts/masks?
            # Or just iterate.
            for b_sub in subsets_4:
                sub_mask = 0
                for k, ai in enumerate(a_sub):
                    # Row ai in the 8x8 matrix
                    # We need columns b_sub[0], b_sub[1], b_sub[2], b_sub[3]
                    for l, bj in enumerate(b_sub):
                        if b_mask & (1 << (ai*8 + bj)):
                            sub_mask |= (1 << (k*4 + l))
                if has_q3[sub_mask]:
                    return False
        return True
        
    # 4. Initial Graph
    # Start with K8,8 (all 1s)
    current_mask = (1 << 64) - 1
    # It is invalid. Remove edges greedily to make it valid.
    while not is_valid_bipartite(current_mask):
        # Find a violation
        # We can modify is_valid to return the violation
        # For speed, just pick a random edge and remove it?
        # Better: find the first violation and remove an edge from it.
        
        # To avoid complex violation finding, let's just do a random removal loop
        # This might be slow if K8,8 is very dense in Q3s.
        # K8,8 has many Q3s.
        
        # Let's try to remove edges that are in many Q3s.
        # For now, random removal until valid.
        # 64 edges. Removing ~10-20 should suffice.
        
        # Pick a random set bit and clear it
        bits = [i for i in range(64) if current_mask & (1 << i)]
        if not bits: break
        idx = random.choice(bits)
        current_mask &= ~(1 << idx)
        
    best_mask = current_mask
    best_edges = current_mask.bit_count()
    
    # 5. Simulated Annealing
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random edge to flip
        idx = random.randint(0, 63)
        new_mask = current_mask ^ (1 << idx)
        
        if is_valid_bipartite(new_mask):
            new_edges = new_mask.bit_count()
            delta = new_edges - current_mask.bit_count()
            
            if delta > 0:
                # Better
                current_mask = new_mask
                if new_edges > best_edges:
                    best_mask = new_mask
                    best_edges = new_edges
            else:
                # Worse (removed an edge)
                # Accept with probability exp(delta / T)
                if random.random() < math.exp(delta / T):
                    current_mask = new_mask
        else:
            # Invalid
            # Reject, or if we want to explore, maybe accept with very low prob?
            # Usually better to stay valid.
            pass
            
        T *= cooling
        
    # 6. Construct final graph
    edges = []
    # Add bipartite edges
    for i in range(8):
        for j in range(8):
            if best_mask & (1 << (i*8 + j)):
                # Vertex mapping: A: 0..7, B: 8..15
                edges.append([i, 8+j])
                
    # Add internal edges
    # A: 0..7
    for i in range(8):
        for j in range(i+1, 8):
            edges.append([i, j])
    # B: 8..15
    for i in range(8, 16):
        for j in range(i+1, 16):
            edges.append([i, j])
            
    return edges
