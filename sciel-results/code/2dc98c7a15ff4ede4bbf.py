import random
import math

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']
    
    # Precompute adjacency bitmasks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    all_mask = (1 << n) - 1
    
    def evaluate(mask):
        if mask.bit_count() < min_vertices:
            return -1
        degrees = set()
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            # Count neighbors in mask
            deg = (mask & adj_mask[idx]).bit_count()
            degrees.add(deg)
            m ^= lsb
        return len(degrees)
    
    best_score = -1
    best_mask = 0
    
    # Simulated Annealing
    # We will run multiple restarts
    # Each restart tries to find a good local optimum
    
    restarts = 50
    max_iters = 20000
    
    for r in range(restarts):
        # Initialize with random subset of size between min_vertices and n
        size = random.randint(min_vertices, n)
        # Randomly select 'size' vertices
        # Method: shuffle all vertices, take first 'size'
        verts = list(range(n))
        random.shuffle(verts)
        current_mask = 0
        for v in verts[:size]:
            current_mask |= (1 << v)
            
        current_score = evaluate(current_mask)
        
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask
            
        T = 10.0
        cooling = 0.999
        min_T = 0.001
        
        for i in range(max_iters):
            # Generate neighbor
            # Options: Add, Remove, Swap
            # Ensure we stay valid (size >= min_vertices)
            
            # Determine possible moves
            # Vertices in mask
            in_verts = []
            out_verts = []
            m = current_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                in_verts.append(idx)
                m ^= lsb
            
            # Out verts are those not in mask
            # We can just pick a random vertex from 0..n-1 and check if in mask
            # Or maintain lists. With N=17, lists are fast.
            
            move_type = random.random()
            
            if move_type < 0.3:
                # Add
                # Pick random vertex not in mask
                # If all in mask, skip
                # Efficient way: pick random v, if in mask, retry or swap?
                # Let's just pick from out_verts if we had them.
                # For simplicity, pick random v in [0, n)
                v = random.randint(0, n-1)
                if (current_mask >> v) & 1:
                    continue # v is already in mask, try again (or swap)
                # Actually, if we want to be efficient, we should precompute out_verts
                # But N is small.
                
                new_mask = current_mask | (1 << v)
                new_score = evaluate(new_mask)
                delta = new_score - current_score
                
                if delta >= 0 or random.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
            elif move_type < 0.6:
                # Remove
                if len(in_verts) <= min_vertices:
                    continue
                v = random.choice(in_verts)
                new_mask = current_mask & ~(1 << v)
                new_score = evaluate(new_mask)
                delta = new_score - current_score
                
                if delta >= 0 or random.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
            else:
                # Swap
                if not in_verts:
                    continue
                v_in = random.choice(in_verts)
                v_out = random.randint(0, n-1)
                if (current_mask >> v_out) & 1:
                    continue # v_out is in mask, not a valid swap out
                new_mask = (current_mask & ~(1 << v_in)) | (1 << v_out)
                new_score = evaluate(new_mask)
                delta = new_score - current_score
                
                if delta >= 0 or random.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
            
            T *= cooling
            if T < min_T:
                break
                
    # Convert best_mask to list of vertices
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx)
        m ^= lsb
        
    return result
