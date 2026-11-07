import random
import math

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']
    
    # Precompute adjacency masks
    adj_masks = [0] * n
    for u, v in edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    # List of all vertices
    all_vertices = list(range(n))
    
    def evaluate(mask):
        """
        Returns the number of distinct degrees in the induced subgraph defined by mask.
        Returns -1 if the subset size is less than min_vertices.
        """
        if mask.bit_count() < min_vertices:
            return -1
            
        degrees = set()
        m = mask
        while m:
            # Get index of the lowest set bit
            lsb = m & -m
            v = lsb.bit_length() - 1
            # Calculate degree of v in the subgraph
            deg = (adj_masks[v] & mask).bit_count()
            degrees.add(deg)
            # Clear the lowest set bit
            m ^= lsb
            
        return len(degrees)

    best_score = -1
    best_mask = 0
    
    # Run multiple restarts of Simulated Annealing
    num_restarts = 100
    max_iters = 10000
    start_temp = 5.0
    end_temp = 0.01
    
    for r in range(num_restarts):
        # Initialize with a random subset of size between min_vertices and n
        size = random.randint(min_vertices, n)
        initial_subset = random.sample(all_vertices, size)
        current_mask = 0
        for v in initial_subset:
            current_mask |= (1 << v)
            
        current_score = evaluate(current_mask)
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask
            
        temp = start_temp
        cooling_factor = (end_temp / start_temp) ** (1.0 / max_iters)
        
        for i in range(max_iters):
            # Generate a neighbor
            move_type = random.random()
            
            if move_type < 0.5:
                # Swap: keep size constant
                # Find in and out lists efficiently
                in_list = []
                out_list = []
                for v in range(n):
                    if (current_mask >> v) & 1:
                        in_list.append(v)
                    else:
                        out_list.append(v)
                        
                if not out_list or not in_list:
                    continue
                
                v_in = random.choice(in_list)
                v_out = random.choice(out_list)
                
                new_mask = current_mask ^ (1 << v_in) ^ (1 << v_out)
                
            elif move_type < 0.75:
                # Add: increase size by 1
                out_list = []
                for v in range(n):
                    if not ((current_mask >> v) & 1):
                        out_list.append(v)
                if not out_list:
                    continue
                v_add = random.choice(out_list)
                new_mask = current_mask | (1 << v_add)
                
            else:
                # Remove: decrease size by 1
                in_list = []
                for v in range(n):
                    if (current_mask >> v) & 1:
                        in_list.append(v)
                if len(in_list) <= min_vertices:
                    continue # Can't remove if we'd go below min_vertices
                v_rem = random.choice(in_list)
                new_mask = current_mask & ~(1 << v_rem)
                
            new_score = evaluate(new_mask)
            
            # Metropolis acceptance criterion
            delta = new_score - current_score
            if delta >= 0:
                current_mask = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
            else:
                # Accept with probability exp(-delta / temp)
                # delta is negative, so -delta is positive
                if random.random() < math.exp(-delta / temp):
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
            
            temp *= cooling_factor
            
    # Convert best_mask to list of vertices
    best_subset = []
    for v in range(n):
        if (best_mask >> v) & 1:
            best_subset.append(v)
            
    return best_subset
