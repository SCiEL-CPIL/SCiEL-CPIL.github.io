import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill
    
    n = 16
    
    # Precompute all 8-vertex subsets
    subsets_8 = []
    subset_masks = []
    for s in itertools.combinations(range(n), 8):
        mask = 0
        for v in s:
            mask |= (1 << v)
        subsets_8.append(s)
        subset_masks.append(mask)
    
    # Precompute all partitions of 8 positions into two sets of 4
    # Each partition is (x_indices, y_indices) where indices are 0..7
    partitions = []
    for x_combo in itertools.combinations(range(8), 4):
        x_indices = tuple(x_combo)
        y_indices = tuple(i for i in range(8) if i not in x_indices)
        partitions.append((x_indices, y_indices))
    
    num_subsets = len(subsets_8)
    num_partitions = len(partitions)
    
    # Precompute masks and vertex lists for each (subset, partition) pair
    # This is heavy memory but fast access.
    # Structure: precomputed[k][j] = (x_mask, y_mask, x_verts, y_verts)
    precomputed = [[None for _ in range(num_partitions)] for _ in range(num_subsets)]
    
    for k, s in enumerate(subsets_8):
        for j, (x_indices, y_indices) in enumerate(partitions):
            x_mask = 0
            y_mask = 0
            x_verts = []
            y_verts = []
            for i in x_indices:
                v = s[i]
                x_mask |= (1 << v)
                x_verts.append(v)
            for i in y_indices:
                v = s[i]
                y_mask |= (1 << v)
                y_verts.append(v)
            precomputed[k][j] = (x_mask, y_mask, tuple(x_verts), tuple(y_verts))
    
    def has_q3(adj_masks):
        for k in range(num_subsets):
            mask = subset_masks[k]
            s = subsets_8[k]
            
            # Quick edge count check
            edge_count = 0
            for v in s:
                edge_count += (adj_masks[v] & mask).bit_count()
            if edge_count < 24: # 12 edges * 2
                continue
            
            for j in range(num_partitions):
                x_mask, y_mask, x_verts, y_verts = precomputed[k][j]
                
                valid = True
                for v in x_verts:
                    if (adj_masks[v] & y_mask).bit_count() < 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                for v in y_verts:
                    if (adj_masks[v] & x_mask).bit_count() < 3:
                        valid = False
                        break
                
                if valid:
                    return True
        return False
    
    # Build initial graph: K_3,13 with A={0,1,2}, B={3..15}
    adj_masks = [0] * n
    current_edges = set()
    
    for a in range(3):
        for b in range(3, 16):
            adj_masks[a] |= (1 << b)
            adj_masks[b] |= (1 << a)
            current_edges.add((a, b))
    
    # Add A-A edges to make K3 on A
    a_edges = [(0,1), (1,2), (0,2)]
    for u, v in a_edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        current_edges.add((u, v))
    
    b_vertices = list(range(3, 16))
    b_edges = [(b_vertices[i], b_vertices[j]) for i in range(len(b_vertices)) for j in range(i+1, len(b_vertices))]
    
    best_edges = set(current_edges)
    best_count = len(current_edges)
    
    # Local search with greedy addition and edge swaps
    # Increase restarts since has_q3 is faster
    for restart in range(500):
        if time.time() > deadline - 10:
            break
        
        # Reset to base state
        adj_masks = [0] * n
        current_edges = set()
        
        for a in range(3):
            for b in range(3, 16):
                adj_masks[a] |= (1 << b)
                adj_masks[b] |= (1 << a)
                current_edges.add((a, b))
        
        for u, v in a_edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            current_edges.add((u, v))
        
        # Greedy phase: try adding B-B edges in random order
        random.shuffle(b_edges)
        for u, v in b_edges:
            if time.time() > deadline - 5:
                break
            if (u, v) in current_edges:
                continue
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            current_edges.add((u, v))
            
            if has_q3(adj_masks):
                adj_masks[u] &= ~(1 << v)
                adj_masks[v] &= ~(1 << u)
                current_edges.remove((u, v))
        
        # Local search phase: try edge swaps
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            random.shuffle(b_edges)
            for u, v in b_edges:
                if time.time() > deadline - 5:
                    break
                if (u, v) in current_edges:
                    continue
                
                # Try adding (u,v)
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                
                if not has_q3(adj_masks):
                    current_edges.add((u, v))
                    improved = True
                    break
                else:
                    # Revert
                    adj_masks[u] &= ~(1 << v)
                    adj_masks[v] &= ~(1 << u)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
    
    result = [[u, v] for u, v in best_edges]
    return result

if __name__ == "__main__":
    start = time.time()
    edges = construct()
    print(f"Found {len(edges)} edges in {time.time() - start:.2f}s")
