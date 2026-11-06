import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    
    # Precompute Q3 masks for K_4,4
    # Vertices 0-3 in A, 4-7 in B
    # Edge (i, j) where i in [0,3], j in [4,7] maps to bit i*4 + (j-4)
    # Total 16 bits.
    
    def is_q3_mask(mask):
        # Check if the bipartite graph defined by mask is Q3
        # Q3 is 3-regular
        deg_a = [0]*4
        deg_b = [0]*4
        for i in range(4):
            for j in range(4):
                if mask & (1 << (i*4 + j)):
                    deg_a[i] += 1
                    deg_b[j] += 1
        if all(d == 3 for d in deg_a) and all(d == 3 for d in deg_b):
            return True
        return False

    q3_masks = []
    # Iterate over all 12-edge subsets of 16 edges
    # This is C(16, 12) = 1820, which is fast
    all_edges = list(range(16))
    for combo in itertools.combinations(all_edges, 12):
        mask = 0
        for idx in combo:
            mask |= (1 << idx)
        if is_q3_mask(mask):
            q3_masks.append(mask)
            
    # adj[i] is a bitmask of neighbors
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
    def get_edge_count():
        return sum(bin(adj[i]).count("1") for i in range(n)) // 2

    # Precompute all 8-subsets and their splits
    subsets_8 = list(itertools.combinations(range(n), 8))
    
    # For each 8-subset, precompute the list of (A, B) splits with bit positions
    precomputed_checks = []
    for sub in subsets_8:
        for A in itertools.combinations(sub, 4):
            B = tuple(v for v in sub if v not in A)
            # Create mapping: for each (a, b) in A x B, store (a, b, mask_bit)
            pairs = []
            for i in range(4):
                a = A[i]
                for j in range(4):
                    b = B[j]
                    pairs.append((a, b, i*4 + j))
            precomputed_checks.append((A, B, pairs))
    
    # Build index: for each edge (u,v), which checks contain it?
    # This allows us to check only relevant subsets when adding an edge
    edge_to_checks = {}
    for idx, (A, B, pairs) in enumerate(precomputed_checks):
        for a, b, mb in pairs:
            if a < b:
                edge = (a, b)
            else:
                edge = (b, a)
            if edge not in edge_to_checks:
                edge_to_checks[edge] = []
            edge_to_checks[edge].append(idx)
    
    def find_q3_edge_full(adj):
        """Find any Q3 in the graph using full scan."""
        indices = list(range(len(precomputed_checks)))
        random.shuffle(indices)
        
        for idx in indices:
            A, B, pairs = precomputed_checks[idx]
            mask = 0
            for a, b, mb in pairs:
                if adj[a] & (1 << b):
                    mask |= (1 << mb)
            
            for qm in q3_masks:
                if (mask & qm) == qm:
                    # Found Q3, return an edge from it
                    for a, b, mb in pairs:
                        if qm & (1 << mb):
                            return (a, b)
                    return (pairs[0][0], pairs[0][1])
                    
        return None

    def find_q3_edge_containing(adj, u, v):
        """Find a Q3 containing edge (u,v), or None."""
        edge = (min(u,v), max(u,v))
        if edge not in edge_to_checks:
            return None
        
        indices = edge_to_checks[edge]
        random.shuffle(indices)
        
        for idx in indices:
            A, B, pairs = precomputed_checks[idx]
            mask = 0
            for a, b, mb in pairs:
                if adj[a] & (1 << b):
                    mask |= (1 << mb)
            
            for qm in q3_masks:
                if (mask & qm) == qm:
                    # Found Q3, return an edge from it
                    for a, b, mb in pairs:
                        if qm & (1 << mb):
                            return (a, b)
                    return (pairs[0][0], pairs[0][1])
                    
        return None

    # Phase 1: Remove edges to make Q3-free
    max_iter = 500
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        edge = find_q3_edge_full(adj)
        if edge is None:
            break
            
        u, v = edge
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    # Phase 2: Addition with targeted Q3 checking
    missing_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if not (adj[u] & (1 << v)):
                missing_edges.append((u, v))
                
    random.shuffle(missing_edges)
    
    for u, v in missing_edges:
        if time.time() > deadline:
            break
            
        # Tentatively add
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Check if Q3 exists containing the new edge
        edge_in_q3 = find_q3_edge_containing(adj, u, v)
        
        if edge_in_q3 is not None:
            # Q3 found containing new edge, remove it
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            # Keep the edge
            pass

    # Convert adj to edge list
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges

# Run once to ensure no syntax errors
# construct()
