import time
import random
import numpy as np

def construct():
    n = 16
    t_start = time.time()
    deadline = t_start + 115.0

    # Precompute Q3 edges
    _Q3_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    _Q3_ADJ = [0] * 8
    for a, b in _Q3_EDGES:
        _Q3_ADJ[a] |= (1 << b)
        _Q3_ADJ[b] |= (1 << a)

    def has_q3_fast(adj):
        """
        Check if graph with adjacency masks 'adj' contains Q3.
        Uses bitmasks for fast intersection.
        """
        # Try all mappings of Q3 vertices to graph vertices
        # To speed up, we can fix the first few vertices or just iterate.
        # Since n=16, 16^8 is too big. But we can prune.
        # Better: iterate over all 4-cycles in G? No, G is dense.
        # Let's use a smarter backtracking with bitmasks.
        
        # Order Q3 vertices for backtracking
        # 0 is connected to 1, 2, 4
        # Let's fix Q3 vertex 0 to some graph vertex u.
        # Then Q3 vertices 1, 2, 4 must be neighbors of u.
        
        for u in range(n):
            # Neighbors of u in G
            neu = adj[u]
            # Q3 neighbors of 0 are 1, 2, 4
            # We need to assign 1, 2, 4 to distinct neighbors of u
            # Let's iterate over permutations of 3 neighbors
            # This might be too slow if degree is high (15). 15*14*13 = 2730 per u. 16*2730 = 43680.
            # Then for each, we need to assign the rest.
            
            # Alternative: Just check if any 4-cycle in G is part of a Q3?
            # Q3 has 6 4-cycles.
            
            # Let's try a different fast check:
            # Count number of Q3 embeddings? No, just existence.
            
            # Let's stick to a simple backtracking but optimized with bitmasks.
            # We'll try to map Q3 vertices 0..7 to G vertices.
            # assignment[q] = g
            assignment = [-1] * 8
            assignment[0] = u
            
            def bt(q):
                if q == 8:
                    # Check all edges
                    for a, b in _Q3_EDGES:
                        if not (adj[assignment[a]] & (1 << assignment[b])):
                            return False
                    return True
                
                # Find neighbors in Q3 that are already assigned
                # And constraints from unassigned
                candidates = 0
                # Candidates must be neighbors of all assigned Q3 neighbors
                for y in range(8):
                    if assignment[y] != -1 and (_Q3_ADJ[q] & (1 << y)):
                        candidates &= adj[assignment[y]]
                
                # Candidates must not be already used
                used_mask = 0
                for g in assignment:
                    if g != -1:
                        used_mask |= (1 << g)
                candidates &= ~used_mask
                
                # Try each candidate
                while candidates:
                    lsb = candidates & (-candidates)
                    g = lsb.bit_length() - 1
                    candidates ^= lsb
                    
                    assignment[q] = g
                    if bt(q + 1):
                        return True
                    assignment[q] = -1
                return False

            if bt(1):
                return True
        
        return False

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def edge_list(adj):
        es = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    es.append([u, v])
        return es

    # Spectral penalty: Trace(A^4) = 2 * (number of 4-cycles) + 4 * (number of triangles * something?)
    # Actually Tr(A^4) = 2 * (sum d_i^2 - sum d_i) + 8 * (# 4-cycles)
    # So minimizing Tr(A^4) roughly minimizes 4-cycles.
    
    def get_penalty(adj):
        A = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & (-mask)
                j = lsb.bit_length() - 1
                A[i, j] = 1
                mask ^= lsb
        A4 = np.linalg.matrix_power(A, 4)
        return np.trace(A4)

    # Start with K16
    adj = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                adj[i] |= (1 << j)
    
    current_count = n * (n - 1) // 2
    best_count = -1
    best_edges = None

    # Since K16 definitely has Q3, we must remove edges until it's Q3-free.
    # Strategy: While has_q3_fast(adj): remove edge that reduces penalty most.
    # To avoid O(1) penalty check for every edge, we can just remove random edges or edges in a found Q3.
    
    # Better: Find a Q3, remove one of its edges.
    # But we need to maximize edges.
    # Let's use a heuristic: 
    # 1. Find a Q3.
    # 2. For each edge in that Q3, estimate cost of removing it (e.g., degree sum).
    # 3. Remove the "best" edge.
    
    # Actually, let's just do a local search.
    # Start with K16.
    # While Q3 exists:
    #   Find a Q3 embedding.
    #   Remove one edge from it.
    # This guarantees Q3-free but might not be optimal.
    
    # To improve, we can try to add back edges later.
    
    # Let's implement find_q3_witness to get the specific edges.
    
    def find_q3_witness(adj):
        for u in range(n):
            assignment = [-1] * 8
            assignment[0] = u
            
            def bt(q):
                if q == 8:
                    return list(assignment)
                
                candidates = 0
                for y in range(8):
                    if assignment[y] != -1 and (_Q3_ADJ[q] & (1 << y)):
                        candidates &= adj[assignment[y]]
                
                used_mask = 0
                for g in assignment:
                    if g != -1:
                        used_mask |= (1 << g)
                candidates &= ~used_mask
                
                while candidates:
                    lsb = candidates & (-candidates)
                    g = lsb.bit_length() - 1
                    candidates ^= lsb
                    
                    assignment[q] = g
                    res = bt(q + 1)
                    if res:
                        return res
                    assignment[q] = -1
                return None

            res = bt(1)
            if res:
                return res
        return None

    # Main loop
    while time.time() < deadline - 10:
        if not has_q3_fast(adj):
            break
        
        witness = find_q3_witness(adj)
        if not witness:
            break
            
        # Get edges of this Q3 in the graph
        q3_edges_in_g = set()
        for a, b in _Q3_EDGES:
            u, v = witness[a], witness[b]
            if u > v: u, v = v, u
            q3_edges_in_g.add((u, v))
        
        # Choose an edge to remove. 
        # Heuristic: remove edge with highest "centrality" or just random from the Q3?
        # Removing an edge from the Q3 breaks it.
        # Let's pick the edge that appears in the most 4-cycles? Too slow.
        # Let's just pick random edge from the Q3 to introduce diversity.
        # Or pick the edge with min degree sum to preserve structure?
        
        # Let's try removing the edge with the minimum sum of degrees of endpoints.
        min_deg_sum = float('inf')
        best_edge_to_remove = None
        for u, v in q3_edges_in_g:
            deg_sum = bin(adj[u]).count('1') + bin(adj[v]).count('1')
            if deg_sum < min_deg_sum:
                min_deg_sum = deg_sum
                best_edge_to_remove = (u, v)
        
        u, v = best_edge_to_remove
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        current_count -= 1
        
        if current_count > best_count:
            best_count = current_count
            best_edges = edge_list(adj)

    # Now we have a Q3-free graph. Let's try to add edges back.
    if best_edges is None:
        best_edges = edge_list(adj)
        best_count = count_edges(adj)

    # Load best edges
    adj = [0] * n
    for u, v in best_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    current_count = count_edges(adj)

    # Greedy add back
    missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
    random.shuffle(missing)
    
    for u, v in missing:
        if time.time() > deadline - 5:
            break
        if not (adj[u] & (1 << v)):
            # Check if adding creates Q3
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if has_q3_fast(adj):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                current_count += 1
    
    if current_count > best_count:
        best_count = current_count
        best_edges = edge_list(adj)

    return best_edges

# Quick test to ensure it doesn't crash
# print(construct())
