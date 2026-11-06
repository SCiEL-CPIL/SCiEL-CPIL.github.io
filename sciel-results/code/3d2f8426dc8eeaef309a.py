import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    def seed_solution():
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    best_result = seed_solution()
    
    # Bitmask-based C4-free construction and verification
    
    def has_independent_set_11(n, adj_masks, node_cap=500000):
        """Check if graph has an independent set of size 11 using branch and bound."""
        nodes_visited = [0]
        
        def dfs(candidates, current_size):
            if nodes_visited[0] > node_cap:
                return False
            if current_size >= 11:
                return True
            if candidates == 0:
                return False
            
            # Upper bound: if we can't reach 11, prune
            if current_size + candidates.bit_count() < 11:
                return False
            
            # Pick vertex with highest degree in induced subgraph for better pruning
            # Try each vertex in candidates
            # Use a heuristic: pick vertex that appears in most edges of induced subgraph
            # For simplicity, iterate through vertices
            
            # Get list of vertices in candidates
            verts = []
            c = candidates
            while c:
                v = (c & -c).bit_length() - 1
                verts.append(v)
                c &= c - 1
            
            # Sort by degree in induced subgraph (descending) for better pruning
            def induced_degree(v):
                # Count neighbors of v that are in candidates
                nbrs = adj_masks[v] & candidates
                return nbrs.bit_count()
            
            verts.sort(key=induced_degree, reverse=True)
            
            for v in verts:
                if nodes_visited[0] > node_cap:
                    return False
                nodes_visited[0] += 1
                
                # Include v: remove v and its neighbors from candidates
                new_candidates = candidates & ~(adj_masks[v] | (1 << v))
                if dfs(new_candidates, current_size + 1):
                    return True
                
                # Exclude v
                new_candidates = candidates & ~(1 << v)
                if dfs(new_candidates, current_size):
                    return True
            
            return False
        
        all_mask = (1 << n) - 1
        return dfs(all_mask, 0)
    
    def build_c4_free_graph(n, seed):
        """Build a C4-free graph incrementally using bitmasks."""
        rng = random.Random(seed)
        adj_masks = [0] * n
        
        # Process vertices in random order
        order = list(range(n))
        rng.shuffle(order)
        
        for idx, v in enumerate(order):
            # Determine which previous vertices v can connect to
            # A vertex u can connect to v if for all w in N(v) already chosen,
            # N(u) and N(w) are disjoint (no common neighbor)
            
            # Collect candidates
            candidates = []
            for i in range(idx):
                u = order[i]
                candidates.append(u)
            
            rng.shuffle(candidates)
            
            # Greedily add edges
            for u in candidates:
                # Check if adding edge (v, u) creates a C4
                # C4 exists if there's a vertex w in N(v) such that N(u) ∩ N(w) ≠ ∅
                # But we need to be careful: we're building incrementally
                # N(v) so far is adj_masks[v]
                # N(u) is adj_masks[u]
                # For each w in N(v), check if adj_masks[u] & adj_masks[w] != 0
                
                conflict = False
                vw = adj_masks[v]
                while vw:
                    w_bit = vw & -vw
                    w = w_bit.bit_length() - 1
                    if adj_masks[u] & adj_masks[w]:
                        conflict = True
                        break
                    vw &= vw - 1
                
                if not conflict:
                    adj_masks[v] |= (1 << u)
                    adj_masks[u] |= (1 << v)
        
        return adj_masks
    
    def adj_masks_to_edges(n, adj_masks):
        """Convert adjacency masks to edge list."""
        edges = []
        for i in range(n):
            mask = adj_masks[i]
            while mask:
                bit = mask & -mask
                j = bit.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask &= mask - 1
        return edges
    
    def min_degree(n, adj_masks):
        return min(adj_masks[i].bit_count() for i in range(n))
    
    def avg_degree(n, adj_masks):
        total = sum(adj_masks[i].bit_count() for i in range(n))
        return total / n
    
    # Try to find a valid graph with as many vertices as possible
    for n in range(43, 30, -1):
        if time.time() > deadline:
            break
        
        # Try multiple seeds
        found = False
        num_trials = 30 if n <= 38 else 15
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            seed = trial * 7919 + n * 31
            adj_masks = build_c4_free_graph(n, seed)
            
            # Quick check: minimum degree should be reasonable
            md = min_degree(n, adj_masks)
            if md < 5:
                continue
            
            # Check for independent set of size 11
            # If no independent set of size 11 exists, graph is valid
            has_is = has_independent_set_11(n, adj_masks, node_cap=200000)
            if not has_is:
                edges = adj_masks_to_edges(n, adj_masks)
                best_result = {'n': n, 'edges': edges}
                found = True
                break
        
        if found:
            break
    
    return best_result

