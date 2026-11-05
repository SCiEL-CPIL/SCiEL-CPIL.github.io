import random
import time
import numpy as np

def construct():
    n = 20
    start_time = time.time()
    max_time = 115.0  # Safety margin
    
    best_edges = []
    best_count = 0
    
    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_valid_add(u, v, adj_mask, common):
        """Check if edge (u,v) can be added without violating K2,3-free constraint."""
        if (adj_mask[u] >> v) & 1:
            return False
        if common[u][v] > 2:
            return False
        
        # Adding (u,v) increases common[u][w] by 1 for all w in N(v), w != u
        # Need common[u][w] < 2 for all w in N(v), w != u
        mask_v = adj_mask[v] & ~(1 << u)
        if mask_v:
            # Check all w in N(v)\{u} have common[u][w] < 2
            w = 0
            m = mask_v
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                if common[u][w] >= 2:
                    return False
                m -= lsb
        
        # Similarly for common[v][w] < 2 for all w in N(u), w != v
        mask_u = adj_mask[u] & ~(1 << v)
        if mask_u:
            w = 0
            m = mask_u
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                if common[v][w] >= 2:
                    return False
                m -= lsb
        
        return True

    def add_edge(u, v, adj_mask, common, edges):
        """Add edge (u,v) to the graph state."""
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edges.append((u, v))
        
        # Update common neighbor counts
        # For each w in N(v)\{u}, common[u][w] += 1
        mask_v = adj_mask[v] & ~(1 << u)
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[u][w] += 1
            common[w][u] += 1
            m -= lsb
        
        # For each w in N(u)\{v}, common[v][w] += 1
        mask_u = adj_mask[u] & ~(1 << v)
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[v][w] += 1
            common[w][v] += 1
            m -= lsb

    def remove_edge(u, v, adj_mask, common, edges):
        """Remove edge (u,v) from the graph state."""
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        edges.remove((u, v))
        
        # For each w in N(v)\{u}, common[u][w] -= 1
        mask_v = adj_mask[v]
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[u][w] -= 1
            common[w][u] -= 1
            m -= lsb
        
        # For each w in N(u)\{v}, common[v][w] -= 1
        mask_u = adj_mask[u]
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[v][w] -= 1
            common[w][v] -= 1
            m -= lsb

    def greedy_fill(adj_mask, common, edges):
        """Greedy fill: add edges in random order if valid."""
        pairs = all_pairs[:]
        random.shuffle(pairs)
        for u, v in pairs:
            if is_valid_add(u, v, adj_mask, common):
                add_edge(u, v, adj_mask, common, edges)

    def local_search(initial_edges, max_iters=300):
        """Local search with kicks to escape local optima."""
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        edges = []
        
        for u, v in initial_edges:
            add_edge(u, v, adj_mask, common, edges)
            
        best_local_edges = list(edges)
        best_local_count = len(edges)
        
        for it in range(max_iters):
            # Try to add any valid edge in random order
            pairs = all_pairs[:]
            random.shuffle(pairs)
            added_any = False
            for u, v in pairs:
                if is_valid_add(u, v, adj_mask, common):
                    add_edge(u, v, adj_mask, common, edges)
                    added_any = True
                    if len(edges) > best_local_count:
                        best_local_count = len(edges)
                        best_local_edges = list(edges)
                    break
            
            if not added_any:
                # Local optimum - apply a kick
                if len(edges) == 0:
                    break
                
                # Remove 1-3 random edges to escape local optimum
                kick_size = random.choice([1, 1, 2, 2, 3])
                to_remove = random.sample(edges, min(kick_size, len(edges)))
                for u, v in to_remove:
                    remove_edge(u, v, adj_mask, common, edges)
                
                # After kick, try to fill again (limited steps)
                for _ in range(30):
                    pairs = all_pairs[:]
                    random.shuffle(pairs)
                    added = False
                    for u, v in pairs:
                        if is_valid_add(u, v, adj_mask, common):
                            add_edge(u, v, adj_mask, common, edges)
                            added = True
                            if len(edges) > best_local_count:
                                best_local_count = len(edges)
                                best_local_edges = list(edges)
                            break
                    if not added:
                        break
        
        return best_local_edges

    def verify_edges(edge_list):
        """Verify that the edge list forms a K2,3-free graph."""
        adj = [0] * n
        for u, v in edge_list:
            if u == v:
                return False
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if bin(common).count('1') > 2:
                    return False
        return True

    # Main search loop
    while time.time() - start_time < max_time:
        # Greedy initialization with random ordering
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        edges = []
        greedy_fill(adj_mask, common, edges)
        
        # Local search with kicks
        result_edges = local_search(edges)
        
        # Verify and update best
        if verify_edges(result_edges) and len(result_edges) > best_count:
            best_count = len(result_edges)
            best_edges = [[u, v] for u, v in result_edges]
        
        # Check time
        if time.time() - start_time > max_time - 5:
            break

    # Fallback if no edges found
    if not best_edges:
        # Try a simple construction: complete bipartite K10,10 is not K2,3-free
        # Use 5 disjoint K4s: each K4 has 6 edges, total 30 edges
        # K4 is K2,3-free since max common neighbors in K4 is 2
        best_edges = []
        for i in range(5):
            base = i * 4
            for u in range(4):
                for v in range(u + 1, 4):
                    best_edges.append([base + u, base + v])
    
    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
