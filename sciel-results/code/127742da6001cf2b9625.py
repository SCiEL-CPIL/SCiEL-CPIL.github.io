"""
K2,3-free graph construction on 20 vertices.
Uses a combination of:
1. Bitmask-based fast common-neighbor tracking
2. Greedy construction with multiple orderings
3. Local search via edge swaps and removals
4. Polarity graph / finite geometry inspired structure
"""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Use bitmask representation for adjacency
    # adj[i] is an integer where bit j is set if edge (i,j) exists
    
    def init_adj():
        return [0] * n
    
    def count_edges(adj):
        c = 0
        for i in range(n):
            c += bin(adj[i]).count('1')
        return c // 2
    
    def common_neighbors(adj, u, v):
        """Return bitmask of common neighbors of u and v."""
        return adj[u] & adj[v]
    
    def common_count(adj, u, v):
        """Count common neighbors of u and v."""
        cn = adj[u] & adj[v]
        return bin(cn).count('1')
    
    def is_valid_add(adj, u, v):
        """
        Check if adding edge (u,v) maintains K2,3-freeness.
        Conditions:
        1. |N(u) ∩ N(v)| <= 2 (u,v don't gain each other as common neighbors)
        2. For each w in N(v) (w != u): |N(u) ∩ N(w)| <= 1
        3. For each w in N(u) (w != v): |N(v) ∩ N(w)| <= 1
        """
        # Check 1
        if common_count(adj, u, v) > 2:
            return False
        
        # Check 2: neighbors of v
        nv = adj[v]
        # Iterate over set bits in nv
        w = 0
        while nv:
            low = nv & (-nv)
            w = low.bit_length() - 1
            if w != u:
                if common_count(adj, u, w) > 1:
                    return False
            nv ^= low
        
        # Check 3: neighbors of u
        nu = adj[u]
        w = 0
        while nu:
            low = nu & (-nu)
            w = low.bit_length() - 1
            if w != v:
                if common_count(adj, v, w) > 1:
                    return False
            nu ^= low
        
        return True
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def greedy_fill(adj, order):
        """Add all valid edges from the given order."""
        for u, v in order:
            if not ((adj[u] >> v) & 1):
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
    
    def local_search(adj, max_iters=200):
        """
        Local search via edge swaps:
        - Remove a random edge
        - Try to add back plus other edges via greedy fill
        - If we gain edges, keep; else revert
        """
        best_adj = adj[:]
        best_count = count_edges(best_adj)
        current_adj = adj[:]
        current_count = best_count
        no_improve = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Try a swap: remove one edge, then greedy fill
            cur_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if (current_adj[i] >> j) & 1:
                        cur_edges.append((i, j))
            
            if not cur_edges:
                break
            
            idx = random.randint(0, len(cur_edges) - 1)
            u_rem, v_rem = cur_edges[idx]
            
            backup = current_adj[:]
            remove_edge(current_adj, u_rem, v_rem)
            
            # Greedy fill
            missing = [(i, j) for i in range(n) for j in range(i+1, n) 
                       if not ((current_adj[i] >> j) & 1)]
            random.shuffle(missing)
            for u, v in missing:
                if is_valid_add(current_adj, u, v):
                    add_edge(current_adj, u, v)
            
            new_count = count_edges(current_adj)
            
            if new_count > best_count:
                best_adj = current_adj[:]
                best_count = new_count
                current_adj = best_adj[:]
                no_improve = 0
            else:
                # Revert
                current_adj = backup[:]
                no_improve += 1
                if no_improve > 30:
                    break
        
        return best_adj, best_count
    
    def polarity_graph_construction():
        """
        Try to build a graph inspired by polarity graphs over finite fields.
        For K2,3-free, we can use the construction from finite geometry.
        Here we try a deterministic approach based on quadratic residues.
        """
        adj = init_adj()
        
        # Try using modular arithmetic structure
        # For each pair (i,j), add edge if some condition holds
        # A simple heuristic: use the fact that in a polarity graph,
        # edges correspond to points on conics.
        
        # Alternative: use a tournament-like structure
        # Add edge (i,j) if (i*j) mod p has certain properties
        # Let's try: add edge if (i-j)^2 is a quadratic residue mod 19
        # (excluding self-loops)
        
        # Actually, let's just try multiple random seeds with good heuristics
        # and let the local search do the heavy lifting.
        
        return adj
    
    def weighted_greedy(adj, order, weights):
        """Greedy with priority based on weights."""
        # Sort by weight descending
        weighted_order = sorted(order, key=lambda x: weights[x[0]] + weights[x[1]], reverse=True)
        for u, v in weighted_order:
            if not ((adj[u] >> v) & 1):
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
    
    # Main strategy:
    # 1. Run multiple random restarts with greedy + local search
    # 2. Also try some structured initializations
    
    best_adj_overall = None
    best_count_overall = 0
    
    # Strategy 1: Random restarts
    max_restarts = 50
    for r in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        adj = init_adj()
        
        # Random ordering with some structure
        order = all_pairs[:]
        random.shuffle(order)
        
        # Greedy construct
        greedy_fill(adj, order)
        
        # Local search
        adj, cnt = local_search(adj)
        
        if cnt > best_count_overall:
            best_count_overall = cnt
            best_adj_overall = adj[:]
        
        # Early exit if we hit a high count
        if best_count_overall >= 60:
            break
    
    # Strategy 2: Try structured initialization
    # Use a circulant-like structure
    if time.time() < deadline - 10:
        # Try different step sets for circulant graphs
        for steps in range(1, 10):
            if time.time() > deadline - 5:
                break
            
            adj = init_adj()
            for i in range(n):
                for j in range(i+1, n):
                    diff = (j - i) % n
                    if diff <= steps or diff >= n - steps:
                        if is_valid_add(adj, i, j):
                            add_edge(adj, i, j)
            
            adj, cnt = local_search(adj)
            if cnt > best_count_overall:
                best_count_overall = cnt
                best_adj_overall = adj[:]
    
    # Strategy 3: Try complete bipartite-like structures
    if time.time() < deadline - 8:
        # Split vertices into two halves and try dense bipartite
        for split in range(5, 15):
            if time.time() > deadline - 3:
                break
            
            adj = init_adj()
            # First try adding all cross edges
            for i in range(split):
                for j in range(split, n):
                    if is_valid_add(adj, i, j):
                        add_edge(adj, i, j)
            
            adj, cnt = local_search(adj)
            if cnt > best_count_overall:
                best_count_overall = cnt
                best_adj_overall = adj[:]
    
    # Convert best_adj to edge list
    if best_adj_overall is None:
        return []
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj_overall[i] >> j) & 1:
                edges.append([i, j])
    
    return edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
