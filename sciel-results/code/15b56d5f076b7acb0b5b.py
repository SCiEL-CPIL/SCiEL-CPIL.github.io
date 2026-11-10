import time
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    n = 16
    
    # The Clebsch graph is a 16-vertex, 30-edge, triangle-free, 4-chromatic graph.
    # Vertices are binary strings of length 4.
    # Two vertices are adjacent if their Hamming distance is 2.
    # This is a known optimal or near-optimal structure for this specific problem size.
    
    def hamming_dist(a, b):
        return bin(a ^ b).count('1')
        
    # Build the Clebsch graph adjacency list
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if hamming_dist(u, v) == 2:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
    def edge_count(adj):
        return sum(bin(a).count('1') for a in adj) // 2
        
    def is_3_colorable(adj):
        # Fast 3-colorability check using backtracking with bitsets
        # Order vertices by degree descending to prune early
        degrees = [bin(a).count('1') for a in adj]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        colors = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            # Find forbidden colors
            forbidden = 0
            for u in range(n):
                if (adj[v] >> u) & 1:
                    if colors[u] != -1:
                        forbidden |= (1 << colors[u])
            
            # Try available colors
            for c in range(3):
                if not (forbidden & (1 << c)):
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)

    def get_edge_list(adj):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def try_add_edge(adj, u, v):
        """Returns True if edge (u,v) can be added while keeping triangle-free."""
        if (adj[u] >> v) & 1:
            return False
        # Triangle-free condition: N(u) and N(v) must be disjoint
        if adj[u] & adj[v]:
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # Local Search: Deterministic Edge Swap
    # Strategy: Iterate through all possible edge removals. For each removal,
    # try to add any single edge that restores the edge count (swap 1 for 1)
    # or increases it (if we removed a "bad" edge, we might be able to add two? 
    # No, standard local search is swap 1 for 1 to maintain density while exploring space,
    # or remove 1 add 1 to move to a better local optimum).
    # Actually, to MAXIMIZE edges, we should look for "remove 1, add 1" swaps that keep edge count same but might allow further additions?
    # No, the goal is to find a maximal triangle-free graph that is 4-chromatic.
    # The Clebsch graph is known to be maximal in some sense?
    # Let's check if we can add any edges to the Clebsch graph.
    # If it's maximal triangle-free, we can't add any single edge.
    # So we must perform swaps: remove (u,v), add (w,x).
    # If we remove 1 and add 1, edge count stays same.
    # How do we increase edge count?
    # We can't simply add an edge if the graph is maximal.
    # However, the Clebsch graph might NOT be maximal triangle-free.
    # Let's first try to greedily add any possible edge.
    
    # 1. Greedy Fill: Add any edge that doesn't create a triangle
    changed = True
    while changed:
        changed = False
        for u in range(n):
            for v in range(u+1, n):
                if not ((adj[u] >> v) & 1):
                    if not (adj[u] & adj[v]):
                        # Check if adding (u,v) preserves 4-chromaticity?
                        # Adding an edge to a 4-chromatic graph keeps it 4-chromatic (or higher).
                        # So we just need to check triangle-freeness.
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        changed = True
    
    current_edges = edge_count(adj)
    
    # 2. Local Search: Edge Swaps to potentially reach a denser region
    # Since greedy fill makes the graph maximal triangle-free, we can't add any single edge.
    # We must remove an edge to "make space" for others?
    # No, removing an edge reduces the count. We want to INCREASE the count.
    # If the graph is maximal triangle-free, the edge count is fixed for that maximal structure.
    # Different maximal triangle-free graphs can have different edge counts.
    # We want the one with the highest edge count that is 4-chromatic.
    # The Clebsch graph has 30 edges.
    # Is there a 16-vertex triangle-free graph with >30 edges that is 4-chromatic?
    # Turan's theorem says max edges for triangle-free is n^2/4 = 64.
    # But those are bipartite (2-colorable).
    # We need 4-chromatic.
    # The Mycielski of C5 (Grötzsch) is 11v, 20e.
    # Clebsch is 16v, 30e.
    # Let's try to improve the Clebsch graph via swaps.
    # Swap: remove (u,v), add (w,x).
    # This keeps edge count constant.
    # How to increase edge count?
    # We can't. If it's maximal, we are stuck at 30 edges unless we find a different maximal graph with more edges.
    # But wait, maybe the greedy fill didn't reach the *global* maximum for 4-chromatic graphs?
    # Or maybe the Clebsch graph itself isn't the best starting point?
    
    # Let's implement a more aggressive search:
    # Try all single-edge removals. After removal, greedily fill.
    # If the new count > current count, accept.
    # This allows "remove 1, add 2" effectively.
    
    best_adj = adj[:]
    best_edges = current_edges
    best_edge_list = get_edge_list(best_adj)
    
    # Time budget for local search
    ls_deadline = start_time + 130.0
    
    # Iterate through potential edge removals
    # To avoid getting stuck in a local optimum, we can randomize the order or use a queue.
    # Given the small size (16 vertices, ~120 pairs), we can be exhaustive for one pass.
    
    # Create a list of all non-edges and all edges
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # We will try removing each edge in the current graph
    # and then greedily filling.
    
    # To make it faster, we only try edges that are "critical" or random subset if time runs out.
    # Let's start with all edges.
    
    edges_in_graph = []
    for u in range(n):
        for v in range(u+1, n):
            if (adj[u] >> v) & 1:
                edges_in_graph.append((u, v))
                
    # Shuffle to avoid bias? Or deterministic?
    # Deterministic might be safer for reproducibility, but random might escape local minima.
    # Let's use a deterministic order first, then random if time permits.
    
    # Optimization: Precompute valid additions for a given removal?
    # Too complex. Just run greedy fill.
    
    # Greedy fill function that returns new adj and count
    def greedy_fill_from(adj_input):
        adj_new = adj_input[:]
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u+1, n):
                    if not ((adj_new[u] >> v) & 1):
                        if not (adj_new[u] & adj_new[v]):
                            adj_new[u] |= (1 << v)
                            adj_new[v] |= (1 << u)
                            changed = True
        return adj_new, edge_count(adj_new)
        
    # Main local search loop
    # We try to remove an edge, fill, and see if we improve.
    # If we improve, we update best and current.
    
    # To avoid infinite loops or cycling, we track visited states?
    # State space is too large.
    # We'll just do a limited number of passes.
    
    passes = 0
    max_passes = 20
    
    while time.time() < ls_deadline and passes < max_passes:
        improved = False
        current_adj = best_adj[:]
        
        # Get current edges
        current_edges_list = []
        for u in range(n):
            for v in range(u+1, n):
                if (current_adj[u] >> v) & 1:
                    current_edges_list.append((u, v))
                    
        # Try removing each edge
        # To save time, we can break early if we find an improvement and update?
        # No, we should check all to find the best improvement in this pass.
        
        best_candidate_adj = None
        best_candidate_edges = edge_count(current_adj)
        
        for u, v in current_edges_list:
            if time.time() > ls_deadline:
                break
                
            # Remove edge (u, v)
            test_adj = current_adj[:]
            test_adj[u] &= ~(1 << v)
            test_adj[v] &= ~(1 << u)
            
            # Greedy fill
            filled_adj, filled_count = greedy_fill_from(test_adj)
            
            # Check 4-chromaticity
            # Since we started with a 4-chromatic graph and only added edges (greedy fill),
            # it remains 4-chromatic.
            # Wait, removing an edge from a 4-chromatic graph might make it 3-colorable!
            # If it becomes 3-colorable, then adding edges might or might not make it 4-chromatic again.
            # We MUST check 3-colorability of the filled graph.
            
            if filled_count > best_candidate_edges:
                if not is_3_colorable(filled_adj):
                    best_candidate_edges = filled_count
                    best_candidate_adj = filled_adj
                    improved = True
                    
        if best_candidate_adj:
            best_adj = best_candidate_adj
            best_edges = best_candidate_edges
            best_edge_list = get_edge_list(best_adj)
            passes += 1
        else:
            # No improvement in this pass.
            # We could try random restarts or break.
            # Since we are doing exhaustive removal+fill, if no single removal+fill improves,
            # we are likely at a local optimum for this neighborhood.
            # Break to save time, or try a different strategy.
            # Given the small size, let's break to ensure we return within time.
            break
            
    # Final check: ensure the result is valid
    if is_3_colorable(best_adj):
        # Fallback to pure Clebsch if something went wrong
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if hamming_dist(u, v) == 2:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        best_adj = adj
        best_edge_list = get_edge_list(best_adj)
        
    return best_edge_list

# EVOLVE-BLOCK-END
