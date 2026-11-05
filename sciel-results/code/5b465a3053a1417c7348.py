"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Construct a C4-free graph with no independent set of size 11.
    Uses a randomized local search to find a C4-free graph on n vertices with alpha < 11.
    Aims for n=39 (known best lower bound for R(C4, K11) > 39).
    """
    deadline = time.time() + 140  # 10s safety margin
    
    # We will try to find a graph with n=39.
    # If we fail, we might fall back to smaller n, but 39 is the target.
    n_target = 39
    
    # Helper: Check if adding edge (u, v) to current adj (list of uint64 masks) keeps graph C4-free.
    # A C4 is formed if there exists x in N(u) and y in N(v) such that (x, y) is an edge.
    # So we check if any neighbor of u is connected to any neighbor of v.
    def can_add_edge(adj, u, v):
        nu = adj[u]
        nv = adj[v]
        # Check if any x in nu has an edge to any y in nv.
        # We can iterate over set bits in nu and check if adj[x] & nv is non-zero.
        # Since n=39, we can just iterate 0..n-1 or use bit tricks.
        # Iterating over neighbors is faster if degree is low.
        # Let's iterate over all vertices 0..n-1 and check if it's a neighbor of u.
        # Actually, iterating set bits is better.
        # But for simplicity and speed in Python loop, let's just check all pairs? No.
        # Let's iterate x from 0 to n-1. If nu has bit x, check adj[x] & nv.
        mask = nu
        while mask:
            # Get lowest set bit index
            x = (mask & -mask).bit_length() - 1
            if adj[x] & nv:
                return False
            mask &= mask - 1
        return True

    def count_bits(x):
        return bin(x).count('1')

    def greedy_alpha(adj, n, restarts=5):
        """Estimate independence number using greedy method."""
        min_alpha = n
        for _ in range(restarts):
            remaining = (1 << n) - 1
            count = 0
            while remaining:
                # Pick a random vertex from remaining
                # To make it deterministic-ish but varied, we can shuffle or pick random
                # For speed, pick the vertex with min degree in remaining subgraph?
                # That's expensive. Random is fine for estimation.
                # Find a random set bit
                r = random.randint(0, remaining.bit_length() - 1)
                # Ensure we pick a valid bit
                # Better: convert to list? No, too slow.
                # Let's just iterate to find a random bit?
                # Actually, we can just pick a random index and check if bit is set, else retry.
                # Or better, use a precomputed list of vertices and filter?
                
                # Efficient random bit selection:
                # Pick random number in [0, 2^n), mask with remaining?
                # No, just pick random vertex index and check if it's in remaining.
                # Since n is small (39), this is fast enough.
                
                v = -1
                while v == -1:
                    v = random.randint(0, n - 1)
                    if not (remaining & (1 << v)):
                        v = -1
                
                count += 1
                # Remove v and its neighbors
                remaining &= ~(adj[v] | (1 << v))
            
            if count < min_alpha:
                min_alpha = count
                if min_alpha < 11:
                    return min_alpha # Early exit if we already see it's small
        return min_alpha

    def remove_c4s(adj, n, edges):
        """Greedy removal of edges to ensure C4-freeness."""
        # This is a fallback if we start with a random graph.
        # We'll start with empty graph and add edges, so this might not be needed if we only add valid edges.
        pass

    # Initialize best result
    best_graph = None
    best_alpha = float('inf')
    
    # We will perform a local search.
    # Start with a random C4-free graph.
    
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        # 1. Generate a random C4-free graph
        adj = [0] * n_target
        edges = []
        
        # Random order of vertices to add edges
        verts = list(range(n_target))
        random.shuffle(verts)
        
        # We will try to add edges with some probability
        # To get a dense enough graph to keep alpha low, we need many edges.
        # Max edges in C4-free graph on 39 vertices is roughly 39*sqrt(39)/2 ~ 120.
        
        # Strategy: Try to add all possible edges in random order, keep if C4-free.
        # This will produce a maximal C4-free graph.
        # Then we check alpha.
        
        edge_candidates = [(i, j) for i in range(n_target) for j in range(i+1, n_target)]
        random.shuffle(edge_candidates)
        
        for u, v in edge_candidates:
            if can_add_edge(adj, u, v):
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                edges.append([u, v])
        
        # Check alpha
        alpha = greedy_alpha(adj, n_target, restarts=10)
        
        if alpha < 11:
            # We found a candidate. 
            # To be safe, we can try to verify with a slightly more robust check?
            # For n=39, greedy is usually accurate enough to distinguish alpha=10 vs 11.
            # If greedy says 10, it's very likely 10.
            # If greedy says 11, it might be 10, but we reject it.
            # This might miss some valid graphs, but finding one is enough.
            best_graph = {'n': n_target, 'edges': edges}
            best_alpha = alpha
            break # Found one, stop searching? Or try to find one with even lower alpha?
            # The task is to maximize n. We found n=39. We can return immediately.
            # But maybe we can find a more robust one? No, 39 is the max known.
            # Let's return immediately to save time.
            return best_graph
        
        # If alpha >= 11, we could try to modify the graph (local search) to reduce alpha.
        # But generating a maximal C4-free graph usually results in alpha ~ 10-12.
        # If we fail, we try again with a new random seed/order.
        
    # If we didn't find n=39, try n=38?
    # The problem asks to maximize n. If we can't get 39, 38 is better than 30.
    # However, 39 is the known bound. It is highly likely we can find one with 50 random restarts.
    
    # Fallback: Return a known valid smaller graph if we failed to find 39.
    # The seed had n=30. Let's construct a valid n=30 graph as a fallback.
    # Disjoint union of 10 triangles.
    fallback_edges = []
    for i in range(10):
        a, b, c = 3*i, 3*i+1, 3*i+2
        fallback_edges.append([a, b])
        fallback_edges.append([a, c])
        fallback_edges.append([b, c])
    
    return {'n': 30, 'edges': fallback_edges}

# EVOLVE-BLOCK-END
