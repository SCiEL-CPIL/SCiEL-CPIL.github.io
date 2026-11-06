"""Rewritten program: bitmask-based simulated annealing to maximize edges in K2,3-free graph on 20 vertices."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130.0  # safety margin before 150s kill
    n = 20
    # Adjacency masks: adj[i] is a bitmask of neighbors of vertex i
    # We maintain this incrementally.
    
    # Precompute all possible edges as (u, v) with u < v
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_index = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)
    
    def init_random():
        """Initialize with a random K2,3-free graph by greedily adding random edges."""
        adj = [0] * n
        edges_set = set()
        # Shuffle edges
        shuffled = all_edges[:]
        random.shuffle(shuffled)
        for (u, v) in shuffled:
            # Check if adding (u,v) violates K2,3
            # Common neighbors of u and v must be <= 2
            common = adj[u] & adj[v]
            if common.bit_count() <= 2:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges_set.add((u, v))
        return adj, edges_set
    
    def count_edges(edges_set):
        return len(edges_set)
    
    def can_add(adj, u, v):
        """Check if edge (u,v) can be added without creating K2,3."""
        common = adj[u] & adj[v]
        return common.bit_count() <= 2
    
    def add_edge(adj, edges_set, u, v):
        """Add edge (u,v), assuming it's valid."""
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edges_set.add((u, v))
    
    def remove_edge(adj, edges_set, u, v):
        """Remove edge (u,v)."""
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        edges_set.discard((u, v))
    
    # Start with random initialization
    adj, edges_set = init_random()
    best_adj = adj[:]
    best_edges_set = edges_set.copy()
    best_count = len(edges_set)
    
    # Simulated annealing
    T = 5.0  # initial temperature
    T_min = 0.01
    cooling_rate = 0.9995
    
    # Precompute for faster common-neighbor checks
    # For each pair (u,v), we need to count common neighbors
    # This is adj[u] & adj[v] bit_count
    
    while time.time() < deadline and T > T_min:
        # Pick a random edge to toggle
        u, v = random.choice(all_edges)
        if (u, v) in edges_set:
            # Try removing
            new_count = len(edges_set) - 1
            delta = new_count - best_count  # not used for acceptance, just for tracking
            # Always accept removals in SA? No, we want to maximize, so removals are bad
            # But we need them to escape local optima
            # Accept with probability based on temperature
            if random.random() < np.exp(1.0 / T):  # exp(-delta/T) where delta = -1
                remove_edge(adj, edges_set, u, v)
        else:
            # Try adding
            if can_add(adj, u, v):
                add_edge(adj, edges_set, u, v)
                new_count = len(edges_set)
                if new_count > best_count:
                    best_count = new_count
                    best_adj = adj[:]
                    best_edges_set = edges_set.copy()
            # If can't add, skip (no point in trying invalid moves)
        
        T *= cooling_rate
        
        # Periodic restarts if stuck
        # Check if we've been stuck for a while by comparing current to best
        # Simple approach: if T drops too low, restart with current best
    
    # Return best found
    return [[u, v] for (u, v) in best_edges_set]
# EVOLVE-BLOCK-END
