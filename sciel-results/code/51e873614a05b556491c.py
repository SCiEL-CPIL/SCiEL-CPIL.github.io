"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt
import itertools

def construct():
    """
    Finds a large subset of [1, 512] such that lcm(a, b) <= 512 for all pairs.
    Uses Branch and Bound on the compatibility graph to find a maximum clique.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    N = 512
    nodes = list(range(1, N + 1))
    
    # Build adjacency matrix/list
    # Two nodes u, v are connected if lcm(u, v) <= 512
    # lcm(u, v) = u * v // gcd(u, v)
    
    # Precompute gcds? Or just check on the fly. 
    # N=512 is small enough for O(N^2) setup.
    
    adj = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            l = (i * j) // gcd(i, j)
            if l <= N:
                adj[i].add(j)
                adj[j].add(i)
    
    # Heuristic: Order nodes by degree (descending) to prune B&B effectively
    # We want to pick high-degree nodes first.
    # However, B&B on Max Clique usually works best if we process vertices in a specific order.
    # Let's create a list of candidates sorted by degree descending.
    # Actually, standard Tomita algorithm for Max Clique uses a specific vertex ordering.
    # Let's use a simple recursive B&B with a global best set.
    
    best_clique = []
    
    # Initial heuristic: Greedy clique to get a good lower bound
    current_clique = []
    remaining = set(nodes)
    while remaining:
        # Pick node with highest degree in remaining
        # For speed, just pick max degree
        max_deg_node = max(remaining, key=lambda x: len(adj[x] & remaining))
        current_clique.append(max_deg_node)
        # Intersect remaining with neighbors of max_deg_node
        remaining = remaining & adj[max_deg_node]
    
    best_clique = current_clique
    best_size = len(best_clique)
    
    # Branch and Bound
    # We will search for a clique larger than best_size.
    # State: (current_clique, candidates)
    # candidates are nodes that can be added to current_clique (adjacent to all in current_clique)
    
    # To improve B&B, we should order the initial candidates.
    # Let's sort nodes by degree descending.
    sorted_nodes = sorted(nodes, key=lambda x: len(adj[x]), reverse=True)
    
    def bnb(clique, candidates, depth):
        nonlocal best_clique, best_size, deadline
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if len(clique) > best_size:
                best_clique = clique[:]
                best_size = len(clique)
            return
        
        # Pruning: If even taking all remaining candidates doesn't beat best_size, stop
        if len(clique) + len(candidates) <= best_size:
            return
        
        # Heuristic upper bound: 
        # Simple: len(clique) + len(candidates)
        # Better: Color bound (greedy coloring of candidates)
        # For N=512, simple bound might be enough if we order well.
        
        # Try each candidate in candidates
        # Order candidates by degree in the subgraph induced by candidates?
        # Or just iterate.
        
        # To speed up, pick a candidate that is likely to be in the max clique.
        # Let's iterate through candidates.
        
        # Optimization: If we find a clique of size N, we can stop.
        # Max possible size is small (around 25-30 based on p4).
        
        # We will iterate over candidates.
        # To ensure we explore the "best" branches first, we can sort candidates by degree.
        # But doing this at every node is expensive.
        
        # Let's use a standard recursive step.
        # Pick a candidate u.
        # Branch 1: Include u. New candidates = candidates \ {u} intersect adj[u].
        # Branch 2: Exclude u. New candidates = candidates \ {u}.
        
        # To be efficient, we usually pick a specific vertex to branch on, often the last one added or one with high degree.
        # A common trick is to branch on the vertex with the highest degree in the candidate set.
        
        # Let's find the vertex in candidates with the highest degree within candidates.
        max_deg_u = None
        max_deg_val = -1
        for u in candidates:
            deg = len(adj[u] & candidates)
            if deg > max_deg_val:
                max_deg_val = deg
                max_deg_u = u
        
        if max_deg_u is None:
            return
            
        u = max_deg_u
        new_candidates = candidates - {u}
        
        # Branch 1: Include u
        # Check if u is connected to all in clique (it is, by definition of candidates)
        clique.append(u)
        new_candidates_include = new_candidates & adj[u]
        bnb(clique, new_candidates_include, depth + 1)
        clique.pop()
        
        # Branch 2: Exclude u
        # We can prune if len(clique) + len(new_candidates) <= best_size
        if len(clique) + len(new_candidates) > best_size:
            bnb(clique, new_candidates, depth + 1)

    # Start B&B
    # Initial candidates: All nodes
    # We sort initial candidates by degree to help the first few steps?
    # The B&B function picks max degree internally, so initial order matters less.
    
    bnb([], set(nodes), 0)
    
    # Convert set to list for return
    # The problem asks for a subset. Order doesn't matter, but list is standard.
    return sorted(best_clique)

# EVOLVE-BLOCK-END
