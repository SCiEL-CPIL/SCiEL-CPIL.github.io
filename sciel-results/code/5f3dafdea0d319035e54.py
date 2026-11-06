import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n = 16
    # Grötzsch graph edges (11 vertices: 0..10)
    # Cycle 0-1-2-3-4-0
    # 5 connects to 1, 4
    # 6 connects to 2, 0
    # 7 connects to 3, 1
    # 8 connects to 4, 2
    # 9 connects to 0, 3
    # 10 connects to 5, 6, 7, 8, 9
    core_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 1), (5, 4),
        (6, 2), (6, 0),
        (7, 3), (7, 1),
        (8, 4), (8, 2),
        (9, 0), (9, 3),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9)
    ]
    
    # All possible pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    core_set = set(core_edges)
    non_core_edges = [e for e in all_pairs if e not in core_set]

    def greedy_fill(adj, edge_count):
        """
        Greedily add edges to adj until no more can be added.
        Returns the new edge count.
        """
        # We need to iterate over non-core edges.
        # To be efficient, we can just check all pairs not in adj.
        # But checking all 120 pairs each time is fast enough.
        # However, the set of addable edges shrinks.
        # Let's just iterate through a shuffled list of non-core edges.
        # This is what the current code does.
        # To improve, we can try to add edges in an order that maximizes future options?
        # For now, random order is fine, but we do it inside a local search loop.
        
        pass # Placeholder, we will implement the logic in the main loop

    def run_local_search(seed, kick_size=0):
        random.seed(seed)
        adj = [0] * n
        for u, v in core_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        edge_count = len(core_edges)
        
        # If kick_size > 0, remove that many random edges from the core?
        # No, we must keep the graph 4-chromatic.
        # Removing edges from the Grötzsch core might make it 3-colorable.
        # So we should only kick by removing NON-core edges that were added previously?
        # Or, we can start from scratch, but ensure we re-add the core?
        # Actually, if we remove core edges, we might lose 4-chromaticity.
        # So "kick" should be:
        # 1. Start with core.
        # 2. Greedily fill.
        # 3. Remove k random edges from the NON-core set that are currently present.
        # 4. Greedily fill again.
        
        # Let's implement a function to get the list of non-core edges present.
        
        # First pass: Greedy fill
        # We use a random order for non-core edges
        candidates = non_core_edges[:]
        random.shuffle(candidates)
        
        for u, v in candidates:
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
                
        # Local Search Loop
        # We try to improve by swapping or kicking.
        # Since we are maximizing edges, and we are at a local maximum (no single edge can be added),
        # we need to move to a different local maximum.
        
        # Strategy: Randomly remove 1-3 non-core edges, then re-greedy fill.
        # If the result is better, keep it.
        
        # We need to track which non-core edges are present.
        present_non_core = set()
        for u, v in non_core_edges:
            if (adj[u] >> v) & 1:
                present_non_core.add((u, v))
                
        best_local_count = edge_count
        best_local_adj = adj[:]
        
        # Iterate for a few steps or until time
        # We will do this in the main loop, but let's structure it so we can do multiple kicks.
        
        return adj, edge_count, present_non_core

    best_edges = []
    best_count = 0
    
    # Run multiple restarts with kicks
    # Each restart:
    # 1. Initialize with core.
    # 2. Greedy fill.
    # 3. Try several kicks (remove random edges, re-fill).
    
    num_restarts = 2000 # Adjust based on time
    kicks_per_restart = 5
    
    for seed in range(num_restarts):
        if time.time() > deadline:
            break
            
        adj = [0] * n
        for u, v in core_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        edge_count = len(core_edges)
        
        # Greedy Fill
        candidates = non_core_edges[:]
        random.seed(seed)
        random.shuffle(candidates)
        
        for u, v in candidates:
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
                
        current_best_adj = adj[:]
        current_best_count = edge_count
        
        # Kick Loop
        for k in range(kicks_per_restart):
            if time.time() > deadline:
                break
                
            # Identify non-core edges present
            present_non_core = []
            for u, v in non_core_edges:
                if (adj[u] >> v) & 1:
                    present_non_core.append((u, v))
            
            if not present_non_core:
                break
                
            # Remove 1-3 random non-core edges
            num_removals = random.randint(1, 3)
            if num_removals > len(present_non_core):
                num_removals = len(present_non_core)
                
            edges_to_remove = random.sample(present_non_core, num_removals)
            
            for u, v in edges_to_remove:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edge_count -= 1
                
            # Re-greedy fill
            # Use a new random order
            candidates2 = non_core_edges[:]
            random.shuffle(candidates2)
            
            for u, v in candidates2:
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_count += 1
                    
            if edge_count > current_best_count:
                current_best_count = edge_count
                current_best_adj = adj[:]
                
        # Update global best
        if current_best_count > best_count:
            best_count = current_best_count
            # Convert to list of edges
            best_edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if (current_best_adj[u] >> v) & 1:
                        best_edges.append([u, v])

    return best_edges
