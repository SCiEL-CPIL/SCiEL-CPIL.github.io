import math
import random
import time

def construct():
    deadline = time.time() + 80  # 80s safety margin before 90s kill
    
    n = 256
    # Build adjacency: adj[i] is set of j (i<=j) where i+j is a perfect square
    # Perfect squares from 4 to 512
    squares = set()
    k = 2
    while k * k <= 512:
        squares.add(k * k)
        k += 1
    
    # Diagonal constraint: 2*a must not be a perfect square
    # So exclude a where 2a is in squares
    excluded_diagonal = set()
    for a in range(1, 257):
        if 2 * a in squares:
            excluded_diagonal.add(a)
    
    # Build adjacency list (only for valid vertices, i.e., not excluded_diagonal)
    # adj[i] = set of j >= i such that i+j is a perfect square (conflict)
    adj = [set() for _ in range(n + 1)]
    for a in range(1, 257):
        if a in excluded_diagonal:
            continue
        for b in range(a, 257):
            if b in excluded_diagonal:
                continue
            if a + b in squares:
                adj[a].add(b)
                if b != a:
                    adj[b].add(a)
    
    valid_nodes = [a for a in range(1, 257) if a not in excluded_diagonal]
    
    def greedy_min_degree(nodes, rng):
        """Min-degree greedy: repeatedly pick node with min degree in remaining graph."""
        if not nodes:
            return []
        
        remaining = set(nodes)
        # Compute degrees within remaining
        result = []
        
        while remaining:
            # Find node with minimum degree in remaining subgraph
            min_deg = float('inf')
            best = None
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best = v
            
            result.append(best)
            remaining.discard(best)
            # Remove neighbors of best from remaining
            for nb in adj[best]:
                remaining.discard(nb)
        
        return result
    
    def greedy_randomized(nodes, rng):
        """Randomized min-degree greedy with some randomness for diversity."""
        if not nodes:
            return []
        
        remaining = set(nodes)
        result = []
        
        while remaining:
            # Compute degrees
            degrees = {}
            min_deg = float('inf')
            for v in remaining:
                deg = len(adj[v] & remaining)
                degrees[v] = deg
                if deg < min_deg:
                    min_deg = deg
            
            # Get all nodes with min degree (or near-min for diversity)
            candidates = [v for v in remaining if degrees[v] <= min_deg + 1]
            if not candidates:
                candidates = list(remaining)
            
            best = rng.choice(candidates)
            result.append(best)
            remaining.discard(best)
            for nb in adj[best]:
                remaining.discard(nb)
        
        return result
    
    # Try multiple strategies and restarts
    best_set = set()
    best_size = 0
    
    # Strategy 1: Deterministic min-degree greedy
    result = greedy_min_degree(valid_nodes, random.Random(42))
    best_set = set(result)
    best_size = len(result)
    
    # Strategy 2: Multiple randomized restarts
    rng = random.Random(12345)
    num_restarts = 0
    while time.time() < deadline - 5:
        # Shuffle and try randomized greedy
        shuffled = valid_nodes[:]
        rng.shuffle(shuffled)
        result = greedy_randomized(shuffled, rng)
        if len(result) > best_size:
            best_size = len(result)
            best_set = set(result)
        num_restarts += 1
        
        # Also try pure min-degree with different initial orderings
        result2 = greedy_min_degree(shuffled, rng)
        if len(result2) > best_size:
            best_size = len(result2)
            best_set = set(result2)
        num_restarts += 1
    
    # Try to improve: local search - try to add nodes that don't conflict
    # Sort by degree ascending for better greedy
    sorted_nodes = sorted(valid_nodes, key=lambda x: len(adj[x]))
    result = greedy_min_degree(sorted_nodes, rng)
    if len(result) > best_size:
        best_size = len(result)
        best_set = set(result)
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
