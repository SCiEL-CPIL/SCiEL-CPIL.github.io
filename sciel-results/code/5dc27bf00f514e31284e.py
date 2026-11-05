import random
import time
import math

def construct():
    n = 31
    target_edges = 91
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_set = set(all_pairs)
    
    def count_c4(adj):
        """
        adj is a list of integers (bitmasks) representing neighbors.
        Returns the number of 4-cycles.
        """
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = (adj[u] & adj[v]).bit_count()
                if common >= 2:
                    total += common * (common - 1) // 2
        return total // 2 # Each C4 is counted twice (once for each pair of opposite vertices)

    # Initial graph: Greedy C4-free + 1 random edge
    # This provides a good starting point close to the optimum (likely 1 or 2 C4s)
    
    best_adj = None
    best_score = float('inf')
    
    start_time = time.time()
    time_limit = 115
    
    # Generate a few initial candidates to pick the best start
    for _ in range(10):
        adj = [0] * n
        edge_count = 0
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if (adj[u] & adj[v]).bit_count() <= 1:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
                if edge_count == 90:
                    break
        
        # Add 1 random edge
        remaining = [p for p in all_pairs if not (adj[p[0]] & (1 << p[1]))]
        if remaining:
            u, v = random.choice(remaining)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edge_count += 1
            
        if edge_count == 91:
            score = count_c4(adj)
            if score < best_score:
                best_score = score
                best_adj = adj[:]
                
    if best_adj is None:
        # Fallback
        return [[i, j] for i, j in all_pairs[:91]]
        
    current_adj = best_adj[:]
    current_score = best_score
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # Pick a random edge to remove
        # We need a list of current edges
        # For efficiency, we can just pick random u, v and check if edge exists?
        # No, better to maintain a list of edges or just sample from all_pairs and check.
        # Given n=31, checking 100 random pairs to find an edge is fast.
        
        # Better: maintain a list of edges for the current graph?
        # Or just use the adjacency matrix to find an edge.
        
        # Let's just pick random pairs from all_pairs until we find an edge and a non-edge.
        # This is O(1) expected time per step.
        
        while True:
            u1, v1 = random.choice(all_pairs)
            if current_adj[u1] & (1 << v1):
                break
        
        while True:
            u2, v2 = random.choice(all_pairs)
            if not (current_adj[u2] & (1 << v2)):
                break
        
        # Perform swap
        current_adj[u1] &= ~(1 << v1)
        current_adj[v1] &= ~(1 << u1)
        current_adj[u2] |= (1 << v2)
        current_adj[v2] |= (1 << u2)
        
        new_score = count_c4(current_adj)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = current_adj[:]
                if best_score == 1:
                    break
        else:
            # Revert
            current_adj[u1] |= (1 << v1)
            current_adj[v1] |= (1 << u1)
            current_adj[u2] &= ~(1 << v2)
            current_adj[v2] &= ~(1 << u2)
            
        T *= cooling
        if T < T_min:
            T = T_min
            
    # Reconstruct edge list from best_adj
    edges = []
    for u in range(n):
        mask = best_adj[u]
        v = u + 1
        while mask:
            if mask & 1:
                edges.append([u, v])
            mask >>= 1
            v += 1
            
    return edges

