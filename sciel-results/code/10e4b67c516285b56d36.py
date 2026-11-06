import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin
    
    n = 16
    m_target = 65
    
    # Initialize with K_{6,10} + 5 matching edges in B
    # A: 0-5, B: 6-15
    adj = np.zeros((n, n), dtype=int)
    
    # Bipartite part
    for u in range(6):
        for v in range(6, 16):
            adj[u, v] = 1
            adj[v, u] = 1
            
    # Matching in B (6-15)
    matching_edges = [(6,7), (8,9), (10,11), (12,13), (14,15)]
    for u, v in matching_edges:
        adj[u, v] = 1
        adj[v, u] = 1
        
    deg = np.array([np.sum(adj[i]) for i in range(n)])
    
    # Verify edge count
    edge_count = np.sum(adj) // 2
    if edge_count != m_target:
        # Fallback to K8,8 + 1 if structure fails (it shouldn't)
        adj = np.zeros((n, n), dtype=int)
        for u in range(8):
            for v in range(8, 16):
                adj[u, v] = 1
                adj[v, u] = 1
        adj[0, 1] = 1
        adj[1, 0] = 1
        deg = np.array([np.sum(adj[i]) for i in range(n)])

    def calculate_score():
        """
        Fast calculation of max triangle degree sum using bitmask.
        """
        max_sum = 0
        # Precompute bitmask of neighbors for each vertex
        masks = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i, j]:
                    mask |= (1 << j)
            masks[i] = mask
        
        # Iterate over edges and find triangles
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    # Common neighbors of u and v
                    common = masks[u] & masks[v]
                    # Only consider w > v to avoid duplicates
                    common &= ~((1 << (v+1)) - 1)  # Clear bits 0..v
                    if common:
                        # Iterate over set bits
                        c = common
                        while c:
                            w = c & (-c)
                            c ^= w
                            wi = w.bit_length() - 1
                            s = deg[u] + deg[v] + deg[wi]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    current_score = calculate_score()
    best_score = current_score
    best_adj = adj.copy()
    
    # Local Search with improved move generation
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_set = set()
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                edge_set.add((i, j))
    
    non_edge_set = set()
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i, j]:
                non_edge_set.add((i, j))
    
    T = 1.0
    cooling = 0.999
    min_T = 0.05
    
    def apply_swap(rem, add):
        """Apply a swap and update deg, edge_set, non_edge_set."""
        u, v = rem
        x, y = add
        adj[u, v] = 0
        adj[v, u] = 0
        adj[x, y] = 1
        adj[y, x] = 1
        deg[u] -= 1
        deg[v] -= 1
        deg[x] += 1
        deg[y] += 1
        edge_set.discard(rem)
        edge_set.discard(add)
        non_edge_set.discard(rem)
        non_edge_set.discard(add)
        edge_set.add(add)
        non_edge_set.add(rem)
    
    def revert_swap(rem, add):
        """Revert a swap."""
        u, v = rem
        x, y = add
        adj[u, v] = 1
        adj[v, u] = 1
        adj[x, y] = 0
        adj[y, x] = 0
        deg[u] += 1
        deg[v] += 1
        deg[x] -= 1
        deg[y] -= 1
        edge_set.discard(rem)
        edge_set.discard(add)
        non_edge_set.discard(rem)
        non_edge_set.discard(add)
        edge_set.add(rem)
        non_edge_set.add(add)
    
    # Main search loop
    while time.time() < deadline - 10:  # Leave 10s for polish
        # Generate a random valid swap
        if not edge_set or not non_edge_set:
            break
            
        removed = random.choice(list(edge_set))
        added = random.choice(list(non_edge_set))
        
        u, v = removed
        x, y = added
        
        # Apply
        apply_swap(removed, added)
        
        new_score = calculate_score()
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = adj.copy()
        elif new_score == current_score:
            # Accept equal to explore
            pass
        else:
            # Metropolis
            delta = new_score - current_score
            if random.random() < np.exp(-delta / T):
                pass
            else:
                # Revert
                revert_swap(removed, added)
                
        T *= cooling
        if T < min_T:
            T = min_T
    
    # Final polish: more aggressive local search with tighter acceptance
    polish_deadline = deadline - 2
    polish_T = 0.1
    polish_cooling = 0.9995
    
    while time.time() < polish_deadline:
        if not edge_set or not non_edge_set:
            break
            
        # Try multiple random swaps, accept only improvements or small worsenings
        improved = False
        for _ in range(5):
            removed = random.choice(list(edge_set))
            added = random.choice(list(non_edge_set))
            
            u, v = removed
            x, y = added
            
            apply_swap(removed, added)
            new_score = calculate_score()
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_adj = adj.copy()
            elif new_score <= current_score + 1:
                # Accept small worsening for exploration
                if random.random() < np.exp(-(new_score - current_score) / polish_T):
                    current_score = new_score
                    improved = True
                else:
                    revert_swap(removed, added)
            else:
                revert_swap(removed, added)
        
        if not improved:
            polish_T *= polish_cooling
            if polish_T < 0.01:
                polish_T = 0.01
    
    # Reconstruct edge list from best_adj
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j]:
                final_edges.append([i, j])
                
    return {'n': n, 'edges': final_edges}
