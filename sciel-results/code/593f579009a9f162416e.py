import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110
    n = 40
    
    # 1. Initialize with 8 disjoint K5s (104 edges)
    edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append((5 * k + a, 5 * k + b))
    
    # Bitset representation for fast operations
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    current_edges = set(edges)
    best_edges = set(edges)
    best_score = len(best_edges)
    
    # 2. Generate candidate edges (inter-clique edges)
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u // 5) != (v // 5):
                candidates.append((u, v))

    # 3. Fast C6 check
    def has_c6_fast(u, v):
        mask_u = adj_mask[u]
        mask_v = adj_mask[v]
        
        # w in N(u) \ {v}
        w_mask = mask_u & ~(1 << v)
        
        while w_mask:
            w_bit = w_mask & -w_mask
            w = w_bit.bit_length() - 1
            w_mask ^= w_bit
            
            # x in N(w) \ {u, v, w}
            mask_w = adj_mask[w]
            x_mask = mask_w & ~(1 << u) & ~(1 << v) & ~(1 << w)
            
            while x_mask:
                x_bit = x_mask & -x_mask
                x = x_bit.bit_length() - 1
                x_mask ^= x_bit
                
                # y in N(x) \ {u, v, w, x}
                mask_x = adj_mask[x]
                y_mask = mask_x & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x)
                
                while y_mask:
                    y_bit = y_mask & -y_mask
                    y = y_bit.bit_length() - 1
                    y_mask ^= y_bit
                    
                    # z in N(y) \ {u, v, w, x, y} AND z in N(v)
                    mask_y = adj_mask[y]
                    z_mask = mask_y & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x) & ~(1 << y)
                    
                    if z_mask & mask_v:
                        return True
        return False

    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        current_edges.add((u, v))

    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        current_edges.remove((u, v))

    def greedy_add():
        """Try to add all possible safe edges. Returns number of edges added."""
        added = 0
        # Shuffle candidates to avoid bias
        # We only iterate over candidates not in current_edges
        # For speed, we just check all candidates
        for u, v in candidates:
            if (u, v) in current_edges:
                continue
            if not has_c6_fast(u, v):
                add_edge(u, v)
                added += 1
        return added

    # Initial greedy pass
    greedy_add()
    if len(current_edges) > best_score:
        best_score = len(current_edges)
        best_edges = set(current_edges)

    random.seed(42)
    
    # 4. Local Search with Multi-Edge Removal
    # Strategy: Remove k random edges, then run greedy_add.
    # If the result is better, keep it. Otherwise, revert.
    
    # We will use a "restart" strategy with increasing k?
    # Or just fixed k=3.
    
    no_progress = 0
    max_no_progress = 50
    
    while time.time() < deadline:
        # Pick k edges to remove
        k = random.randint(2, 5)
        if len(current_edges) < k:
            break
            
        # Select k random edges to remove
        # We should prioritize removing edges that are "peripheral" or random
        # Let's just pick random edges from current_edges
        edges_to_remove = random.sample(list(current_edges), k)
        
        # Save state
        state_backup = set(current_edges)
        mask_backup = adj_mask[:]
        
        # Remove edges
        for u, v in edges_to_remove:
            remove_edge(u, v)
            
        # Run greedy add
        greedy_add()
        
        # Check if improved
        if len(current_edges) > best_score:
            best_score = len(current_edges)
            best_edges = set(current_edges)
            no_progress = 0
        else:
            # Revert
            # Clear current
            for u, v in current_edges:
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
            current_edges.clear()
            adj_mask = mask_backup
            current_edges = state_backup
            no_progress += 1
            
        if no_progress > max_no_progress:
            # Try larger k or restart from best?
            # Let's restart from best_edges to try different directions
            # Clear current
            for u, v in current_edges:
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
            current_edges.clear()
            adj_mask = [0]*n
            for u, v in best_edges:
                add_edge(u, v)
            no_progress = 0

    return [[a, b] for (a, b) in sorted(best_edges)]
