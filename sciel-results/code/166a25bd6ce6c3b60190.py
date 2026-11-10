import time
import numpy as np

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin

    def can_add_mask(adj_masks, u, v):
        """
        Check if adding edge (u,v) violates K3,3.
        A new K3,3 must use the edge (u,v).
        Case 1: u, a, b in one part, v, c, d in other.
        Requires v~a, v~b and |N(u) & N(a) & N(b)| >= 2.
        Case 2: v, a, b in one part, u, c, d in other.
        Requires u~a, u~b and |N(v) & N(a) & N(b)| >= 2.
        """
        mu = adj_masks[u]
        mv = adj_masks[v]
        
        for a in range(n):
            if a == u or a == v:
                continue
            ma = adj_masks[a]
            a_neigh_v = (ma >> v) & 1
            a_neigh_u = (ma >> u) & 1
            
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                mb = adj_masks[b]
                
                # Case 1: u, a, b are the triple on one side
                if a_neigh_v and ((mb >> v) & 1):
                    common = mu & ma & mb
                    if common.bit_count() >= 2:
                        return False

                # Case 2: v, a, b are the triple on one side
                if a_neigh_u and ((mb >> u) & 1):
                    common = mv & ma & mb
                    if common.bit_count() >= 2:
                        return False
        return True

    best_edges = []
    best_count = 0
    
    rng = np.random.default_rng(42)
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    num_restarts = 800
    
    for restart in range(num_restarts):
        if time.time() > deadline - 3:
            break
            
        adj_masks = [0] * n
        edges = []
        
        # Random order
        order = all_pairs[:]
        rng.shuffle(order)
        
        for u, v in order:
            if can_add_mask(adj_masks, u, v):
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges.append((u, v))
                
        current_count = len(edges)
        
        # Local Search: Greedy Add Pass (repeated until no improvement)
        improved = True
        while improved:
            if time.time() > deadline - 3:
                break
            improved = False
            # Build missing list
            missing = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not ((adj_masks[i] >> j) & 1):
                        missing.append((i, j))
            if not missing:
                break
            rng.shuffle(missing)
            
            for u, v in missing:
                if can_add_mask(adj_masks, u, v):
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    edges.append((u, v))
                    current_count += 1
                    improved = True
        
        # Local Search: Edge Swap (Remove 1, Add 2) - several rounds
        for _ in range(30):
            if time.time() > deadline - 3:
                break
            if not edges:
                break
            idx = int(rng.integers(0, len(edges)))
            u, v = edges[idx]
            
            # Remove (u,v)
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)
            edges.pop(idx)
            
            # Try to add 2 edges
            added = []
            missing = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (i, j) != (u, v) and not ((adj_masks[i] >> j) & 1):
                        missing.append((i, j))
            rng.shuffle(missing)
            
            for a, b in missing:
                if len(added) == 2:
                    break
                if can_add_mask(adj_masks, a, b):
                    adj_masks[a] |= (1 << b)
                    adj_masks[b] |= (1 << a)
                    edges.append((a, b))
                    added.append((a, b))
                    
            if len(added) < 2:
                # Revert
                for a, b in added:
                    adj_masks[a] &= ~(1 << b)
                    adj_masks[b] &= ~(1 << a)
                    edges.remove((a, b))
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges.append((u, v))
            else:
                # Improvement!
                current_count = len(edges)
                # Re-run greedy add to fill up
                while True:
                    if time.time() > deadline - 3:
                        break
                    improved = False
                    missing = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if not ((adj_masks[i] >> j) & 1):
                                missing.append((i, j))
                    if not missing:
                        break
                    rng.shuffle(missing)
                    for a, b in missing:
                        if can_add_mask(adj_masks, a, b):
                            adj_masks[a] |= (1 << b)
                            adj_masks[b] |= (1 << a)
                            edges.append((a, b))
                            current_count += 1
                            improved = True
                    if not improved:
                        break

        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
    return [list(e) for e in best_edges]
