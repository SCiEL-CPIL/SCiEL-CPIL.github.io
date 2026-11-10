import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s
    
    n = 20
    best_edges = []
    best_count = 0
    rng = random.Random(42)
    
    # Precompute popcount for 20-bit integers? 
    # 2^20 is about 1 million. A list of 1M ints is ~8MB. This is acceptable.
    # Or just use int.bit_count() if available.
    # Let's use a safe method:
    def popcount(x):
        # Python 3.10+ has int.bit_count()
        if hasattr(int, "bit_count"):
            return x.bit_count()
        else:
            # Fallback
            return bin(x).count('1')
            
    def common_neighbors(adj, u, v):
        """Count common neighbors of u and v using bitsets."""
        return popcount(adj[u] & adj[v])
    
    def is_k23_free(adj):
        """Check no pair has >= 3 common neighbors."""
        for u in range(n):
            for v in range(u+1, n):
                if common_neighbors(adj, u, v) >= 3:
                    return False
        return True
        
    def count_violations(adj):
        """Count pairs with >= 3 common neighbors."""
        count = 0
        for u in range(n):
            for v in range(u+1, n):
                if common_neighbors(adj, u, v) >= 3:
                    count += 1
        return count
    
    def repair(adj):
        """Remove edges to eliminate all K2,3 violations."""
        changed = True
        while changed and (time.time() - start_time) < deadline - 5:
            changed = False
            for u in range(n):
                for v in range(u+1, n):
                    if (adj[u] >> v) & 1:
                        if common_neighbors(adj, u, v) >= 3:
                            # Remove edge (u, v)
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            changed = True
        return adj
    
    def greedy_add(adj):
        """Try to add edges greedily."""
        improved = True
        while improved and (time.time() - start_time) < deadline - 5:
            improved = False
            # Collect candidates
            candidates = []
            for u in range(n):
                for v in range(u+1, n):
                    if not ((adj[u] >> v) & 1):
                        cn = common_neighbors(adj, u, v)
                        if cn < 3:
                            candidates.append((cn, u, v))
            candidates.sort()
            
            for cn, u, v in candidates:
                if (adj[u] >> v) & 1:
                    continue
                # Check if adding (u, v) is valid
                # Adding (u, v) affects pairs (u, w) and (v, w)
                valid = True
                for w in range(n):
                    if w == u or w == v:
                        continue
                    # Check pair (u, w)
                    # If v is neighbor of w, common neighbors of (u, w) increases by 1
                    if (adj[v] >> w) & 1:
                        if common_neighbors(adj, u, w) >= 2: # becomes 3
                            valid = False
                            break
                    # Check pair (v, w)
                    # If u is neighbor of w, common neighbors of (v, w) increases by 1
                    if (adj[u] >> w) & 1:
                        if common_neighbors(adj, v, w) >= 2: # becomes 3
                            valid = False
                            break
                            
                if valid:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
        return adj
    
    def local_search(adj, max_iters=2000):
        """Local search: try edge swaps to increase edge count."""
        iters = 0
        while iters < max_iters and (time.time() - start_time) < deadline - 5:
            iters += 1
            added = False
            
            # Try to add a random edge
            for _ in range(10): # Reduced from 50
                u, v = rng.sample(range(n), 2)
                if not ((adj[u] >> v) & 1) and common_neighbors(adj, u, v) < 3:
                    # Check validity
                    valid = True
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        if (adj[v] >> w) & 1:
                            if common_neighbors(adj, u, w) >= 2:
                                valid = False
                                break
                        if (adj[u] >> w) & 1:
                            if common_neighbors(adj, v, w) >= 2:
                                valid = False
                                break
                    if valid:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        added = True
                        break
                        
            if not added:
                # Try a swap: remove edge (a,b), add edge (c,d)
                # Get list of existing edges
                edges_list = []
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj[i] >> j) & 1:
                            edges_list.append((i, j))
                
                if edges_list:
                    a, b = rng.choice(edges_list)
                    for _ in range(10): # Reduced from 30
                        c, d = rng.sample(range(n), 2)
                        if (adj[c] >> d) & 1 or (c, d) == (a, b) or (c, d) == (b, a):
                            continue
                        # Try swap
                        # Remove (a, b)
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                        # Add (c, d)
                        adj[c] |= (1 << d)
                        adj[d] |= (1 << c)
                        
                        # Check validity
                        valid = True
                        # Only need to check pairs involving a, b, c, d
                        # But to be safe and simple, check all pairs involving these vertices
                        # Actually, removing (a,b) can only *decrease* common neighbors for pairs (a,w) and (b,w).
                        # Adding (c,d) can only *increase* common neighbors for pairs (c,w) and (d,w).
                        # So we only need to check pairs (c,w) and (d,w) for w != c, d.
                        # And we must ensure that the removal didn't fix a violation that is now re-introduced? No, removal helps.
                        # So we only need to check the new edge's impact.
                        for w in range(n):
                            if w == c or w == d:
                                continue
                            if (adj[d] >> w) & 1:
                                if common_neighbors(adj, c, w) >= 3:
                                    valid = False
                                    break
                            if (adj[c] >> w) & 1:
                                if common_neighbors(adj, d, w) >= 3:
                                    valid = False
                                    break
                            
                        if valid:
                            break
                        else:
                            # Revert
                            adj[c] &= ~(1 << d)
                            adj[d] &= ~(1 << c)
                            adj[a] |= (1 << b)
                            adj[b] |= (1 << a)
        return adj
    
    # Multiple restarts
    for restart in range(100): # Increased restarts, faster per restart
        if time.time() - start_time > deadline - 10:
            break
        
        # Random initialization
        density = 0.3 + 0.4 * (restart % 5) / 5
        adj = [0] * n
        for u in range(n):
            for v in range(u+1, n):
                if rng.random() < density:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Repair
        adj = repair(adj)
        
        # Greedy add
        adj = greedy_add(adj)
        
        # Local search
        adj = local_search(adj, max_iters=1000) # Reduced iters
        
        # Count edges
        edge_count = sum(popcount(adj[i]) for i in range(n)) // 2
        if edge_count > best_count:
            best_count = edge_count
            best_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if (adj[i] >> j) & 1:
                        best_edges.append([i, j])
    
    return best_edges
