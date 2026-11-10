import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    n = 20
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    m_total = len(all_edges)
    
    def get_initial_graph():
        """
        Create a random valid graph to start.
        We can try to build a dense graph greedily.
        """
        adj = [0] * n
        common = [[0] * n for _ in range(n)]
        # Try to add edges in random order
        edges = all_edges[:]
        random.shuffle(edges)
        for u, v in edges:
            if can_add(adj, common, u, v):
                do_add(adj, common, u, v)
        return adj, common

    def can_add(adj, common, u, v):
        if (adj[u] >> v) & 1:
            return True
        # Check constraints
        # 1. common[u][v] <= 2 (already true if valid)
        if common[u][v] >= 3:
            return False
        
        # 2. For all w in N(v) \ {u}, common[u][w] <= 1
        nv = adj[v] & ~(1 << u)
        while nv:
            lsb = nv & -nv
            w = lsb.bit_length() - 1
            if common[u][w] >= 2:
                return False
            nv ^= lsb
        
        # 3. For all w in N(u) \ {v}, common[v][w] <= 1
        nu = adj[u] & ~(1 << v)
        while nu:
            lsb = nu & -nu
            w = lsb.bit_length() - 1
            if common[v][w] >= 2:
                return False
            nu ^= lsb
            
        return True

    def do_add(adj, common, u, v):
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        # Update common counts
        # For each w in N(v), common[u][w] += 1
        nv = adj[v] & ~(1 << u)
        while nv:
            lsb = nv & -nv
            w = lsb.bit_length() - 1
            common[u][w] += 1
            common[w][u] += 1
            nv ^= lsb
            
        # For each w in N(u), common[v][w] += 1
        nu = adj[u] & ~(1 << v)
        while nu:
            lsb = nu & -nu
            w = lsb.bit_length() - 1
            common[v][w] += 1
            common[w][v] += 1
            nu ^= lsb
            
        # Update common[u][v] and common[v][u]
        # The number of common neighbors of u and v is the number of w in N(u) \cap N(v)
        # This is (adj[u] & adj[v]).bit_count() - 2 (excluding u and v themselves? No, u not in N(u))
        # Actually, common[u][v] should be the count of w such that w is neighbor of both.
        # Before adding, common[u][v] was count of w in N(u) \cap N(v).
        # After adding, does it change?
        # N(u) gains v. N(v) gains u.
        # v is not in N(v). u is not in N(u).
        # So N(u) \cap N(v) does not gain u or v.
        # So common[u][v] remains the same?
        # Wait.
        # Let's check.
        # If w is in N(u) \cap N(v), then w is neighbor of u and v.
        # Adding edge (u,v) doesn't change N(u) \cap N(v) for w != u,v.
        # So common[u][v] is unchanged.
        # However, we need to update common[u][v] to reflect the new state?
        # No, common[u][v] is defined as number of common neighbors.
        # So it stays the same.
        pass

    def do_remove(adj, common, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        # Update common counts (reverse of add)
        # For each w in N(v) \ {u}, common[u][w] -= 1
        # Note: after removal, v is not in N(u).
        # The neighbors w of v (excluding u) are the ones that lost u as a common neighbor?
        # No.
        # Before removal, u was in N(v).
        # For any w in N(v) \ {u}, w was in N(v).
        # Was u in N(w)?
        # If w was a common neighbor of u and v, then u in N(w) and v in N(w).
        # So w in N(u) \cap N(v).
        # So for w in N(u) \cap N(v), common[u][w] and common[v][w] decrease?
        
        # Let's just recalculate or use the symmetric logic.
        # When we added (u,v):
        # For w in N(v) (old, excluding u), common[u][w] += 1.
        # So when removing, for w in N(v) (current, excluding u), common[u][w] -= 1.
        nv = adj[v] & ~(1 << u)
        while nv:
            lsb = nv & -nv
            w = lsb.bit_length() - 1
            common[u][w] -= 1
            common[w][u] -= 1
            nv ^= lsb
            
        nu = adj[u] & ~(1 << v)
        while nu:
            lsb = nu & -nu
            w = lsb.bit_length() - 1
            common[v][w] -= 1
            common[w][v] -= 1
            nu ^= lsb
        pass

    def edge_count(adj):
        return sum(a.bit_count() for a in adj) // 2

    def is_valid(common):
        for u in range(n):
            for v in range(u + 1, n):
                if common[u][v] > 2:
                    return False
        return True

    best_adj = None
    best_common = None
    best_count = -1
    
    # Simulated Annealing
    rng = random.Random(42)
    
    while time.time() < deadline:
        adj, common = get_initial_graph()
        current_count = edge_count(adj)
        
        # Local improvement first
        improved = True
        while improved:
            improved = False
            # Shuffle edges to avoid bias
            edges = all_edges[:]
            rng.shuffle(edges)
            for u, v in edges:
                if not ((adj[u] >> v) & 1) and can_add(adj, common, u, v):
                    do_add(adj, common, u, v)
                    improved = True
        
        current_count = edge_count(adj)
        
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
            best_common = [row[:] for row in common]
            
        # Try some random moves to escape local optima?
        # Given the time, simple restarts might be better than SA if SA is not tuned well.
        # But let's try a few SA steps if time permits.
        
        if time.time() < deadline - 5.0:
            T = 50.0
            for _ in range(1000):
                if time.time() > deadline - 2.0:
                    break
                u, v = rng.choice(all_edges)
                is_present = (adj[u] >> v) & 1
                
                if is_present:
                    # Try to remove
                    do_remove(adj, common, u, v)
                    new_count = current_count - 1
                    # Check validity (should be valid if we remove)
                    # Calculate violation change? 
                    # Removing edges never creates violations.
                    # So it's always valid.
                    # We accept removal only if it allows adding more edges later?
                    # In SA, we might accept removal if T is high.
                    # But we want to maximize edges.
                    # So we should only remove if we can add back more.
                    # This is complex. Let's stick to greedy + restarts.
                    do_add(adj, common, u, v)
                else:
                    # Try to add
                    if can_add(adj, common, u, v):
                        do_add(adj, common, u, v)
                        current_count += 1
                    else:
                        # Cannot add. Do nothing.
                        pass
        
        # Final check
        current_count = edge_count(adj)
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
            best_common = [row[:] for row in common]

    if best_adj is None:
        # Fallback
        adj, common = get_initial_graph()
        best_adj = adj
    
    res = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                res.append([u, v])
    return res
