import time
import random
import math

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110.0

    def find_violation(adj):
        """Find a K3,3 violation. Returns (u, v, w) triple with >=3 common neighbors, or None."""
        for u in range(n):
            for v in range(u + 1, n):
                for w in range(v + 1, n):
                    common = adj[u] & adj[v] & adj[w]
                    if common.bit_count() >= 3:
                        return (u, v, w)
        return None

    def is_valid(adj):
        """Check if the graph is K3,3 free."""
        return find_violation(adj) is None

    def count_edges(adj):
        return sum(a.bit_count() for a in adj) // 2

    def is_valid_after_add_fast(adj, u, v):
        """
        Fast check: only verify triples that include the new edge (u,v).
        """
        nu = adj[u]  # before adding v
        nv = adj[v]  # before adding u
        
        # Triples (u, x, y) for x < y, x,y != v
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue
                old_common = nu & adj[x] & adj[y]
                if old_common.bit_count() == 2 and (nv & (1 << x)) and (nv & (1 << y)):
                    return False
        
        # Triples (v, x, y) for x < y, x,y != u
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue
                old_common = nv & adj[x] & adj[y]
                if old_common.bit_count() == 2 and (nu & (1 << x)) and (nu & (1 << y)):
                    return False
        
        return True

    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append((u, v))
        return edges

    def get_non_edges(adj):
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    non_edges.append((u, v))
        return non_edges

    # Phase 1: Greedy Construction
    # Start with an empty graph and add edges greedily.
    # We try multiple random orderings to find a good starting point.
    best_adj = [0] * n
    best_count = 0

    for trial in range(30):
        if time.time() > deadline - 30.0:
            break

        adj = [0] * n
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline - 30.0:
                break
            if adj[u] & (1 << v):
                continue
            # Tentatively add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not is_valid_after_add_fast(adj, u, v):
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        c = count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]

    # Phase 2: Simulated Annealing
    # We will try to improve the best_adj found in Phase 1.
    current_adj = best_adj[:]
    current_count = best_count
    best_sa_adj = current_adj[:]
    best_sa_count = current_count
    
    T = 1.0
    cooling = 0.9995
    min_T = 0.01
    
    edges = get_edges(current_adj)
    non_edges = get_non_edges(current_adj)
    
    while time.time() < deadline - 5.0 and T > min_T:
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.3:
            # Add a random non-edge
            if not non_edges:
                T *= cooling
                continue
            u, v = random.choice(non_edges)
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            
            if is_valid_after_add_fast(current_adj, u, v):
                new_count = current_count + 1
                # Always accept improvement
                current_count = new_count
                # Update non_edges list (remove this edge)
                # This is inefficient, but we can rebuild occasionally or just mark it
                # For simplicity, we'll just check validity and update count.
                # We need to maintain a valid list of non-edges for future steps.
                # Let's rebuild non_edges every 100 steps or if it changes significantly.
                # Actually, rebuilding every step is O(n^2) which is 190, acceptable.
                # But let's try to be smarter. We'll just pick from a static list and check if it's still a non-edge.
                pass
            else:
                # Revert
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                # Reject (hard constraint violation)
                T *= cooling
                continue
                
        elif move_type < 0.6:
            # Remove a random edge
            if not edges:
                T *= cooling
                continue
            u, v = random.choice(edges)
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            
            new_count = current_count - 1
            # Acceptance probability for worsening move
            delta_E = -1
            if random.random() < math.exp(-delta_E / T):
                current_count = new_count
            else:
                # Revert
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                
        else:
            # Swap: remove one edge, add another
            if not edges or not non_edges:
                T *= cooling
                continue
            u1, v1 = random.choice(edges)
            u2, v2 = random.choice(non_edges)
            
            # Remove (u1, v1)
            current_adj[u1] &= ~(1 << v1)
            current_adj[v1] &= ~(1 << u1)
            
            # Try add (u2, v2)
            current_adj[u2] |= (1 << v2)
            current_adj[v2] |= (1 << u2)
            
            if is_valid_after_add_fast(current_adj, u2, v2):
                # Count remains same
                # Accept with probability 1 (neutral) or based on some other metric?
                # Usually we accept neutral moves to explore.
                # But we must ensure we don't get stuck.
                # Let's accept neutral moves with probability 0.5
                if random.random() < 0.5:
                    # Keep change
                    pass
                else:
                    # Revert both
                    current_adj[u2] &= ~(1 << v2)
                    current_adj[v2] &= ~(1 << u2)
                    current_adj[u1] |= (1 << v1)
                    current_adj[v1] |= (1 << u1)
            else:
                # Revert both
                current_adj[u2] &= ~(1 << v2)
                current_adj[v2] &= ~(1 << u2)
                current_adj[u1] |= (1 << v1)
                current_adj[v1] |= (1 << u1)
                
        if current_count > best_sa_count:
            best_sa_count = current_count
            best_sa_adj = current_adj[:]
            
        # Rebuild edge lists periodically to keep them fresh
        if random.random() < 0.05:
            edges = get_edges(current_adj)
            non_edges = get_non_edges(current_adj)
            
        T *= cooling

    # Final check and return
    if best_sa_count > best_count:
        best_count = best_sa_count
        best_adj = best_sa_adj

    # Convert best_adj to list of edges
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                result.append([u, v])
    return result
