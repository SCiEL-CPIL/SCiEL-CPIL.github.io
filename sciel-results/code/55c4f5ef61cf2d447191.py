import time
import random

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin for 150s limit

    def elapsed():
        return time.time() - start_time

    def build_edges(adj, n):
        edges = []
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                if ai & (1 << j):
                    edges.append([i, j])
        return edges

    def min_deg(adj, n):
        m = n
        for i in range(n):
            d = adj[i].bit_count()
            if d < m:
                m = d
        return m

    def find_and_remove_c4(adj, n):
        """
        Finds one C4 and removes one edge from it.
        Returns True if a C4 was found and an edge removed.
        Returns False if no C4 exists.
        """
        for u in range(n):
            adj_u = adj[u]
            for v in range(u + 1, n):
                common = adj_u & adj[v]
                if common.bit_count() >= 2:
                    # Find two distinct common neighbors
                    tmp = common
                    c1 = -1
                    c2 = -1
                    while tmp:
                        lsb = tmp & -tmp
                        idx = lsb.bit_length() - 1
                        if c1 == -1:
                            c1 = idx
                        else:
                            c2 = idx
                            break
                        tmp ^= lsb
                    
                    if c1 != -1 and c2 != -1:
                        # C4 is u - c1 - v - c2 - u
                        # Remove edge (u, c1)
                        adj[u] &= ~(1 << c1)
                        adj[c1] &= ~(1 << u)
                        return True
        return False

    def try_n(n, target_deg, time_budget):
        # Try multiple random seeds/probabilities
        # Higher p gives higher min degree but more C4s
        # We want min deg >= target, so start reasonably high
        # For n=47, target=8. p=0.2 -> exp deg 9.4. p=0.3 -> exp deg 14.1.
        
        ps = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        # Shuffle ps to vary attempts
        random.shuffle(ps)
        
        for p in ps:
            if elapsed() > start_time + time_budget:
                break
                
            for attempt in range(100):
                if elapsed() > start_time + time_budget:
                    break
                
                # Generate random graph
                adj = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < p:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                
                # Check initial min degree
                if min_deg(adj, n) < target_deg:
                    continue
                
                # Repair phase: remove edges to break C4s
                # We must ensure min degree doesn't drop below target
                max_iter = 2000
                for _ in range(max_iter):
                    if elapsed() > start_time + time_budget:
                        break
                    
                    # Check if we still satisfy min degree constraint
                    # Doing this every step is O(n). With n=47, it's cheap.
                    if min_deg(adj, n) < target_deg:
                        break # Failed this attempt
                    
                    if not find_and_remove_c4(adj, n):
                        # No C4s left. Valid graph!
                        if min_deg(adj, n) >= target_deg:
                            return build_edges(adj, n)
                        else:
                            break # Should not happen if we checked above, but safe
                
                # If loop finished without breaking due to time or min_deg fail, 
                # and we exited because no C4s were found, we would have returned.
                # If we broke due to min_deg, we continue to next attempt.
        
        return None

    # Try n=47 first
    # Target min degree: 47 - 39 = 8
    res = try_n(47, 8, 60)
    if res is not None:
        return {'n': 47, 'edges': res}

    # Try n=46
    # Target min degree: 46 - 39 = 7
    res = try_n(46, 7, 40)
    if res is not None:
        return {'n': 46, 'edges': res}

    # Try n=45
    # Target min degree: 45 - 39 = 6
    res = try_n(45, 6, 20)
    if res is not None:
        return {'n': 45, 'edges': res}

    # Fallback: return a valid small graph
    # Complete graph K_1 is C4-free, min deg 0. n-39 = 0. Valid.
    # But we want max n.
    # Let's just return n=39, empty edges? No, min deg must be >= 0.
    # K_39 is C4-free? No, K_5 has C4.
    # Actually, any bipartite graph is C4-free? No, K_{2,2} is C4.
    # A graph with max degree 1 is C4-free.
    # Let's just return n=1, no edges.
    return {'n': 1, 'edges': []}
