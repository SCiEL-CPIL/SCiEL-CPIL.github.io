import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 110
    random.seed(42)
    
    def find_k4(adj, n):
        """Find one K4 using bitsets. adj is list of ints."""
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    # Common neighbors of i and j
                    common = adj[i] & adj[j]
                    # Remove i and j from common
                    common &= ~(1 << i)
                    common &= ~(1 << j)
                    # Check for edge in common
                    # Iterate through set bits in common
                    c = common
                    while c:
                        # Get lowest set bit
                        lsb = c & -c
                        k = lsb.bit_length() - 1
                        c ^= lsb
                        # Check if k is connected to any other in common
                        # We need to check if there is an edge between k and some m in common (m > k to avoid duplicates? No, just any)
                        # Actually, if we find any edge in the induced subgraph of common, we have a K4.
                        # Let's just check if adj[k] & common has any bits set (excluding k itself, but k is in common so adj[k] doesn't have k).
                        # Wait, adj[k] & common will include k? No, simple graph, no self loops.
                        # But common contains k. adj[k] does not contain k.
                        # So if (adj[k] & common) != 0, there is an edge.
                        if (adj[k] & common):
                            # Find the specific vertex m
                            m_bit = adj[k] & common
                            m = (m_bit & -m_bit).bit_length() - 1
                            return (i, j, k, m)
        return None

    def find_indep6(adj, n):
        """Find one independent 6-set using bitsets and backtracking."""
        # Non-adjacency mask for vertex v: all vertices not connected to v, excluding v
        # non_adj[v] = ~adj[v] & ((1 << n) - 1) & ~(1 << v)
        non_adj = [0] * n
        full_mask = (1 << n) - 1
        for v in range(n):
            non_adj[v] = (~adj[v]) & full_mask & ~(1 << v)
            
        def backtrack(remaining_mask, chosen):
            if len(chosen) == 6:
                return tuple(chosen)
            # Pruning: if popcount(remaining_mask) < 6 - len(chosen), return None
            if bin(remaining_mask).count('1') < 6 - len(chosen):
                return None
            
            # Get list of vertices in remaining_mask
            # To make it deterministic and fast, we can just iterate
            r = remaining_mask
            while r:
                lsb = r & -r
                v = lsb.bit_length() - 1
                r ^= lsb
                
                # Try including v
                new_chosen = chosen + [v]
                # New remaining: vertices in remaining_mask that are non-adjacent to v
                # They must be in remaining_mask AND in non_adj[v]
                new_remaining = remaining_mask & non_adj[v]
                # Remove v from new_remaining (it's not in non_adj[v] anyway, but safe)
                new_remaining &= ~(1 << v)
                
                result = backtrack(new_remaining, new_chosen)
                if result is not None:
                    return result
            return None
        
        return backtrack((1 << n) - 1, [])

    def is_valid(adj, n):
        return find_k4(adj, n) is None and find_indep6(adj, n) is None

    def make_adj_bitset(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def extract_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def random_graph_bitset(n, p):
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    best_result = None
    
    # 1. Try the product construction from the seed (fast)
    n35 = 35
    D5 = {1, 4}
    D7 = {1, 2, 4}
    adj35 = [0] * n35
    for i in range(n35):
        a1, b1 = divmod(i, 7)
        for j in range(i+1, n35):
            a2, b2 = divmod(j, 7)
            da = (a2 - a1) % 5
            db = (b2 - b1) % 7
            if (da in D5 and db in D7) or (da not in D5 and db not in D7):
                adj35[i] |= (1 << j)
                adj35[j] |= (1 << i)
                
    if is_valid(adj35, n35):
        best_result = {'n': n35, 'edges': extract_edges(adj35, n35)}
        return best_result

    # 2. Repair the 35-vertex graph using bitset local search
    adj_repair = adj35[:]
    
    for step in range(5000):
        if time.time() > deadline - 5:
            break
        
        k4 = find_k4(adj_repair, n35)
        if k4:
            i, j, k, l = k4
            # Pick a random edge from the K4 to remove
            pair_options = [(i, j), (i, k), (i, l), (j, k), (j, l), (k, l)]
            u, v = random.choice(pair_options)
            adj_repair[u] &= ~(1 << v)
            adj_repair[v] &= ~(1 << u)
            continue
        
        indep6 = find_indep6(adj_repair, n35)
        if indep6:
            u, v = random.sample(indep6, 2)
            adj_repair[u] |= (1 << v)
            adj_repair[v] |= (1 << u)
            continue
        
        # Valid
        if is_valid(adj_repair, n35):
            best_result = {'n': n35, 'edges': extract_edges(adj_repair, n35)}
            return best_result

    # 3. If repair failed, try random restarts for n=35
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        adj_rand = random_graph_bitset(35, 0.45)
        for step in range(1000):
            if time.time() > deadline - 5:
                break
            k4 = find_k4(adj_rand, 35)
            if k4:
                i, j, k, l = k4
                pair_options = [(i, j), (i, k), (i, l), (j, k), (j, l), (k, l)]
                u, v = random.choice(pair_options)
                adj_rand[u] &= ~(1 << v)
                adj_rand[v] &= ~(1 << u)
                continue
            indep6 = find_indep6(adj_rand, 35)
            if indep6:
                u, v = random.sample(indep6, 2)
                adj_rand[u] |= (1 << v)
                adj_rand[v] |= (1 << u)
                continue
            if is_valid(adj_rand, 35):
                best_result = {'n': 35, 'edges': extract_edges(adj_rand, 35)}
                return best_result
            break # No violations found in this step? Wait, if no k4 and no indep6, it's valid.
            # The loop above breaks if both are None.
            # If it breaks, we should check validity again to be sure.
            if is_valid(adj_rand, 35):
                best_result = {'n': 35, 'edges': extract_edges(adj_rand, 35)}
                return best_result

    # 4. Fallback to smaller n if 35 fails
    for target_n in range(34, 15, -1):
        if time.time() > deadline - 5:
            break
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            adj_rand = random_graph_bitset(target_n, 0.45)
            for step in range(500):
                if time.time() > deadline - 5:
                    break
                k4 = find_k4(adj_rand, target_n)
                if k4:
                    i, j, k, l = k4
                    pair_options = [(i, j), (i, k), (i, l), (j, k), (j, l), (k, l)]
                    u, v = random.choice(pair_options)
                    adj_rand[u] &= ~(1 << v)
                    adj_rand[v] &= ~(1 << u)
                    continue
                indep6 = find_indep6(adj_rand, target_n)
                if indep6:
                    u, v = random.sample(indep6, 2)
                    adj_rand[u] |= (1 << v)
                    adj_rand[v] |= (1 << u)
                    continue
                if is_valid(adj_rand, target_n):
                    if best_result is None or target_n > best_result['n']:
                        best_result = {'n': target_n, 'edges': extract_edges(adj_rand, target_n)}
                    break
            if best_result is not None and best_result['n'] == target_n:
                break
        if best_result is not None and best_result['n'] == target_n:
            break

    if best_result is None:
        # Return a valid small graph, e.g., Paley 17
        squares_17 = {x*x % 17 for x in range(1, 17)}
        edges_17 = [[i, j] for i in range(17) for j in range(i+1, 17) if (j-i) % 17 in squares_17]
        best_result = {'n': 17, 'edges': edges_17}

    return best_result
