import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 20
    # adj[u] is an integer bitmask of neighbors of u
    # Initialize with the seed graph: 5 disjoint K4s
    adj = [0] * n
    for k in range(5):
        for a in range(4):
            for b in range(a + 1, 4):
                u = 4 * k + a
                v = 4 * k + b
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    best_edges = sum(bin(a).count('1') for a in adj) // 2
    best_adj = adj[:]
    
    def popcount(x):
        return bin(x).count('1')
    
    def can_add(adj_state, u, v):
        # Check if adding (u,v) keeps graph K2,3-free
        if (adj_state[u] >> v) & 1:
            return False
        
        # 1. Common neighbors of u and v must be <= 2
        common = adj_state[u] & adj_state[v]
        if popcount(common) > 2:
            return False
        
        # 2. For each w in common, the pair (u,w) and (v,w) gain 1 common neighbor
        #    So their current common neighbor counts must be <= 1
        #    If current count is >= 2, adding (u,v) makes it >= 3 -> Invalid
        while common:
            # Get lowest set bit
            lsb = common & (-common)
            w = lsb.bit_length() - 1
            common ^= lsb
            
            # Count common neighbors of u and w (excluding v, since v is not yet neighbor of u)
            # Note: v is in N(w) because w is in common. v is NOT in N(u) yet.
            # So N(u) & N(w) does not contain v.
            # After adding (u,v), v becomes in N(u). v is already in N(w).
            # So common neighbors of (u,w) increase by 1.
            # We need current count <= 1.
            common_uw = adj_state[u] & adj_state[w]
            if popcount(common_uw) > 1:
                return False
                
            # Similarly for v and w
            # u is in N(w). u is NOT in N(v) yet.
            # After adding, u becomes in N(v).
            # Common neighbors of (v,w) increase by 1.
            common_vw = adj_state[v] & adj_state[w]
            if popcount(common_vw) > 1:
                return False
                
        return True

    def count_edges(adj_state):
        return sum(bin(a).count('1') for a in adj_state) // 2

    # Local Search
    # We will try random restarts with random initial graphs if the seed doesn't yield good results
    # But first, try to improve the seed.
    
    # To allow exploration, we can also start from random sparse graphs
    # Let's define a function to run local search on a given state
    def local_search(adj_state, max_iters=5000):
        current_edges = count_edges(adj_state)
        # List of all possible edges
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_edges)
        
        stuck_count = 0
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            added_any = False
            # Try to add edges in random order
            for u, v in all_edges:
                if time.time() > deadline:
                    break
                if not ((adj_state[u] >> v) & 1):
                    if can_add(adj_state, u, v):
                        adj_state[u] |= (1 << v)
                        adj_state[v] |= (1 << u)
                        current_edges += 1
                        added_any = True
                        # Re-shuffle or just continue? Continuing is fine, but 
                        # re-shuffling helps break local minima caused by ordering bias.
                        # However, re-shuffling every addition is O(n^2). 
                        # Let's just break and restart the loop to re-shuffle.
                        break
            if added_any:
                stuck_count = 0
                random.shuffle(all_edges)
            else:
                stuck_count += 1
                if stuck_count > 10:
                    break
        
        return current_edges, adj_state

    # Run local search from seed
    current_edges, adj_current = local_search(adj[:], 2000)
    if current_edges > best_edges:
        best_edges = current_edges
        best_adj = adj_current[:]

    # Try some random starts
    # A random graph with p=0.3 might have ~57 edges, but many K2,3s.
    # We will generate a random graph and then greedily remove invalid edges? 
    # No, it's easier to build up.
    # Let's try random sparse starts.
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Random start: each edge present with prob 0.2
        adj_rand = [0] * n
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.2:
                    adj_rand[u] |= (1 << v)
                    adj_rand[v] |= (1 << u)
        
        # Ensure validity of start? 
        # A random sparse graph might already have K2,3s. 
        # We should prune it first or just rely on local search to fix it?
        # Local search as defined only ADDS edges. It cannot REMOVE.
        # So we must start with a valid graph.
        
        # Prune the random graph to be valid
        # Iteratively remove edges that create K2,3
        # This is expensive. 
        # Alternative: Start with empty graph and add random edges?
        
        # Let's stick to improving the seed and maybe doing "edge flips" (remove one, add many)
        # From the current best state, try removing a random edge and then greedy adding.
        
        if best_edges > 0:
            adj_trial = best_adj[:]
            # Pick a random edge to remove
            edges_to_remove = []
            for u in range(n):
                for v in range(u+1, n):
                    if (adj_trial[u] >> v) & 1:
                        edges_to_remove.append((u, v))
            if edges_to_remove:
                u, v = random.choice(edges_to_remove)
                adj_trial[u] &= ~(1 << v)
                adj_trial[v] &= ~(1 << u)
                
                # Greedy add from here
                new_edges, adj_new = local_search(adj_trial, 1000)
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj = adj_new[:]

    # Convert best_adj back to list of edges
    result_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                result_edges.append([u, v])
                
    return result_edges
