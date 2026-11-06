import time
import random

def construct():
    n = 20
    start_time = time.time()
    # Use a conservative deadline to ensure we return before the 120s kill
    deadline = start_time + 100.0 

    def is_valid_after_add(adj, u, v):
        """
        Checks if adding edge (u,v) to adj creates a K3,3.
        Optimized: only check triples where the new edge creates a new common neighbor.
        When we add (u,v):
        - For triple {u, a, b} (a,b ≠ u): common neighbors gain v iff v is neighbor of both a and b
        - For triple {v, a, b} (a,b ≠ v, a,b ≠ u): common neighbors gain u iff u is neighbor of both a and b
        - For triple {u, a, v}: common neighbors unchanged (v is in triple, u is in triple)
        - For triple {v, a, u}: same
        So we only need to check:
        - Triples {u, a, b} where a,b are both neighbors of v (and a,b ≠ u)
        - Triples {v, a, b} where a,b are both neighbors of u (and a,b ≠ v, a,b ≠ u)
        """
        # Triples {u, a, b} where a,b are neighbors of v
        # v's neighbors (excluding u since we just added it, but adj[v] now includes u)
        # We need a,b ≠ u, a,b distinct, both neighbors of v
        v_neighbors = adj[v] & ~(1 << u) & ~(1 << v)  # neighbors of v, excluding u and v
        # Iterate over pairs from v_neighbors
        vn_list = []
        for a in range(n):
            if v_neighbors & (1 << a):
                vn_list.append(a)
        
        for i in range(len(vn_list)):
            a = vn_list[i]
            for j in range(i + 1, len(vn_list)):
                b = vn_list[j]
                # Check triple {u, a, b}: common neighbors
                common = adj[u] & adj[a] & adj[b]
                if common.bit_count() >= 3:
                    return False
        
        # Triples {v, a, b} where a,b are neighbors of u
        u_neighbors = adj[u] & ~(1 << v) & ~(1 << u)  # neighbors of u, excluding v and u
        un_list = []
        for a in range(n):
            if u_neighbors & (1 << a):
                un_list.append(a)
        
        for i in range(len(un_list)):
            a = un_list[i]
            for j in range(i + 1, len(un_list)):
                b = un_list[j]
                # Check triple {v, a, b}: common neighbors
                common = adj[v] & adj[a] & adj[b]
                if common.bit_count() >= 3:
                    return False
        
        return True

    def count_edges(adj):
        return sum(a.bit_count() for a in adj) // 2

    def greedy_fill(adj):
        """
        Greedily adds edges in random order.
        Returns the number of edges in the final adj.
        """
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(pairs)
        for u, v in pairs:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            
            # Quick check: if u and v already have 2 common neighbors, adding edge
            # could create issues, but let's just try
            # Try adding edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if is_valid_after_add(adj, u, v):
                pass # Keep edge
            else:
                # Rollback
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        return count_edges(adj)

    best_adj = [0] * n
    best_count = 0
    current_adj = [0] * n
    
    # Initial greedy fill from empty graph
    current_count = greedy_fill(current_adj)
    if current_count > best_count:
        best_count = current_count
        best_adj = current_adj[:]

    # Local search / Restart strategy
    # Since greedy_fill is fast, we can do many restarts.
    # We also try a "kick" strategy: remove a few random edges, then re-greedy.
    
    iterations = 0
    no_improve_count = 0
    while time.time() < deadline:
        iterations += 1
        current_adj = best_adj[:]
        
        # Adaptive removal: start with fewer, increase if stuck
        if no_improve_count < 10:
            num_removals = random.randint(1, 3)
        elif no_improve_count < 30:
            num_removals = random.randint(2, 6)
        else:
            num_removals = random.randint(3, 10)
        
        edges_to_remove = []
        # Get current edges
        for u in range(n):
            for v in range(u + 1, n):
                if current_adj[u] & (1 << v):
                    edges_to_remove.append((u, v))
        
        if not edges_to_remove:
            continue
            
        random.shuffle(edges_to_remove)
        for u, v in edges_to_remove[:num_removals]:
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
        
        current_count = greedy_fill(current_adj)
        
        if current_count > best_count:
            best_count = current_count
            best_adj = current_adj[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # If stuck for too long, do a full restart
        if no_improve_count >= 50:
            current_adj = [0] * n
            current_count = greedy_fill(current_adj)
            if current_count > best_count:
                best_count = current_count
                best_adj = current_adj[:]
            no_improve_count = 0

    # Final polish: try adding any missing edge that doesn't create K3,3
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u in range(n):
            for v in range(u + 1, n):
                if time.time() > deadline:
                    break
                if best_adj[u] & (1 << v):
                    continue
                # Try adding
                best_adj[u] |= (1 << v)
                best_adj[v] |= (1 << u)
                if is_valid_after_add(best_adj, u, v):
                    improved = True
                else:
                    best_adj[u] &= ~(1 << v)
                    best_adj[v] &= ~(1 << u)
            if time.time() > deadline:
                break

    # Convert best_adj to list of edges
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                result.append([u, v])
    return result
