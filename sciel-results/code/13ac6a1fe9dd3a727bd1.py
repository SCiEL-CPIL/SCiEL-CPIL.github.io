import time
import random

N = 20
DEADLINE_FRACTION = 0.95

def construct():
    start_time = time.time()
    deadline = start_time + 120 * DEADLINE_FRACTION

    # Precompute all possible edges
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    def can_add(adj, u, v):
        """
        Check if adding edge (u, v) violates the K3,3-free condition.
        A K3,3 is formed if there exist 3 vertices with 3 common neighbors.
        Adding (u, v) only affects triples containing u or v as one of the vertices 
        for which v or u becomes a new common neighbor.
        
        Specifically:
        1. If a, b are neighbors of v, then v is a common neighbor of {u, a, b} 
           if u is also a neighbor of a and b. 
           We need |N(u) \cap N(a) \cap N(b)| <= 1 (before adding u-v, v is not in N(u)).
           After adding, v is in N(u), so total common neighbors is count + 1.
           We need count + 1 <= 2 => count <= 1.
        2. Symmetrically for u.
        """
        # Check if edge already exists
        if adj[u] & (1 << v):
            return False
        
        # Case 1: v becomes a common neighbor of {u, a, b}
        # This requires a, b to be neighbors of v.
        # And we check common neighbors of u, a, b.
        for a in range(N):
            if a == u or a == v:
                continue
            if not (adj[a] & (1 << v)):
                continue
            for b in range(a + 1, N):
                if b == u or b == v:
                    continue
                if not (adj[b] & (1 << v)):
                    continue
                # Check common neighbors of u, a, b
                common = adj[u] & adj[a] & adj[b]
                if common.bit_count() > 1:
                    return False
        
        # Case 2: u becomes a common neighbor of {v, a, b}
        # This requires a, b to be neighbors of u.
        # And we check common neighbors of v, a, b.
        for a in range(N):
            if a == u or a == v:
                continue
            if not (adj[a] & (1 << u)):
                continue
            for b in range(a + 1, N):
                if b == u or b == v:
                    continue
                if not (adj[b] & (1 << u)):
                    continue
                # Check common neighbors of v, a, b
                common = adj[v] & adj[a] & adj[b]
                if common.bit_count() > 1:
                    return False
        
        return True

    def get_edge_list(adj):
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i] & (1 << j):
                    edges.append((i, j))
        return edges

    def greedy_fill(adj, time_limit=None):
        """
        Greedily add as many missing edges as possible.
        Returns the updated adjacency list.
        """
        missing = [(i, j) for i in range(N) for j in range(i + 1, N) if not (adj[i] & (1 << j))]
        random.shuffle(missing)
        
        for u, v in missing:
            if time_limit and time.time() > time_limit:
                break
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search(adj, max_time=1.0):
        """
        Local search with kick mechanism.
        """
        ls_start = time.time()
        current_edges = sum(x.bit_count() for x in adj) // 2
        best_adj = adj[:]
        best_edges = current_edges
        
        # Initial greedy fill
        adj = greedy_fill(adj, deadline)
        current_edges = sum(x.bit_count() for x in adj) // 2
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]
            
        missing = [(i, j) for i in range(N) for j in range(i + 1, N) if not (adj[i] & (1 << j))]
        
        no_improve_count = 0
        max_no_improve = 100
        
        while time.time() - ls_start < max_time and no_improve_count < max_no_improve:
            if time.time() > deadline - 2:
                break
                
            if not missing:
                break
                
            # Try to add a random missing edge
            idx = random.randint(0, len(missing) - 1)
            u, v = missing[idx]
            
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                missing.pop(idx)
                no_improve_count = 0
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
            else:
                # Kick: remove 1-3 random edges, then greedy fill
                kick_size = random.choice([1, 1, 1, 2, 2, 3])
                existing = get_edge_list(adj)
                if not existing:
                    break
                    
                removed = []
                for _ in range(min(kick_size, len(existing))):
                    if not existing:
                        break
                    idx2 = random.randint(0, len(existing) - 1)
                    a, b = existing[idx2]
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    current_edges -= 1
                    removed.append((a, b))
                    existing.pop(idx2)
                    
                # Greedy fill from current state
                adj = greedy_fill(adj, deadline)
                current_edges = sum(x.bit_count() for x in adj) // 2
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                # Rebuild missing list
                missing = [(i, j) for i in range(N) for j in range(i + 1, N) if not (adj[i] & (1 << j))]
                
        return best_adj, best_edges

    best_adj = None
    best_count = 0

    # Multiple random restarts to find a good basin
    # Each restart gets a fraction of the time
    # We will use a loop that runs until deadline
    
    # Initial random graph to break symmetry
    adj = [0] * N
    for u in range(N):
        for v in range(u + 1, N):
            if random.random() < 0.3:
                if can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    
    # Greedy fill
    adj = greedy_fill(adj, deadline)
    count = sum(x.bit_count() for x in adj) // 2
    if count > best_count:
        best_count = count
        best_adj = adj[:]
        
    # Local search
    adj_ls, count_ls = local_search(best_adj, max_time=5.0)
    if count_ls > best_count:
        best_count = count_ls
        best_adj = adj_ls[:]
        
    # Main loop: Kick and search
    while time.time() < deadline - 1:
        # Kick the best solution
        adj_k = best_adj[:]
        kick_size = random.randint(2, 5)
        existing = get_edge_list(adj_k)
        if not existing:
            break
            
        random.shuffle(existing)
        for i in range(min(kick_size, len(existing))):
            u, v = existing[i]
            adj_k[u] &= ~(1 << v)
            adj_k[v] &= ~(1 << u)
            
        # Greedy fill
        adj_k = greedy_fill(adj_k, deadline)
        
        # Local search on the kicked solution
        adj_ls, count_ls = local_search(adj_k, max_time=2.0)
        
        if count_ls > best_count:
            best_count = count_ls
            best_adj = adj_ls[:]
            
    # Final polish: try to add any remaining missing edges
    if best_adj:
        best_adj = greedy_fill(best_adj, deadline)
        final_count = sum(x.bit_count() for x in best_adj) // 2
        if final_count > best_count:
            best_count = final_count
            # best_adj is already updated in place by greedy_fill if we passed it directly? 
            # No, greedy_fill returns adj. But I modified adj in place inside greedy_fill? 
            # Yes, `adj[u] |= ...` modifies the list passed in.
            # So best_adj is updated.

    if best_adj is None:
        return []

    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
