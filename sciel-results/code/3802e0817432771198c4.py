import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def has_k4(adj, n):
        """Check if graph has a K4. Return True if found."""
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    m = common
                    while m:
                        lsb = m & -m
                        w = lsb.bit_length() - 1
                        m ^= lsb
                        if common & adj[w]:
                            return True
        return False

    def has_independent_6(adj, n):
        """Check if graph has an independent set of size 6."""
        full_mask = (1 << n) - 1
        comp_adj = [0] * n
        for i in range(n):
            comp_adj[i] = full_mask ^ adj[i] ^ (1 << i)
        
        def dfs(verts, depth):
            if depth == 6:
                return True
            if not verts:
                return False
            v = verts
            while v:
                lsb = v & -v
                u = lsb.bit_length() - 1
                v ^= lsb
                new_verts = v & comp_adj[u]
                if dfs(new_verts, depth + 1):
                    return True
        return dfs(full_mask, 0)

    def is_valid(n, adj):
        if has_k4(adj, n):
            return False
        if has_independent_6(adj, n):
            return False
        return True

    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def count_edges_in_subset(mask, adj):
        """Count edges in the induced subgraph defined by mask"""
        total_deg = 0
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            m ^= lsb
            total_deg += (adj[u] & mask).bit_count()
        return total_deg // 2

    def count_indep_4_in_subset(mask, adj):
        """Count independent sets of size 4 in the induced subgraph defined by mask"""
        vertices = []
        m = mask
        while m:
            lsb = m & -m
            vertices.append(lsb.bit_length() - 1)
            m ^= lsb
        
        count = 0
        n_v = len(vertices)
        if n_v < 4:
            return 0
        
        for i in range(n_v):
            for j in range(i+1, n_v):
                if (adj[vertices[i]] >> vertices[j]) & 1:
                    continue
                for k in range(j+1, n_v):
                    if (adj[vertices[i]] >> vertices[k]) & 1 or (adj[vertices[j]] >> vertices[k]) & 1:
                        continue
                    for l in range(k+1, n_v):
                        if (adj[vertices[i]] >> vertices[l]) & 1 or (adj[vertices[j]] >> vertices[l]) & 1 or (adj[vertices[k]] >> vertices[l]) & 1:
                            continue
                        count += 1
        return count

    def delta_violations(u, v, adj, n, is_edge):
        """
        Returns the change in total violations if edge (u,v) is flipped.
        is_edge: True if (u,v) is currently an edge.
        """
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        
        # Independent set candidates: vertices not adjacent to u or v, and not u,v
        non_common = ((1 << n) - 1) & ~(adj[u] | adj[v]) & ~(1 << u) & ~(1 << v)
        
        e_count = count_edges_in_subset(common, adj)
        i4_count = count_indep_4_in_subset(non_common, adj)
        
        if is_edge:
            # Removing edge: 
            # Destroys K4s: -e_count
            # Creates I6s: +i4_count (for each I4 in non_common, adding u,v makes I6)
            return -e_count + i4_count
        else:
            # Adding edge:
            # Creates K4s: +e_count
            # Destroys I6s: -i4_count
            return e_count - i4_count

    def local_search(n, time_limit, seed=None):
        """Local search to find a valid graph on n vertices."""
        if seed is not None:
            random.seed(seed)
        
        start = time.time()
        deadline_ls = start + time_limit
        
        # Start with random graph
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        best_adj = None
        best_score = float('inf')
        
        # We want to minimize total violations.
        # Total violations = #K4 + #I6.
        # We can estimate this or just use the greedy improvement loop.
        
        improved = True
        while improved and time.time() < deadline_ls:
            improved = False
            best_delta = 0
            best_u = -1
            best_v = -1
            
            # Evaluate all edges
            for u in range(n):
                for v in range(u+1, n):
                    is_edge = bool((adj[u] >> v) & 1)
                    d = delta_violations(u, v, adj, n, is_edge)
                    if d < best_delta:
                        best_delta = d
                        best_u = u
                        best_v = v
                        improved = True
                        
            if best_u != -1:
                # Flip edge
                if (adj[best_u] >> best_v) & 1:
                    adj[best_u] &= ~(1 << best_v)
                    adj[best_v] &= ~(1 << best_u)
                else:
                    adj[best_u] |= (1 << best_v)
                    adj[best_v] |= (1 << best_u)
                    
                # Check if we reached a valid state early
                if not has_k4(adj, n) and not has_independent_6(adj, n):
                    return adj
        
        # If we exit the loop, check if it's valid
        if not has_k4(adj, n) and not has_independent_6(adj, n):
            return adj
        
        return None

    def paley_graph(p):
        """Generate Paley graph for prime p."""
        squares = {x * x % p for x in range(1, p)}
        adj = [0] * p
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    best_graph = None
    best_n = 0
    
    # 1. Try known constructions first (fast)
    candidates = []
    for p in [37, 29, 23, 19, 17]:
        if p <= 40:
            candidates.append((p, paley_graph(p)))
    
    # Also try a few other small primes or just stick to Paley
    # Clebsch graph is 16 vertices, Paley 17 is better.
    
    for n_cand, adj_cand in candidates:
        if time.time() > deadline:
            break
        if is_valid(n_cand, adj_cand):
            if n_cand > best_n:
                best_n = n_cand
                best_graph = adj_cand[:]

    # 2. Local search for larger n
    # Start from 35 down to best_n + 1
    # If we found 37, we are done (R(4,6) > 37 is a strong result, likely max for this constraint set? No, R(4,6) is unknown but >35. 37 is a strong lower bound.)
    # Actually, the task says "Exoo's 35-vertex graph is the published record; 36 vertices would be a new lower bound."
    # So finding 36 or 37 is a win.
    
    # Prioritize higher n.
    for target_n in range(35, best_n, -1):
        if time.time() > deadline:
            break
        
        # Try multiple restarts for each n
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        if remaining < 1:
            break
            
        # Give each target a few seconds
        # If target_n is high, it's harder.
        time_per_restart = min(5.0, remaining / 5.0)
        
        found = False
        for r in range(10): # Max 10 restarts per n
            if time.time() > deadline:
                break
            
            adj = local_search(target_n, time_per_restart, seed=42 + r)
            if adj is not None:
                if is_valid(target_n, adj):
                    if target_n > best_n:
                        best_n = target_n
                        best_graph = adj
                    found = True
                    break
        
        if found:
            # If we found a valid graph for this n, do we try higher?
            # The loop goes downwards, so if we start at 35 and find it, we return.
            # If we start at 35 and don't find it, we try 34, etc.
            # But we want the MAX n.
            # So if we find one at 35, we stop.
            # If we find one at 34, we stop (since 35 failed).
            # This logic is correct for "Maximize n".
            break

    # Fallback: if best_graph is None (shouldn't happen since Paley 17 is valid)
    if best_graph is None:
        # Fallback to Paley 17
        best_graph = paley_graph(17)
        best_n = 17

    return {'n': best_n, 'edges': get_edges(best_graph, best_n)}

# Ensure random is available
random.seed(42)
