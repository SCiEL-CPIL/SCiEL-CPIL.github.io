import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 16
    target_edges = 65
    
    # Initialize with K_{8,8} + 1 edge
    # Vertices 0-7 in A, 8-15 in B
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    edges.append([0, 1])
    
    best_edges = edges[:]
    
    def build_state(edge_list):
        """Build adjacency bitmasks and degrees."""
        adj = [0] * n
        deg = [0] * n
        for u, v in edge_list:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg

    def compute_score(adj, deg):
        """Compute max triangle degree sum."""
        max_sum = 0
        # Iterate over all triples
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    continue
                for k in range(j + 1, n):
                    if (adj[i] & (1 << k)) and (adj[j] & (1 << k)):
                        s = deg[i] + deg[j] + deg[k]
                        if s > max_sum:
                            max_sum = s
        return max_sum

    def get_worst_triangle(adj, deg):
        """Return the triangle with the maximum degree sum."""
        max_sum = 0
        worst = None
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    continue
                for k in range(j + 1, n):
                    if (adj[i] & (1 << k)) and (adj[j] & (1 << k)):
                        s = deg[i] + deg[j] + deg[k]
                        if s > max_sum:
                            max_sum = s
                            worst = (i, j, k)
        return worst, max_sum

    def is_valid_edge_list(edge_list):
        if len(edge_list) != target_edges:
            return False
        seen = set()
        for u, v in edge_list:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in seen:
                return False
            seen.add(key)
        return True

    # Local Search
    def local_search(edge_list, max_iters=20000):
        current_edges = edge_list[:]
        adj, deg = build_state(current_edges)
        current_score = compute_score(adj, deg)
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() > deadline - 5.0:
                break
            improved = False
            iters += 1
            
            worst_tri, _ = get_worst_triangle(adj, deg)
            if worst_tri is None:
                break
                
            # Try to improve by modifying edges incident to the worst triangle
            i, j, k = worst_tri
            candidate_edges_to_remove = [(i, j), (j, k), (i, k)]
            
            # Also try random swaps if targeted fails or as a mix
            # Here we focus on targeted first
            for rem_u, rem_v in candidate_edges_to_remove:
                if time.time() > deadline - 5.0:
                    break
                
                # Remove edge (rem_u, rem_v)
                new_edges = []
                for e in current_edges:
                    if (e[0] == rem_u and e[1] == rem_v) or (e[0] == rem_v and e[1] == rem_u):
                        continue
                    new_edges.append(e)
                
                # Add a random non-edge
                # Sample a few candidates
                for _ in range(10):
                    if time.time() > deadline - 5.0:
                        break
                    u_add = random.randint(0, n - 1)
                    v_add = random.randint(0, n - 1)
                    if u_add == v_add:
                        continue
                    if u_add > v_add:
                        u_add, v_add = v_add, u_add
                    
                    # Check if edge exists in new_edges (approximate via adj of current, but we removed one)
                    # Safer: check against new_edges set
                    edge_set = set()
                    for a, b in new_edges:
                        if a > b: a, b = b, a
                        edge_set.add((a, b))
                    
                    if (u_add, v_add) in edge_set:
                        continue
                    
                    new_edges.append([u_add, v_add])
                    
                    # Check validity (simple graph)
                    if not is_valid_edge_list(new_edges):
                        continue
                        
                    new_adj, new_deg = build_state(new_edges)
                    new_score = compute_score(new_adj, new_deg)
                    
                    if new_score < current_score:
                        current_edges = new_edges
                        current_score = new_score
                        adj, deg = new_adj, new_deg
                        improved = True
                        break
                if improved:
                    break
            
            # If no improvement from targeted, try a few random swaps
            if not improved:
                for _ in range(5):
                    if time.time() > deadline - 5.0:
                        break
                    # Remove random edge
                    idx = random.randint(0, len(current_edges) - 1)
                    rem_edge = current_edges[idx]
                    new_edges = current_edges[:idx] + current_edges[idx+1:]
                    
                    # Add random non-edge
                    for _ in range(5):
                        u_add = random.randint(0, n - 1)
                        v_add = random.randint(0, n - 1)
                        if u_add == v_add:
                            continue
                        if u_add > v_add:
                            u_add, v_add = v_add, u_add
                        
                        # Check existence
                        exists = False
                        for a, b in new_edges:
                            if (a == u_add and b == v_add) or (a == v_add and b == u_add):
                                exists = True
                                break
                        if exists:
                            continue
                        
                        new_edges.append([u_add, v_add])
                        
                        if not is_valid_edge_list(new_edges):
                            continue

                        new_adj, new_deg = build_state(new_edges)
                        new_score = compute_score(new_adj, new_deg)
                        
                        if new_score < current_score:
                            current_edges = new_edges
                            current_score = new_score
                            adj, deg = new_adj, new_deg
                            improved = True
                            break
                    if improved:
                        break

        return current_edges, current_score

    best_score = compute_score(*build_state(best_edges))
    
    # Multi-start
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb best solution
        perturbed = best_edges[:]
        num_swaps = random.randint(2, 5)
        for _ in range(num_swaps):
            idx = random.randint(0, len(perturbed) - 1)
            u1, v1 = perturbed[idx]
            
            # Find a non-edge to replace
            while True:
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u2 == v2:
                    continue
                if u2 > v2:
                    u2, v2 = v2, u2
                
                exists = False
                for a, b in perturbed:
                    if (a == u2 and b == v2) or (a == v2 and b == u2):
                        exists = True
                        break
                if not exists:
                    break
            
            perturbed[idx] = [u2, v2]
        
        if not is_valid_edge_list(perturbed):
            continue
            
        local_edges, local_score = local_search(perturbed, max_iters=5000)
        
        if local_score < best_score:
            best_score = local_score
            best_edges = local_edges[:]
            
        # Also check if perturbation itself is better (unlikely but possible)
        pert_score = compute_score(*build_state(perturbed))
        if pert_score < best_score:
            best_score = pert_score
            best_edges = perturbed[:]

    return {'n': n, 'edges': best_edges}
