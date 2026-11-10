import itertools
import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n = 18
    all_vertices_mask = (1 << n) - 1
    
    def is_k4_free(adj):
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    continue
                for k in range(j + 1, n):
                    if not (adj[i] & (1 << k)) or not (adj[j] & (1 << k)):
                        continue
                    for l in range(k + 1, n):
                        if (adj[i] & (1 << l)) and (adj[j] & (1 << l)) and (adj[k] & (1 << l)):
                            return False
        return True

    def get_triangles(adj):
        triangles = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    continue
                common = adj[i] & adj[j] & ~(1 << i) & ~(1 << j)
                k_mask = common & (sum(1 << x for x in range(j + 1, n)))
                while k_mask:
                    lsb = k_mask & -k_mask
                    k = lsb.bit_length() - 1
                    triangles.append((i, j, k))
                    k_mask ^= lsb
        return triangles

    def max_triangle_free_size(adj, triangles):
        """Branch and bound to find largest triangle-free induced subgraph."""
        if not triangles:
            return n
        
        tri_masks = []
        for (i, j, k) in triangles:
            tri_masks.append((1 << i) | (1 << j) | (1 << k))
        
        best = [0]
        deadline_local = deadline - 5.0
        
        def branch(remaining_mask):
            if time.time() > deadline_local:
                return
            cnt = bin(remaining_mask).count('1')
            if cnt <= best[0]:
                return
            # Check if triangle-free
            for tm in tri_masks:
                if (remaining_mask & tm) == tm:
                    # Find vertices of this triangle in remaining
                    verts = []
                    for v in range(n):
                        if tm & (1 << v):
                            verts.append(v)
                    # Branch on removing each vertex
                    for v in verts:
                        new_mask = remaining_mask & ~(1 << v)
                        branch(new_mask)
                    return
            # No triangle found, this is a valid triangle-free set
            if cnt > best[0]:
                best[0] = cnt
        
        branch(all_vertices_mask)
        return best[0]

    def remove_k4s(adj):
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] & (1 << j)):
                        continue
                    for k in range(j + 1, n):
                        if not (adj[i] & (1 << k)) or not (adj[j] & (1 << k)):
                            continue
                        for l in range(k + 1, n):
                            if (adj[i] & (1 << l)) and (adj[j] & (1 << l)) and (adj[k] & (1 << l)):
                                adj[i] &= ~(1 << l)
                                adj[l] &= ~(1 << i)
                                changed = True
                                break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
        return adj

    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def local_search(adj, max_iters=300):
        """Edge flip local search to reduce max triangle-free size."""
        current_score = max_triangle_free_size(adj, get_triangles(adj))
        if not is_k4_free(adj):
            return adj, current_score
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 15:
            improved = False
            iters += 1
            for i in range(n):
                for j in range(i + 1, n):
                    has_edge = bool(adj[i] & (1 << j))
                    new_adj = adj[:]
                    if has_edge:
                        new_adj[i] &= ~(1 << j)
                        new_adj[j] &= ~(1 << i)
                    else:
                        new_adj[i] |= (1 << j)
                        new_adj[j] |= (1 << i)
                    
                    if not is_k4_free(new_adj):
                        continue
                    
                    s = max_triangle_free_size(new_adj, get_triangles(new_adj))
                    if s < current_score:
                        current_score = s
                        adj = new_adj
                        improved = True
                        break
                if improved:
                    break
        return adj, current_score

    def simulated_annealing(adj, max_iters=5000, t_start=1.0, t_end=0.01):
        """Simulated annealing with incremental evaluation."""
        current_adj = adj[:]
        current_score = max_triangle_free_size(current_adj, get_triangles(current_adj))
        best_adj = current_adj[:]
        best_score = current_score
        
        if not is_k4_free(current_adj):
            return best_adj, best_score
        
        # Precompute all edge pairs
        edge_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        t = t_start
        cooling = (t_end / t_start) ** (1.0 / max(1, max_iters))
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Pick a random edge to flip
            i, j = random.choice(edge_pairs)
            new_adj = current_adj[:]
            has_edge = bool(current_adj[i] & (1 << j))
            if has_edge:
                new_adj[i] &= ~(1 << j)
                new_adj[j] &= ~(1 << i)
            else:
                new_adj[i] |= (1 << j)
                new_adj[j] |= (1 << i)
            
            if not is_k4_free(new_adj):
                continue
            
            new_score = max_triangle_free_size(new_adj, get_triangles(new_adj))
            
            delta = new_score - current_score
            if delta <= 0 or random.random() < math.exp(-delta / max(t, 1e-10)):
                current_adj = new_adj
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = new_adj[:]
            
            t *= cooling
        
        return best_adj, best_score

    best_score = 12
    best_edges = None
    
    # Seed: K6,6,6 complete tripartite
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    seed_adj = [0] * n
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    seed_adj[a] |= (1 << b)
                    seed_adj[b] |= (1 << a)
    best_edges = adj_to_edges(seed_adj)
    best_score = 12
    
    densities = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    # Phase 1: Random restarts with local search
    while time.time() < deadline - 40:
        p = random.choice(densities)
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        adj = remove_k4s(adj)
        
        if not is_k4_free(adj):
            continue
        
        s = max_triangle_free_size(adj, get_triangles(adj))
        
        if s < best_score:
            best_score = s
            best_edges = adj_to_edges(adj)
        
        # Local search on promising candidates
        if s <= 11 and time.time() < deadline - 50:
            refined_adj, refined_score = local_search(adj, max_iters=100)
            if refined_score < best_score:
                best_score = refined_score
                best_edges = adj_to_edges(refined_adj)
        
        if best_score <= 10:
            break
    
    # Phase 2: Simulated annealing from best and random starts
    if time.time() < deadline - 20:
        # SA from best
        adj = [0] * n
        for e in best_edges:
            u, v = e
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        sa_adj, sa_score = simulated_annealing(adj, max_iters=3000)
        if sa_score < best_score:
            best_score = sa_score
            best_edges = adj_to_edges(sa_adj)
    
    # More SA runs from different starts
    while time.time() < deadline - 15:
        p = random.choice(densities)
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        adj = remove_k4s(adj)
        
        if not is_k4_free(adj):
            continue
        
        sa_adj, sa_score = simulated_annealing(adj, max_iters=2000)
        if sa_score < best_score:
            best_score = sa_score
            best_edges = adj_to_edges(sa_adj)
        
        if best_score <= 10:
            break
    
    # Final polish on best
    if best_edges is not None and time.time() < deadline - 5:
        adj = [0] * n
        for e in best_edges:
            u, v = e
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        refined_adj, refined_score = local_search(adj, max_iters=200)
        if refined_score < best_score:
            best_score = refined_score
            best_edges = adj_to_edges(refined_adj)
    
    return best_edges

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
