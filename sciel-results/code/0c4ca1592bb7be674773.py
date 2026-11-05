"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    n = 31
    target_edges = 91

    def build_polarity_graph():
        """Construct a polarity graph on PG(2,5) - C4-free with 90 edges."""
        q = 5
        points = []
        seen = set()
        for x0 in range(q):
            for x1 in range(q):
                for x2 in range(q):
                    if x0 == 0 and x1 == 0 and x2 == 0:
                        continue
                    if x0 != 0:
                        inv = pow(x0, q-2, q)
                        canon = (1, x1 * inv % q, x2 * inv % q)
                    elif x1 != 0:
                        inv = pow(x1, q-2, q)
                        canon = (0, 1, x2 * inv % q)
                    else:
                        canon = (0, 0, 1)
                    if canon not in seen:
                        seen.add(canon)
                        points.append(canon)
        
        assert len(points) == 31, f"Expected 31 points, got {len(points)}"
        
        def are_conjugate(p, r):
            return (p[0]*r[2] + p[1]*r[1] + p[2]*r[0]) % q == 0
        
        adj = [set() for _ in range(31)]
        edges = []
        for i in range(31):
            for j in range(i+1, 31):
                if are_conjugate(points[i], points[j]):
                    adj[i].add(j)
                    adj[j].add(i)
                    edges.append([i, j])
        
        return edges, adj

    def count_c4_from_bitmasks(adj_bm):
        """Count 4-cycles using bitmask adjacency. adj_bm[i] is a bitmask of neighbors of i."""
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_bm[u] >> v) & 1:
                    common = (adj_bm[u] & adj_bm[v]).bit_count()
                    if common >= 2:
                        total += common * (common - 1) // 2
        return total // 2

    # Step 1: Build the polarity graph (C4-free, 90 edges)
    core_edges, core_adj = build_polarity_graph()
    
    if len(core_edges) != 90:
        # Fallback: greedy C4-free construction
        core_edges = []
        core_adj = [set() for _ in range(n)]
        all_pairs = list(range(n*(n-1)//2))
        random.shuffle(all_pairs)
        for idx in all_pairs:
            if len(core_edges) >= 90:
                break
            u = idx // (n - 1)
            v = (idx % (n - 1)) + u + 1
            if u >= v:
                continue
            common = core_adj[u] & core_adj[v]
            if len(common) < 2:
                core_edges.append([u, v])
                core_adj[u].add(v)
                core_adj[v].add(u)
    
    # Step 2: Add the 91st edge to minimize total C4
    edge_set = set()
    for e in core_edges:
        edge_set.add(tuple(sorted(e)))
    
    # Use bitmask adjacency for speed
    adj_bm = [0] * n
    for i in range(n):
        for j in core_adj[i]:
            adj_bm[i] |= (1 << j)
    
    best_add_c4 = float('inf')
    best_add_edge = None
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in edge_set:
                common = (adj_bm[i] & adj_bm[j]).bit_count()
                new_c4s = common * (common - 1) // 2
                if new_c4s < best_add_c4:
                    best_add_c4 = new_c4s
                    best_add_edge = (i, j)
    
    if best_add_edge is not None:
        core_edges.append(list(best_add_edge))
        core_adj[best_add_edge[0]].add(best_add_edge[1])
        core_adj[best_add_edge[1]].add(best_add_edge[0])
        edge_set.add(best_add_edge)
        adj_bm[best_add_edge[0]] |= (1 << best_add_edge[1])
        adj_bm[best_add_edge[1]] |= (1 << best_add_edge[0])
    
    current_c4 = count_c4_from_bitmasks(adj_bm)
    
    # Step 3: Fast local search with bitmask adjacency
    if time.time() < deadline - 3:
        # Precompute all non-edges
        all_non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (i,j) not in edge_set:
                    all_non_edges.append((i,j))
        
        # Single-swap local search with bitmask
        improved = True
        no_improve_rounds = 0
        max_no_improve = 3
        
        while improved and time.time() < deadline - 3:
            improved = False
            no_improve_rounds = 0
            random.shuffle(all_non_edges)
            edge_list = [list(e) for e in core_edges]
            
            for e_idx in range(len(edge_list)):
                if time.time() > deadline - 3:
                    break
                u, v = edge_list[e_idx]
                u, v = sorted([u, v])
                
                # Delta from removing (u,v)
                common_uv = (adj_bm[u] & adj_bm[v]).bit_count()
                delta_remove = -common_uv * (common_uv - 1) // 2
                
                # Temporarily remove edge (u,v)
                adj_bm[u] &= ~(1 << v)
                adj_bm[v] &= ~(1 << u)
                
                best_delta = 0
                best_new_edge = None
                
                for (a, b) in all_non_edges:
                    if (a, b) == (u, v):
                        continue
                    common_ab = (adj_bm[a] & adj_bm[b]).bit_count()
                    delta_add = common_ab * (common_ab - 1) // 2
                    total_delta = delta_remove + delta_add
                    
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_new_edge = (a, b)
                
                # Restore (u,v)
                adj_bm[u] |= (1 << v)
                adj_bm[v] |= (1 << u)
                
                if best_new_edge is not None and best_delta < 0:
                    # Perform the swap
                    a, b = best_new_edge
                    adj_bm[u] &= ~(1 << v)
                    adj_bm[v] &= ~(1 << u)
                    adj_bm[a] |= (1 << b)
                    adj_bm[b] |= (1 << a)
                    edge_set.discard((u, v))
                    edge_set.add((a, b))
                    core_edges[e_idx] = [a, b]
                    current_c4 += best_delta
                    improved = True
                    # Update non-edges list
                    if (a, b) in all_non_edges:
                        all_non_edges.remove((a, b))
                    all_non_edges.append((u, v))
                    break
            
            if not improved:
                no_improve_rounds += 1
                if no_improve_rounds >= max_no_improve:
                    break
        
        # Step 4: Multi-swap (2-edge) moves to escape local optima
        if time.time() < deadline - 5:
            for multi_round in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Try 2-edge swaps: remove edges e1, e2 and add edges f1, f2
                best_total_delta = 0
                best_swap = None
                
                # Sample a subset of edge pairs and non-edge pairs for speed
                sample_size = min(100, len(core_edges))
                edge_indices = random.sample(range(len(core_edges)), sample_size)
                sample_size_ne = min(50, len(all_non_edges))
                ne_sample = random.sample(all_non_edges, sample_size_ne)
                
                for i1 in range(len(edge_indices)):
                    if time.time() > deadline - 5:
                        break
                    e1 = tuple(sorted(core_edges[edge_indices[i1]]))
                    
                    for i2 in range(i1+1, len(edge_indices)):
                        e2 = tuple(sorted(core_edges[edge_indices[i2]]))
                        
                        # Remove both edges
                        a1, b1 = e1
                        a2, b2 = e2
                        adj_bm[a1] &= ~(1 << b1)
                        adj_bm[b1] &= ~(1 << a1)
                        adj_bm[a2] &= ~(1 << b2)
                        adj_bm[b2] &= ~(1 << a2)
                        
                        common_e1 = (adj_bm[a1] & adj_bm[b1]).bit_count()
                        common_e2 = (adj_bm[a2] & adj_bm[b2]).bit_count()
                        delta_remove = -(common_e1*(common_e1-1)//2) - (common_e2*(common_e2-1)//2)
                        
                        best_add_delta = 0
                        best_add_pair = None
                        
                        for f1 in ne_sample:
                            if f1 == e1 or f1 == e2:
                                continue
                            fa, fb = f1
                            common_f1 = (adj_bm[fa] & adj_bm[fb]).bit_count()
                            delta_add_f1 = common_f1*(common_f1-1)//2
                            
                            # Temporarily add f1
                            adj_bm[fa] |= (1 << fb)
                            adj_bm[fb] |= (1 << fa)
                            
                            for f2 in ne_sample:
                                if f2 == e1 or f2 == e2 or f2 == f1:
                                    continue
                                ga, gb = f2
                                # Check if f2 shares vertices with f1 in a way that creates issues
                                common_f2 = (adj_bm[ga] & adj_bm[gb]).bit_count()
                                delta_add_f2 = common_f2*(common_f2-1)//2
                                total_add = delta_add_f1 + delta_add_f2
                                total_delta = delta_remove + total_add
                                
                                if total_delta < best_add_delta:
                                    best_add_delta = total_delta
                                    best_add_pair = (f1, f2)
                            
                            # Remove f1
                            adj_bm[fa] &= ~(1 << fb)
                            adj_bm[fb] &= ~(1 << fa)
                        
                        # Restore both removed edges
                        adj_bm[a1] |= (1 << b1)
                        adj_bm[b1] |= (1 << a1)
                        adj_bm[a2] |= (1 << b2)
                        adj_bm[b2] |= (1 << a2)
                        
                        if best_add_pair is not None and best_add_delta < 0:
                            if best_add_delta < best_total_delta:
                                best_total_delta = best_add_delta
                                best_swap = (e1, e2, best_add_pair)
                
                if best_swap is not None:
                    e1, e2, (f1, f2) = best_swap
                    a1, b1 = e1
                    a2, b2 = e2
                    fa, fb = f1
                    ga, gb = f2
                    
                    # Perform the 2-swap
                    adj_bm[a1] &= ~(1 << b1)
                    adj_bm[b1] &= ~(1 << a1)
                    adj_bm[a2] &= ~(1 << b2)
                    adj_bm[b2] &= ~(1 << a2)
                    adj_bm[fa] |= (1 << fb)
                    adj_bm[fb] |= (1 << fa)
                    adj_bm[ga] |= (1 << gb)
                    adj_bm[gb] |= (1 << ga)
                    
                    edge_set.discard(e1)
                    edge_set.discard(e2)
                    edge_set.add(f1)
                    edge_set.add(f2)
                    
                    # Update core_edges
                    core_edges[edge_indices[edge_indices.index(e1)]] = [fa, fb]
                    # Find and update second edge
                    for idx in range(len(core_edges)):
                        if tuple(sorted(core_edges[idx])) == e2:
                            core_edges[idx] = [ga, gb]
                            break
                    
                    # Update non-edges
                    for ne in [f1, f2]:
                        if ne in all_non_edges:
                            all_non_edges.remove(ne)
                    all_non_edges.append(e1)
                    all_non_edges.append(e2)
                    
                    current_c4 += best_total_delta
        
        # Step 5: Final polish - try all single swaps again with current state
        if time.time() < deadline - 2:
            all_non_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if (i,j) not in edge_set:
                        all_non_edges.append((i,j))
            
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                random.shuffle(all_non_edges)
                
                for e_idx in range(len(core_edges)):
                    if time.time() > deadline - 2:
                        break
                    u, v = sorted(core_edges[e_idx])
                    
                    common_uv = (adj_bm[u] & adj_bm[v]).bit_count()
                    delta_remove = -common_uv * (common_uv - 1) // 2
                    
                    adj_bm[u] &= ~(1 << v)
                    adj_bm[v] &= ~(1 << u)
                    
                    best_delta = 0
                    best_new_edge = None
                    
                    for (a, b) in all_non_edges:
                        if (a, b) == (u, v):
                            continue
                        common_ab = (adj_bm[a] & adj_bm[b]).bit_count()
                        delta_add = common_ab * (common_ab - 1) // 2
                        total_delta = delta_remove + delta_add
                        
                        if total_delta < best_delta:
                            best_delta = total_delta
                            best_new_edge = (a, b)
                    
                    adj_bm[u] |= (1 << v)
                    adj_bm[v] |= (1 << u)
                    
                    if best_new_edge is not None and best_delta < 0:
                        a, b = best_new_edge
                        adj_bm[u] &= ~(1 << v)
                        adj_bm[v] &= ~(1 << u)
                        adj_bm[a] |= (1 << b)
                        adj_bm[b] |= (1 << a)
                        edge_set.discard((u, v))
                        edge_set.add((a, b))
                        core_edges[e_idx] = [a, b]
                        current_c4 += best_delta
                        improved = True
                        if (a, b) in all_non_edges:
                            all_non_edges.remove((a, b))
                        all_non_edges.append((u, v))
                        break
        
        # Final verification
        current_c4 = count_c4_from_bitmasks(adj_bm)
    
    # Ensure exactly 91 distinct edges
    unique_edges = [list(e) for e in sorted(edge_set)]
    
    if len(unique_edges) > target_edges:
        # Remove edges that contribute most to C4
        unique_edges = unique_edges[:target_edges]
    
    if len(unique_edges) < target_edges:
        adj_temp_bm = [0] * n
        for e in unique_edges:
            adj_temp_bm[e[0]] |= (1 << e[1])
            adj_temp_bm[e[1]] |= (1 << e[0])
        for i in range(n):
            for j in range(i+1, n):
                if len(unique_edges) >= target_edges:
                    break
                if not (adj_temp_bm[i] >> j) & 1:
                    unique_edges.append([i, j])
                    adj_temp_bm[i] |= (1 << j)
                    adj_temp_bm[j] |= (1 << i)
            if len(unique_edges) >= target_edges:
                break
    
    return unique_edges[:target_edges]
# EVOLVE-BLOCK-END
