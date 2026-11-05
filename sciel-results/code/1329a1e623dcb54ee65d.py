import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    n = 31
    q = 5

    # Build polarity graph of PG(2, q)
    points = []
    for y in range(q):
        for z in range(q):
            points.append((1, y, z))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    assert len(points) == n
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[j]
            if (x1*x2 + y1*y2 + z1*z2) % q == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
    
    m = len(edges)
    
    def count_c4s(adj_sets):
        total = 0
        for u in range(n):
            nu = adj_sets[u]
            for v in range(u+1, n):
                nv = adj_sets[v]
                common = len(nu & nv)
                total += common * (common - 1) // 2
        return total // 2
    
    def build_adj_matrix(adj_sets):
        A = np.zeros((n, n), dtype=np.float64)
        for u in range(n):
            for v in adj_sets[u]:
                A[u, v] = 1.0
        return A
    
    def spectral_c4_estimate(A):
        """Estimate C4 count using spectral properties.
        For adjacency matrix A, trace(A^4) = 2*m + 4*sum(edges) + 8*C4
        where sum(edges) counts 2-paths. More precisely:
        trace(A^4) = 4*sum_{u<v} C(common(u,v),2) + 2*sum_u d_u + ... 
        Actually: trace(A^4) = 2*m + 4*(number of 2-paths) + 8*(number of C4s)
        A 2-path is u-w-v where u~w, w~v, u!=v. Count = sum_w C(d_w, 2).
        So C4 = (trace(A^4) - 2*m - 4*sum_w C(d_w,2)) / 8
        """
        A2 = A @ A
        A4 = A2 @ A2
        trace_A4 = np.trace(A4)
        m_val = int(np.sum(A) / 2)
        degs = np.sum(A, axis=1)
        two_paths = 0.0
        for d in degs:
            two_paths += d * (d - 1) / 2
        c4_est = (trace_A4 - 2*m_val - 4*two_paths) / 8.0
        return max(0, int(round(c4_est)))
    
    def edge_c4_contribution(adj_sets, u, v):
        """Count C4s that use edge (u,v)."""
        count = 0
        for x in adj_sets[v]:
            if x == u: continue
            for y in adj_sets[u]:
                if y == v or y == x: continue
                if y in adj_sets[x]:
                    count += 1
        return count
    
    if m == 90:
        # Need to add 1 edge. Find the non-edge that creates fewest C4s.
        edge_set = set()
        for e in edges:
            edge_set.add((e[0], e[1]) if e[0] < e[1] else (e[1], e[0]))
        
        best_cost = float('inf')
        best_edge = None
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in edge_set:
                    nu = adj[i]
                    nv = adj[j]
                    count = 0
                    for a in nu:
                        for b in nv:
                            if a != b and b in adj[a]:
                                count += 1
                    if count < best_cost:
                        best_cost = count
                        best_edge = [i, j]
        
        if best_edge is not None:
            u, v = best_edge
            edges.append(best_edge)
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v) if u < v else (v, u))
    
    elif m > 91:
        k = m - 91
        if k == 2:
            best_c4s = count_c4s(adj)
            best_remove = None
            for i in range(m):
                if time.time() > deadline - 10:
                    break
                for j in range(i+1, m):
                    new_adj = [set(s) for s in adj]
                    u1, v1 = edges[i]
                    u2, v2 = edges[j]
                    new_adj[u1].discard(v1)
                    new_adj[v1].discard(u1)
                    new_adj[u2].discard(v2)
                    new_adj[v2].discard(u2)
                    c4s = count_c4s(new_adj)
                    if c4s < best_c4s:
                        best_c4s = c4s
                        best_remove = (i, j)
                    if time.time() > deadline - 5:
                        break
                if time.time() > deadline - 5:
                    break
            if best_remove is not None:
                i, j = best_remove
                edges_to_remove = {tuple(sorted(edges[i])), tuple(sorted(edges[j]))}
                edges = [e for e in edges if tuple(sorted(e)) not in edges_to_remove]
                adj = [set() for _ in range(n)]
                for e in edges:
                    adj[e[0]].add(e[1])
                    adj[e[1]].add(e[0])
    
    elif m < 91:
        edge_set = set()
        for e in edges:
            edge_set.add((e[0], e[1]) if e[0] < e[1] else (e[1], e[0]))
        while len(edges) < 91:
            best_cost = float('inf')
            best_edge = None
            for i in range(n):
                for j in range(i+1, n):
                    if (i, j) not in edge_set:
                        nu = adj[i]
                        nv = adj[j]
                        count = 0
                        for a in nu:
                            for b in nv:
                                if a != b and b in adj[a]:
                                    count += 1
                        if count < best_cost:
                            best_cost = count
                            best_edge = [i, j]
            if best_edge is not None:
                edges.append(best_edge)
                u, v = best_edge
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v) if u < v else (v, u))
            else:
                break
    
    assert len(edges) == 91, f"Expected 91 edges, got {len(edges)}"
    
    # Local search with spectral guidance
    current_c4s = count_c4s(adj)
    best_c4s_global = current_c4s
    best_edges = [e[:] for e in edges]
    
    edge_set = set()
    for e in edges:
        edge_set.add((e[0], e[1]) if e[0] < e[1] else (e[1], e[0]))
    
    non_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in edge_set:
                non_edges.append((i, j))
    
    # Precompute edge C4 contributions for prioritization
    def compute_edge_contributions(adj_sets, edge_list):
        contribs = []
        for e in edge_list:
            u, v = e
            contribs.append(edge_c4_contribution(adj_sets, u, v))
        return contribs
    
    # Spectral-guided local search
    improved = True
    iteration = 0
    while improved and time.time() < deadline - 15:
        improved = False
        iteration += 1
        
        # Compute edge contributions to prioritize swaps
        contribs = compute_edge_contributions(adj, edges)
        
        # Sort edges by contribution (highest first) - these are most "expensive"
        edge_order = sorted(range(len(edges)), key=lambda x: -contribs[x])
        
        # For each edge (in order of contribution), try swapping with low-cost non-edges
        for idx in edge_order:
            if time.time() > deadline - 15:
                break
            
            u, v = edges[idx]
            edge_contrib = contribs[idx]
            
            # Try candidate non-edges, prioritizing those with low potential C4 cost
            # Sort non-edges by estimated cost
            candidates = []
            for (a, b) in non_edges:
                # Quick cost estimate
                nu = adj[a]
                nv = adj[b]
                est_cost = 0
                for x in nu:
                    if x in nv:
                        est_cost += 1
                candidates.append((est_cost, a, b))
            
            candidates.sort()
            
            # Try top candidates
            tried = 0
            for est_cost, a, b in candidates[:50]:
                if time.time() > deadline - 15:
                    break
                if tried >= 30:
                    break
                
                # Skip if (a,b) would create a triangle with (u,v) endpoints
                if a == u or a == v or b == u or b == v:
                    tried += 1
                    continue
                
                new_adj = [set(s) for s in adj]
                new_adj[u].discard(v)
                new_adj[v].discard(u)
                new_adj[a].add(b)
                new_adj[b].add(a)
                
                new_c4s = count_c4s(new_adj)
                if new_c4s < current_c4s:
                    edges[idx] = [a, b]
                    adj = new_adj
                    edge_set.remove((u, v) if u < v else (v, u))
                    edge_set.add((a, b) if a < b else (b, a))
                    non_edges.remove((a, b))
                    non_edges.append((u, v))
                    current_c4s = new_c4s
                    improved = True
                    if current_c4s < best_c4s_global:
                        best_c4s_global = current_c4s
                        best_edges = [e[:] for e in edges]
                    break
                tried += 1
    
    # Use best found
    if best_c4s_global < current_c4s:
        edges = best_edges
        adj = [set() for _ in range(n)]
        for e in edges:
            adj[e[0]].add(e[1])
            adj[e[1]].add(e[0])
    
    # Final verification
    assert len(edges) == 91
    edge_set_final = set()
    for e in edges:
        key = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
        assert key not in edge_set_final, f"Duplicate edge {key}"
        edge_set_final.add(key)
    
    return edges
