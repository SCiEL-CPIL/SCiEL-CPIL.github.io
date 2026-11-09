"""Improved program: construct() returns 91 edges on vertices 0..30 minimizing C4s.
Uses polarity graph cores with simulated annealing and 2-swap local search.
"""
import numpy as np
import time
import random

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin under 120s kill

    q = 5
    # Enumerate all points of PG(2,5)
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, 3, 5)
                    nx, ny, nz = 1, (y * inv_x) % 5, (z * inv_x) % 5
                elif y != 0:
                    inv_y = pow(y, 3, 5)
                    nx, ny, nz = 0, 1, (z * inv_y) % 5
                else:
                    nx, ny, nz = 0, 0, 1
                key = (nx, ny, nz)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Try multiple polarity forms (including non-diagonal symmetric forms)
    best_result = None
    best_total_c4 = float('inf')
    
    # Generate symmetric 3x3 matrices over GF(5) that are non-degenerate
    forms = []
    for a in range(1, 5):
        for b in range(1, 5):
            for c in range(1, 5):
                forms.append((a, b, c, None))
    
    for a in [1, 2, 3, 4]:
        for b in [1, 2, 3, 4]:
            for c in [1, 2, 3, 4]:
                for d in [1, 2, 3, 4]:
                    forms.append((a, b, c, d))
    
    random.seed(42)
    random.shuffle(forms)
    
    cores_found = []
    for form in forms:
        if time.time() > deadline - 60:
            break
        
        a, b, c, d = form
        
        edges_core = []
        for i in range(31):
            for j in range(i+1, 31):
                x1, y1, z1 = points[i]
                x2, y2, z2 = points[j]
                if d is None:
                    val = (a*x1*x2 + b*y1*y2 + c*z1*z2) % 5
                else:
                    val = (a*x1*x2 + d*x1*y2 + d*y1*x2 + b*y1*y2 + c*z1*z2) % 5
                if val == 0:
                    edges_core.append([i, j])
        
        if len(edges_core) != 90:
            continue
        
        adj = [set() for _ in range(31)]
        for u, v in edges_core:
            adj[u].add(v)
            adj[v].add(u)
        
        best_edge = None
        best_add = float('inf')
        
        for u in range(31):
            for v in range(u+1, 31):
                if v in adj[u]:
                    continue
                add = 0
                for a2 in adj[u]:
                    for b2 in adj[v]:
                        if b2 in adj[a2]:
                            add += 1
                if add < best_add:
                    best_add = add
                    best_edge = [u, v]
        
        if best_edge is None:
            continue
        
        total_c4 = best_add
        
        if total_c4 < best_total_c4:
            best_total_c4 = total_c4
            best_result = edges_core + [best_edge]
        
        cores_found.append((edges_core, adj.copy()))
        
        if time.time() > deadline - 60:
            break
    
    if best_result is None:
        return _greedy_construct()
    
    final_edges = best_result
    adj = [set() for _ in range(31)]
    for u, v in final_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def count_total_c4(adj_local):
        total = 0
        for u in range(31):
            for v in range(u+1, 31):
                cn = len(adj_local[u] & adj_local[v])
                total += cn * (cn - 1) // 2
        return total // 4
    
    current_c4 = count_total_c4(adj)
    best_c4 = current_c4
    best_edges = [list(e) for e in final_edges]
    
    # Simulated annealing with single-edge swaps
    T_start = 10.0
    T_end = 0.01
    cooling = 0.995
    
    all_pairs = [(i, j) for i in range(31) for j in range(i+1, 31)]
    
    T = T_start
    sa_iterations = 0
    
    while time.time() < deadline - 30.0 and T > T_end:
        sa_iterations += 1
        
        idx = random.randint(0, 90)
        u, v = final_edges[idx]
        
        for _ in range(20):
            a3, b3 = random.choice(all_pairs)
            if b3 not in adj[a3]:
                break
        else:
            T *= cooling
            continue
        
        c4_destroyed = 0
        for a2 in adj[u]:
            if a2 == v:
                continue
            for b2 in adj[v]:
                if b2 == u:
                    continue
                if b2 in adj[a2]:
                    c4_destroyed += 1
        
        c4_created = 0
        for a2 in adj[a3]:
            for b2 in adj[b3]:
                if b2 in adj[a2]:
                    c4_created += 1
        
        delta = c4_created - c4_destroyed
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            adj[u].remove(v)
            adj[v].remove(u)
            final_edges[idx] = [a3, b3]
            adj[a3].add(b3)
            adj[b3].add(a3)
            
            current_c4 += delta
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edges = [list(e) for e in final_edges]
        
        T *= cooling
    
    # Use the best found
    if best_c4 < count_total_c4(adj):
        final_edges = [list(e) for e in best_edges]
        adj = [set() for _ in range(31)]
        for u, v in final_edges:
            adj[u].add(v)
            adj[v].add(u)
        current_c4 = best_c4
    else:
        best_c4 = current_c4
        best_edges = [list(e) for e in final_edges]
    
    # Final polish: greedy 1-swap descent (exact, exhaustive per edge)
    final_time_limit = deadline - 5.0
    improved = True
    while improved and time.time() < final_time_limit:
        improved = False
        for i in range(91):
            if time.time() > final_time_limit:
                break
            u, v = final_edges[i]
            
            # Remove edge (u,v)
            adj[u].discard(v)
            adj[v].discard(u)
            c4_base = count_total_c4(adj)
            
            best_new = None
            best_d = float('inf')
            
            # Try all possible non-edges as replacement
            for a3, b3 in all_pairs:
                if b3 in adj[a3]:
                    continue
                # Allow any non-edge, even sharing vertices with (u,v)
                # since we're replacing (u,v) entirely
                
                adj[a3].add(b3)
                adj[b3].add(a3)
                c4_new = count_total_c4(adj)
                adj[a3].discard(b3)
                adj[b3].discard(a3)
                
                d = c4_new - c4_base
                if d < best_d:
                    best_d = d
                    best_new = (a3, b3)
            
            # Restore (u,v)
            adj[u].add(v)
            adj[v].add(u)
            
            if best_new is not None and best_d < 0:
                a3, b3 = best_new
                adj[u].remove(v)
                adj[v].remove(u)
                adj[a3].add(b3)
                adj[b3].add(a3)
                final_edges[i] = [a3, b3]
                current_c4 += best_d
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_edges = [list(e) for e in final_edges]
                improved = True
    
    # Use the best found
    if best_c4 < count_total_c4(adj):
        final_edges = [list(e) for e in best_edges]
        adj = [set() for _ in range(31)]
        for u, v in final_edges:
            adj[u].add(v)
            adj[v].add(u)
    
    # Validate
    assert len(final_edges) == 91
    edge_set = set(tuple(sorted(e)) for e in final_edges)
    assert len(edge_set) == 91
    
    return final_edges

def _greedy_construct():
    """Fallback greedy construction."""
    random.seed(42)
    
    edges = []
    adj = [set() for _ in range(31)]
    
    remaining = []
    for i in range(31):
        for j in range(i+1, 31):
            remaining.append([i, j])
    
    random.shuffle(remaining)
    
    for _ in range(91):
        best_e = None
        best_add = float('inf')
        for e in remaining:
            u, v = e
            if v in adj[u]:
                continue
            add = 0
            for a in adj[u]:
                for b in adj[v]:
                    if b in adj[a]:
                        add += 1
            if add < best_add:
                best_add = add
                best_e = e
            if add == 0:
                break
        if best_e is None:
            break
        u, v = best_e
        edges.append(best_e)
        adj[u].add(v)
        adj[v].add(u)
        remaining.remove(best_e)
    
    while len(edges) < 91 and remaining:
        for e in remaining:
            u, v = e
            if v not in adj[u]:
                edges.append(e)
                adj[u].add(v)
                adj[v].add(u)
                remaining.remove(e)
                break
    
    return edges[:91]
