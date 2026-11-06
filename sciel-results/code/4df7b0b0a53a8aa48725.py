import random
import time
import numpy as np

def construct():
    n = 31
    target_edges = 91
    start_time = time.time()
    # Safety margin: return 5 seconds before 150s limit
    deadline = start_time + 145.0
    
    # Build PG(2,5) polarity graph
    q = 5
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    key = (1, y * pow(x, q-2, q) % q, z * pow(x, q-2, q) % q)
                elif y != 0:
                    key = (0, 1, z * pow(y, q-2, q) % q)
                else:
                    key = (0, 0, 1)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    def is_on_polar(p, qpt):
        x, y, z = p
        X, Y, Z = qpt
        return (x * Y + y * X + z * Z) % q == 0
    
    edges = set()
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if is_on_polar(points[i], points[j]):
                edges.add((i, j))
                adj[i].add(j)
                adj[j].add(i)
    
    assert len(edges) == 90, f"Expected 90 edges, got {len(edges)}"
    
    def get_c4s_for_edge(u, v, adj_sets):
        """Find all C4s containing edge (u,v)."""
        c4s = set()
        # Type 1: (u,v) is a side: u-v-x-y-u
        # x in N(v), y in N(u), x~y, x!=y
        for x in adj_sets[v]:
            if x == u:
                continue
            for y in adj_sets[u]:
                if y == v or y == x:
                    continue
                if x in adj_sets[y]:
                    c4 = frozenset([u, v, x, y])
                    c4s.add(c4)
        # Type 2: (u,v) is a diagonal: u-x-v-y-u
        # x, y in N(u) cap N(v)
        common = adj_sets[u] & adj_sets[v]
        common_list = list(common)
        for i in range(len(common_list)):
            for j in range(i + 1, len(common_list)):
                x = common_list[i]
                y = common_list[j]
                c4 = frozenset([u, v, x, y])
                c4s.add(c4)
        return c4s
    
    # Find best edge to add to reach 91
    best_add = None
    min_c4s_created = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edges:
                c4s = get_c4s_for_edge(i, j, adj)
                if len(c4s) < min_c4s_created:
                    min_c4s_created = len(c4s)
                    best_add = (i, j)
    
    u_add, v_add = best_add
    edges.add((u_add, v_add))
    adj[u_add].add(v_add)
    adj[v_add].add(u_add)
    
    # Initialize all C4s
    all_c4s = set()
    for u, v in edges:
        all_c4s.update(get_c4s_for_edge(u, v, adj))
    
    current_c4_count = len(all_c4s)
    best_c4_count = current_c4_count
    best_edges = set(edges)
    best_adj = [set(s) for s in adj]
    best_c4s = set(all_c4s)
    
    random.seed(42)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def is_edge(u, v, edge_set):
        return (u, v) in edge_set or (v, u) in edge_set

    iterations = 0
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline and iterations < 2000000:
        iterations += 1
        
        if not edges:
            break
            
        improved = False
        
        # Strategy 1: Greedy Swap
        # Find the edge that is in the most C4s
        # We can approximate this by checking a few random edges or all edges.
        # With 91 edges, checking all is fast.
        max_c4_edge = None
        max_c4_count_for_edge = -1
        c4s_per_edge = {}
        
        # Calculate C4 count per edge
        # This is O(E * (deg^2)) which is small.
        # But we already have all_c4s.
        # We can just pick a random edge for speed, or the one with highest degree?
        # Let's pick the edge with the highest sum of degrees of its endpoints as a heuristic?
        # Or just random.
        
        # Actually, let's do a random swap first, and if we stall, do greedy.
        
        if no_improve_count < 1000:
            # Random Swap
            u1, v1 = random.choice(list(edges))
            c4s_to_remove = get_c4s_for_edge(u1, v1, adj)
            
            # Try multiple random additions
            for _ in range(20):
                u2, v2 = random.choice(all_pairs)
                if is_edge(u2, v2, edges):
                    continue
                if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                    continue
                
                # Simulate removal of (u1, v1)
                adj[u1].discard(v1)
                adj[v1].discard(u1)
                
                c4s_to_add = get_c4s_for_edge(u2, v2, adj)
                
                # Restore
                adj[u1].add(v1)
                adj[v1].add(u1)
                
                removed_only = c4s_to_remove - c4s_to_add
                added_only = c4s_to_add - c4s_to_remove
                
                new_c4_count = current_c4_count - len(removed_only) + len(added_only)
                
                if new_c4_count < current_c4_count:
                    # Accept
                    adj[u1].discard(v1)
                    adj[v1].discard(u1)
                    adj[u2].add(v2)
                    adj[v2].add(u2)
                    
                    edges.discard((u1, v1))
                    edges.add((u2, v2))
                    
                    all_c4s.difference_update(c4s_to_remove)
                    all_c4s.update(c4s_to_add)
                    
                    current_c4_count = len(all_c4s)
                    improved = True
                    no_improve_count = 0
                    
                    if current_c4_count < best_c4_count:
                        best_c4_count = current_c4_count
                        best_edges = set(edges)
                        best_adj = [set(s) for s in adj]
                        best_c4s = set(all_c4s)
                    break
        else:
            # Greedy Swap: Remove edge with most C4s
            # Find edge with max C4s
            max_count = -1
            target_edge = None
            target_c4s = None
            
            # Sample a few edges to find a good candidate to remove
            # Or iterate all if fast enough
            for e in edges:
                u, v = e
                c4s_e = get_c4s_for_edge(u, v, adj)
                # Count how many of these C4s are in all_c4s?
                # Actually, get_c4s_for_edge returns C4s *containing* the edge.
                # These are a subset of all_c4s.
                # We want to remove an edge that breaks many C4s.
                # The number of C4s broken is |c4s_e|.
                # But some C4s might be re-formed by the new edge?
                # We want to minimize the net change.
                # For now, just pick the edge with the most C4s.
                if len(c4s_e) > max_count:
                    max_count = len(c4s_e)
                    target_edge = e
                    target_c4s = c4s_e
            
            if target_edge:
                u1, v1 = target_edge
                c4s_to_remove = target_c4s
                
                # Find best non-edge to add
                best_u2, best_v2 = None, None
                best_delta = float('inf')
                
                # Sample non-edges
                # There are ~400 non-edges. We can try a few.
                random.shuffle(all_pairs)
                for u2, v2 in all_pairs[:50]:
                    if is_edge(u2, v2, edges):
                        continue
                    if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                        continue
                    
                    adj[u1].discard(v1)
                    adj[v1].discard(u1)
                    
                    c4s_to_add = get_c4s_for_edge(u2, v2, adj)
                    
                    adj[u1].add(v1)
                    adj[v1].add(u1)
                    
                    removed_only = c4s_to_remove - c4s_to_add
                    added_only = c4s_to_add - c4s_to_remove
                    delta = len(added_only) - len(removed_only)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_u2, best_v2 = u2, v2
                        best_c4s_to_add = c4s_to_add
                
                if best_u2 is not None and best_delta < 0:
                    u2, v2 = best_u2, best_v2
                    adj[u1].discard(v1)
                    adj[v1].discard(u1)
                    adj[u2].add(v2)
                    adj[v2].add(u2)
                    
                    edges.discard((u1, v1))
                    edges.add((u2, v2))
                    
                    all_c4s.difference_update(c4s_to_remove)
                    all_c4s.update(best_c4s_to_add)
                    
                    current_c4_count = len(all_c4s)
                    improved = True
                    no_improve_count = 0
                    
                    if current_c4_count < best_c4_count:
                        best_c4_count = current_c4_count
                        best_edges = set(edges)
                        best_adj = [set(s) for s in adj]
                        best_c4s = set(all_c4s)
        
        if not improved:
            no_improve_count += 1
            # If stuck for a long time, perturb
            if no_improve_count > max_no_improve and time.time() < deadline - 20:
                # Perturb
                k = random.randint(2, 4)
                edge_list = list(edges)
                random.shuffle(edge_list)
                for _ in range(min(k, len(edge_list))):
                    e = edge_list.pop()
                    u, v = e
                    adj[u].discard(v)
                    adj[v].discard(u)
                    edges.discard(e)
                
                non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) 
                            if not is_edge(i, j, edges)]
                random.shuffle(non_edges)
                for _ in range(min(k, len(non_edges))):
                    e = non_edges.pop()
                    u, v = e
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.add(e)
                
                # Recompute
                all_c4s = set()
                for u, v in edges:
                    all_c4s.update(get_c4s_for_edge(u, v, adj))
                current_c4_count = len(all_c4s)
                
                if current_c4_count < best_c4_count:
                    best_c4_count = current_c4_count
                    best_edges = set(edges)
                    best_adj = [set(s) for s in adj]
                    best_c4s = set(all_c4s)
                
                no_improve_count = 0

    # Return best result
    best_edges_list = [[u, v] for u, v in sorted(best_edges)]
    return best_edges_list
