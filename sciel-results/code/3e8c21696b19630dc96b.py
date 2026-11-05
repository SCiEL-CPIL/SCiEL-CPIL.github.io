import time
from collections import defaultdict
import random

def instance(n, p, seed):
    import random
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def decompose(edges):
    if not edges:
        return []
    
    # Initial greedy decomposition
    remaining = set(edges)
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    paths = []
    
    while remaining:
        # Pick edge with highest degree sum to start longer paths
        best_e = None
        best_score = -1
        for e in remaining:
            u, v = e
            score = len(adj[u]) + len(adj[v])
            if score > best_score:
                best_score = score
                best_e = e
        
        if best_e is None:
            break
            
        u, v = best_e
        remaining.remove(best_e)
        adj[u].discard(v)
        adj[v].discard(u)
        
        path = [u, v]
        path_set = {u, v}
        
        # Extend from v
        curr = v
        while True:
            next_v = None
            best_deg = -1
            for w in sorted(adj[curr]):
                if w not in path_set:
                    deg = len(adj[w])
                    if deg > best_deg:
                        best_deg = deg
                        next_v = w
            
            if next_v is None:
                break
            
            e2 = tuple(sorted((curr, next_v)))
            if e2 in remaining:
                remaining.remove(e2)
                adj[curr].discard(next_v)
                adj[next_v].discard(curr)
                path.append(next_v)
                path_set.add(next_v)
                curr = next_v
            else:
                break

        # Extend from u
        curr = u
        while True:
            next_v = None
            best_deg = -1
            for w in sorted(adj[curr]):
                if w not in path_set:
                    deg = len(adj[w])
                    if deg > best_deg:
                        best_deg = deg
                        next_v = w
            
            if next_v is None:
                break
            
            e2 = tuple(sorted((curr, next_v)))
            if e2 in remaining:
                remaining.remove(e2)
                adj[curr].discard(next_v)
                adj[next_v].discard(curr)
                path.insert(0, next_v)
                path_set.add(next_v)
                curr = next_v
            else:
                break

        paths.append(path)
        
    # Local search: try to merge paths
    deadline = time.time() + 100
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Rebuild edge usage from current paths
        path_edges = set()
        for p in paths:
            for i in range(len(p) - 1):
                path_edges.add(tuple(sorted((p[i], p[i+1]))))
                
        remaining_edges = set(edges) - path_edges
        
        # Try to merge every pair
        for i in range(len(paths)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(paths)):
                if time.time() > deadline:
                    break
                    
                # Try to connect end of path i to start of path j
                # Or start to end, end to end, start to start
                # We need a path in remaining_edges from end_i to start_j
                # such that it doesn't visit any vertex in path i or path j
                
                # For efficiency, only try a few connection types
                # Type 1: end(i) -> start(j)
                # Type 2: start(i) -> start(j)
                # Type 3: end(i) -> end(j)
                # Type 4: start(i) -> end(j)
                
                # We will try to find a simple path in remaining_edges
                # between two endpoints, avoiding vertices in the two paths
                
                def find_connector(start, end, forbidden_vertices):
                    # BFS in remaining_edges from start to end
                    # avoiding forbidden_vertices
                    if start == end:
                        return []
                    if start in forbidden_vertices or end in forbidden_vertices:
                        return None
                            
                    visited = {start}
                    queue = [(start, [start])]
                    
                    while queue:
                        curr, path = queue.pop(0)
                        if curr == end:
                            return path
                            
                        # Get neighbors in remaining_edges
                        for u, v in remaining_edges:
                            if u == curr and v not in visited and v not in forbidden_vertices:
                                visited.add(v)
                                queue.append((v, path + [v]))
                            elif v == curr and u not in visited and u not in forbidden_vertices:
                                visited.add(u)
                                queue.append((u, path + [u]))
                                
                    return None
                
                # Get vertices in paths i and j
                verts_i = set(paths[i])
                verts_j = set(paths[j])
                forbidden = verts_i | verts_j
                
                # Try 4 connection types
                # 1. end(i) to start(j): new path = path_i + connector[1:] + path_j
                #    Wait, connector starts with start(i). So connector = [s, ..., e]
                #    Merged: path_i + connector[1:] + path_j
                #    But we must ensure the edge (end_i, s) is in remaining?
                #    No, the connector path uses edges from remaining_edges.
                #    The first edge of connector is (start_i, next).
                #    So we need (end_i, start_i) to be... wait.
                #    Path i ends at end_i. Path j starts at start_j.
                #    We want to connect end_i to start_j via a path in remaining edges.
                #    Let connector be a path [v0, v1, ..., vk] where v0=end_i and vk=start_j?
                #    No, the connector must use edges from remaining.
                #    If we connect end_i to start_j, the new path is:
                #    path_i + [v1, ..., vk-1] + path_j
                #    where connector is [end_i, v1, ..., vk-1, start_j]
                #    And edges (end_i, v1), ..., (vk-1, start_j) are in remaining.
                #    And vertices v1...vk-1 are not in path_i or path_j.
                
                # So we search for a path from end_i to start_j in remaining_edges
                # avoiding vertices in (verts_i - {end_i}) U (verts_j - {start_j})
                
                forbidden_1 = (verts_i - {paths[i][-1]}) | (verts_j - {paths[j][0]})
                conn = find_connector(paths[i][-1], paths[j][0], forbidden_1)
                if conn is not None:
                    # Merge
                    # conn = [end_i, ..., start_j]
                    # New path = path_i + conn[1:-1] + path_j
                    # Wait, conn[0] is end_i, which is already in path_i.
                    # conn[-1] is start_j, which is already in path_j.
                    new_path = paths[i] + conn[1:-1] + paths[j]
                    paths.pop(j)
                    paths.pop(i)
                    paths.append(new_path)
                    improved = True
                    break # Restart outer loops? Or just continue?
                    # Restarting is safer to update remaining_edges
                    # But for speed, we can just mark improved and break out
                    # Let's break out of all loops to rebuild state
                    # Actually, breaking inner loop and letting outer continue might use stale remaining_edges
                    # So we break out completely.
                    break
                    
                # 2. start(i) to end(j)
                #    New path = reverse(path_i) + conn[1:-1] + reverse(path_j)? No.
                #    Path i: s_i ... e_i
                #    Path j: s_j ... e_j
                #    Connect s_i to e_j.
                #    New path: e_i ... s_i + conn[1:-1] + e_j ... s_j (reversed j)
                #    Let's just reverse path j and connect s_i to s_j(rev) = e_j
                #    Actually, simpler:
                #    Reverse path j: e_j ... s_j
                #    Connect s_i to e_j.
                #    New path: reverse(path_i) + conn[1:-1] + path_j_rev
                #    Wait, path_i is s_i...e_i. Reverse is e_i...s_i.
                #    Connector from s_i to e_j.
                #    New: e_i...s_i + (connector without ends) + e_j...s_j
                #    This is valid.
                
                forbidden_2 = (verts_i - {paths[i][0]}) | (verts_j - {paths[j][-1]})
                conn = find_connector(paths[i][0], paths[j][-1], forbidden_2)
                if conn is not None:
                    # conn = [s_i, ..., e_j]
                    # New path = reverse(paths[i]) + conn[1:-1] + reverse(paths[j])
                    new_path = list(reversed(paths[i])) + conn[1:-1] + list(reversed(paths[j]))
                    paths.pop(j)
                    paths.pop(i)
                    paths.append(new_path)
                    improved = True
                    break
                    
                # 3. end(i) to end(j)
                #    Reverse path j. Connect e_i to e_j.
                #    New path: path_i + conn[1:-1] + reverse(path_j)
                
                forbidden_3 = (verts_i - {paths[i][-1]}) | (verts_j - {paths[j][-1]})
                conn = find_connector(paths[i][-1], paths[j][-1], forbidden_3)
                if conn is not None:
                    new_path = paths[i] + conn[1:-1] + list(reversed(paths[j]))
                    paths.pop(j)
                    paths.pop(i)
                    paths.append(new_path)
                    improved = True
                    break
                    
                # 4. start(i) to start(j)
                #    Reverse path i. Connect s_i to s_j.
                #    New path: reverse(path_i) + conn[1:-1] + path_j
                
                forbidden_4 = (verts_i - {paths[i][0]}) | (verts_j - {paths[j][0]})
                conn = find_connector(paths[i][0], paths[j][0], forbidden_4)
                if conn is not None:
                    new_path = list(reversed(paths[i])) + conn[1:-1] + paths[j]
                    paths.pop(j)
                    paths.pop(i)
                    paths.append(new_path)
                    improved = True
                    break
                    
            if improved:
                break
                
    return paths

def construct():
    deadline = time.time() + 140
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
        
        if time.time() > deadline:
             pass
             
    return results
