import random
import time

def construct():
    """
    Construct a dense graph on 15 vertices with no simple path of length 3
    having endpoints of equal degree.
    """
    n = 15
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    best_edges = None
    best_edge_count = -1
    
    # Helper to check validity and count violations
    def get_violations(adj, deg):
        violations = []
        for u in range(n):
            for v in adj[u]:
                if v < u: continue # Each edge once
                for s in adj[u]:
                    if s == v: continue
                    for t in adj[v]:
                        if t == u or t == s: continue
                        if deg[s] == deg[t]:
                            violations.append((s, u, v, t))
        return violations

    def count_violations(adj, deg):
        count = 0
        for u in range(n):
            for v in adj[u]:
                if v < u: continue
                for s in adj[u]:
                    if s == v: continue
                    for t in adj[v]:
                        if t == u or t == s: continue
                        if deg[s] == deg[t]:
                            count += 1
        return count

    def compute_degrees(adj):
        return [len(adj[i]) for i in range(n)]

    def is_valid(adj):
        deg = compute_degrees(adj)
        return count_violations(adj, deg) == 0

    def edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    # Main search loop
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        # Initialize dense random graph
        # Try different densities
        p = 0.9 if attempt % 3 == 0 else (0.8 if attempt % 3 == 1 else 0.7)
        adj = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i][j] = True
                    adj[j][i] = True
        
        # Convert to list of neighbors for faster access
        adj_list = [set() for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    adj_list[i].add(j)
        
        deg = [len(adj_list[i]) for i in range(n)]
        
        # Repair loop
        max_repair_steps = 200
        for step in range(max_repair_steps):
            if time.time() > deadline:
                break
                
            v_count = count_violations(adj_list, deg)
            if v_count == 0:
                # Valid graph found
                edges = []
                for i in range(n):
                    for j in adj_list[i]:
                        if i < j:
                            edges.append([i, j])
                if len(edges) > best_edge_count:
                    best_edge_count = len(edges)
                    best_edges = edges
                break
            
            # Find one violation
            # Re-find a specific violation to guide repair
            viol = None
            for u in range(n):
                if viol: break
                for v in adj_list[u]:
                    if v < u: continue
                    for s in adj_list[u]:
                        if s == v: continue
                        for t in adj_list[v]:
                            if t == u or t == s: continue
                            if deg[s] == deg[t]:
                                viol = (s, u, v, t)
                                break
                        if viol: break
                    if viol: break
                if viol: break
            
            if not viol:
                break # Should not happen if v_count > 0
                
            s, u, v, t = viol
            
            # Candidates for removal
            candidates = set()
            candidates.add((s, u))
            candidates.add((u, v))
            candidates.add((v, t))
            
            # Add random neighbors of s and t to change degrees
            if adj_list[s]:
                rand_s_nei = random.choice(list(adj_list[s]))
                candidates.add((s, rand_s_nei))
            if adj_list[t]:
                rand_t_nei = random.choice(list(adj_list[t]))
                candidates.add((t, rand_t_nei))
            
            best_candidate = None
            best_v_count = v_count + 1 # We want to minimize
            
            for c in candidates:
                a, b = c
                if a > b: a, b = b, a
                if not adj[a][b]:
                    continue
                    
                # Simulate removal
                adj[a][b] = False
                adj[b][a] = False
                adj_list[a].remove(b)
                adj_list[b].remove(a)
                deg[a] -= 1
                deg[b] -= 1
                
                new_v_count = count_violations(adj_list, deg)
                
                if new_v_count < best_v_count:
                    best_v_count = new_v_count
                    best_candidate = (a, b)
                    
                # Restore
                adj[a][b] = True
                adj[b][a] = True
                adj_list[a].add(b)
                adj_list[b].add(a)
                deg[a] += 1
                deg[b] += 1
            
            if best_candidate:
                a, b = best_candidate
                adj[a][b] = False
                adj[b][a] = False
                adj_list[a].remove(b)
                adj_list[b].remove(a)
                deg[a] -= 1
                deg[b] -= 1
            else:
                # No improvement, try random removal to escape
                all_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i][j]]
                if all_edges:
                    a, b = random.choice(all_edges)
                    adj[a][b] = False
                    adj[b][a] = False
                    adj_list[a].remove(b)
                    adj_list[b].remove(a)
                    deg[a] -= 1
                    deg[b] -= 1
                else:
                    break # No edges left

    if best_edges is None:
        # Fallback to seed if no valid graph found (should not happen)
        return [[0, v] for v in range(1, n)]
        
    return best_edges
