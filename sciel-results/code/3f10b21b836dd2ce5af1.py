import time
import random

def construct(k, g):
    deadline = time.time() + 145.0
    
    def luw_construct(k, g):
        # ... (same as before) ...
        pass

    def random_repair(k, g, N, max_steps=10000):
        # 1. Generate random 4-regular graph
        # Use a simple algorithm:
        # Start with a ring (2-regular).
        # Add random chords to make it 4-regular.
        # This is tricky.
        # Better: Use the "Configuration Model" with rejection.
        
        # Simple 4-regular generator:
        # C_N(1, 2) is 4-regular.
        adj = {i: set() for i in range(N)}
        edges = set()
        for i in range(N):
            for j in [(i+1)%N, (i+2)%N]:
                if i < j:
                    adj[i].add(j)
                    adj[j].add(i)
                    edges.add((i, j))
                else:
                    adj[i].add(j)
                    adj[j].add(i)
                    edges.add((j, i))
                    
        # Note: C_N(1,2) has girth 4 or 5.
        
        # 2. Repair
        for step in range(max_steps):
            if time.time() > deadline - 5:
                return None
                
            # Find a short cycle
            # BFS from a random start node
            start = random.randint(0, N-1)
            visited = {start: (start, -1, 0)}
            queue = [(start, -1, 0)]
            cycle_found = None
            
            while queue:
                u, p, d = queue.pop(0)
                if d >= g // 2:
                    continue
                for v in adj[u]:
                    if v == p:
                        continue
                    if v not in visited:
                        visited[v] = (u, p, d+1)
                        queue.append((v, u, d+1))
                    else:
                        # Check if v is in the current path? 
                        # If v is visited, it might be from a different branch.
                        # If we meet a visited node, we have a cycle.
                        # Length = d + 1 + visited[v][2]
                        # But we must ensure it's not the parent.
                        if v != p and d + 1 + visited[v][2] < g:
                            # Reconstruct cycle
                            # Path from start to u:
                            path1 = [u]
                            cur = u
                            while cur != start:
                                cur = visited[cur][0]
                                path1.append(cur)
                                if cur == v: break # Should not happen if v is not in path1
                            # Path from start to v:
                            path2 = [v]
                            cur = v
                            while cur != start:
                                cur = visited[cur][0]
                                path2.append(cur)
                            # Combine
                            # The cycle is path1 + path2 (reversed)
                            # We just need two edges to swap.
                            # Pick an edge from path1 and an edge from path2.
                            # Or just pick two random edges from the union.
                            cycle_edges = set()
                            for i in range(len(path1)-1):
                                cycle_edges.add((min(path1[i], path1[i+1]), max(path1[i], path1[i+1])))
                            for i in range(len(path2)-1):
                                cycle_edges.add((min(path2[i], path2[i+1]), max(path2[i], path2[i+1])))
                            
                            if len(cycle_edges) >= 2:
                                cycle_found = list(cycle_edges)
                                break
                if cycle_found:
                    break
                    
            if not cycle_found:
                # No short cycle found from this start. 
                # Try a few more starts? 
                # For simplicity, if we don't find one, we assume it's ok? 
                # No, we must be sure. 
                # But for N < 1000, if girth is high, BFS is fast.
                # If we do 10 random BFS and find no short cycle, it's likely girth >= g.
                # Let's do a full check if we are close to the limit.
                # For now, let's assume if we don't find one quickly, we might be ok.
                # Actually, let's just do a full girth check if we haven't found a cycle in a few steps.
                
                # To be safe, let's run a full check.
                # If full check passes, return.
                if is_girth_ok(adj, g):
                    return N, list(edges)
                # If not ok, but we didn't find a specific cycle in the quick check,
                # just do a random swap.
                
            if cycle_found:
                # Pick two edges to swap
                e1, e2 = random.sample(cycle_found, 2)
                u, v = e1
                x, y = e2
                # Swap
                if u == x or u == y or v == x or v == y:
                    continue
                if (u, x) in adj[u] or (v, y) in adj[v]:
                    continue
                if u == x: continue
                
                # Perform swap
                adj[u].remove(v)
                adj[v].remove(u)
                adj[x].remove(y)
                adj[y].remove(x)
                
                adj[u].add(x)
                adj[x].add(u)
                adj[v].add(y)
                adj[y].add(v)
                
                edges.remove(e1)
                edges.remove(e2)
                edges.add((u, x))
                edges.add((v, y))
            else:
                # Random swap
                e1 = random.choice(list(edges))
                e2 = random.choice(list(edges))
                if e1 == e2: continue
                u, v = e1
                x, y = e2
                if u == x or u == y or v == x or v == y:
                    continue
                if (u, x) in adj[u] or (v, y) in adj[v]:
                    continue
                if u == x: continue
                
                adj[u].remove(v)
                adj[v].remove(u)
                adj[x].remove(y)
                adj[y].remove(x)
                
                adj[u].add(x)
                adj[x].add(u)
                adj[v].add(y)
                adj[y].add(v)
                
                edges.remove(e1)
                edges.remove(e2)
                edges.add((u, x))
                edges.add((v, y))
                
        return None

    def is_girth_ok(adj, g):
        # Full BFS from every node
        for start in range(len(adj)):
            visited = {start: (start, -1, 0)}
            queue = [(start, -1, 0)]
            while queue:
                u, p, d = queue.pop(0)
                if d >= g // 2:
                    continue
                for v in adj[u]:
                    if v == p:
                        continue
                    if v not in visited:
                        visited[v] = (u, p, d+1)
                        queue.append((v, u, d+1))
                    else:
                        if v != p and d + 1 + visited[v][2] < g:
                            return False
        return True

    # Try N values
    # LUW N is ~2000.
    # We want N < 2000.
    # Let's try 320, 400, 500, 1000.
    
    for N in [320, 400, 500, 800, 1000]:
        if time.time() > deadline - 20:
            break
        res = random_repair(k, g, N, max_steps=5000)
        if res:
            return res
            
    return luw_construct(k, g)
