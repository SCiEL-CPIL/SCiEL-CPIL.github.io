import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    for n in [46, 45, 44]:
        min_deg = n - 39
        if min_deg < 1:
            min_deg = 1
            
        found = False
        attempts = 0
        while time.time() < deadline and not found and attempts < 50:
            attempts += 1
            # Initialize
            adj = [[0]*n for _ in range(n)]
            edges = []
            
            # Random graph
            p = min_deg / n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        adj[i][j] = 1
                        adj[j][i] = 1
                        edges.append([i, j])
                        
            # Ensure min degree
            for _ in range(100):
                degrees = [sum(adj[i]) for i in range(n)]
                if min(degrees) >= min_deg:
                    break
                low = [i for i in range(n) if degrees[i] < min_deg]
                if not low: break
                u = random.choice(low)
                cands = [v for v in range(n) if v != u and adj[u][v] == 0]
                if not cands: break
                v = random.choice(cands)
                adj[u][v] = 1
                adj[v][u] = 1
                edges.append([u, v])
                degrees[u] += 1
                degrees[v] += 1
                
            if min([sum(adj[i]) for i in range(n)]) < min_deg:
                continue
                
            # Remove C4s
            removed = True
            while removed and time.time() < deadline:
                removed = False
                # Find a C4
                c4 = None
                for u in range(n):
                    for v in range(u+1, n):
                        common = [w for w in range(n) if adj[u][w] and adj[v][w]]
                        if len(common) > 1:
                            # u, v have >1 common neighbors
                            # Pick two common neighbors c, d
                            c, d = common[0], common[1]
                            # Cycle u-c-v-d-u
                            # Remove edge (u,v) or (c,d) or (u,c) etc.
                            # Removing (u,v) is safe if it doesn't create new C4s?
                            # Removing an edge cannot create a C4.
                            c4 = (u, v)
                            break
                    if c4: break
                if c4:
                    u, v = c4
                    adj[u][v] = 0
                    adj[v][u] = 0
                    edges.remove([u, v])
                    removed = True
                    
                    # Check min degree and repair
                    degrees = [sum(adj[i]) for i in range(n)]
                    if min(degrees) < min_deg:
                        # Try to add an edge
                        low = [i for i in range(n) if degrees[i] < min_deg]
                        repaired = False
                        for _ in range(100):
                            if not low: break
                            u2 = random.choice(low)
                            cands = [v2 for v2 in range(n) if v2 != u2 and adj[u2][v2] == 0]
                            # Check if adding (u2, v2) creates C4
                            valid_cands = []
                            for v2 in cands:
                                common = [w for w in range(n) if adj[u2][w] and adj[v2][w]]
                                if len(common) <= 1:
                                    valid_cands.append(v2)
                            if valid_cands:
                                v2 = random.choice(valid_cands)
                                adj[u2][v2] = 1
                                adj[v2][u2] = 1
                                edges.append([u2, v2])
                                degrees[u2] += 1
                                degrees[v2] += 1
                                if degrees[u2] >= min_deg:
                                    low.remove(u2)
                                repaired = True
                                break
                        if not repaired and min(degrees) < min_deg:
                            break # Restart
                            
            # Final check
            if min([sum(adj[i]) for i in range(n)]) >= min_deg:
                # Verify no C4
                has_c4 = False
                for u in range(n):
                    for v in range(u+1, n):
                        common = [w for w in range(n) if adj[u][w] and adj[v][w]]
                        if len(common) > 1:
                            has_c4 = True
                            break
                    if has_c4: break
                if not has_c4:
                    found = True
                    return {'n': n, 'edges': edges}
                    
    return {'n': 39, 'edges': []}
