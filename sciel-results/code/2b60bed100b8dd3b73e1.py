import time
import random

def construct():
    deadline = time.time() + 110.0
    
    q = 5
    n_base = q * q + q + 1
    
    points = []
    for x in range(q):
        for y in range(q):
            points.append([x, y, 1])
    for x in range(q):
        points.append([x, 1, 0])
    
    adj = [set() for _ in range(n_base)]
    edges = []
    for i in range(n_base):
        for j in range(i + 1, n_base):
            pi, pj = points[i], points[j]
            val = (pi[0] * pj[1] + pi[1] * pj[0] + pi[2] * pj[2]) % q
            if val == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
    
    n = n_base
    target = 43
    
    # Greedy addition
    for _ in range(target - n_base):
        if time.time() > deadline:
            break
        
        # Try to add a vertex connected to a random subset
        added = False
        for size in range(0, 6):
            if time.time() > deadline:
                break
            for _ in range(200):
                if time.time() > deadline:
                    break
                if size == 0:
                    neigh = []
                else:
                    neigh = random.sample(range(n), size)
                
                # Check C4 constraint
                valid = True
                for a in range(len(neigh)):
                    for b in range(a+1, len(neigh)):
                        if adj[neigh[a]] & adj[neigh[b]]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Add temporarily
                adj.append(set())
                for v in neigh:
                    adj[v].add(n)
                    adj[n].add(v)
                
                # Check independence number (greedy lower bound)
                # If greedy finds an independent set of size 11, reject
                remaining = set(range(n+1))
                indep = 0
                while remaining:
                    # Pick min degree vertex
                    min_v = min(remaining, key=lambda v: len(adj[v] & remaining))
                    indep += 1
                    remaining.discard(min_v)
                    remaining -= adj[min_v]
                    if indep >= 11:
                        break
                
                if indep < 11:
                    # Accept
                    n += 1
                    for v in neigh:
                        edges.append([v, n-1])
                    added = True
                    break
                else:
                    # Revert
                    adj.pop()
                    for v in neigh:
                        adj[v].discard(n)
                        adj[n].discard(v)
            
            if added:
                break
        
        if not added:
            break
    
    return {'n': n, 'edges': edges}
