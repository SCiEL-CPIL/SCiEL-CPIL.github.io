import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 150.0
    n = 20
    best_edges = []
    best_count = 0
    
    def is_k33_free(adj):
        # Check if graph with adjacency matrix adj is K3,3 free
        # Condition: For all triples {u,v,w}, |N(u) & N(v) & N(w)| <= 2
        for u in range(n):
            for v in range(u+1, n):
                for w in range(v+1, n):
                    common = 0
                    for x in range(n):
                        if adj[u][x] and adj[v][x] and adj[w][x]:
                            common += 1
                    if common > 2:
                        return False
        return True

    def count_edges(adj):
        return sum(sum(row) for row in adj) // 2

    def greedy_construction(seed):
        random.seed(seed)
        adj = [[0]*n for _ in range(n)]
        edges_to_try = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            # Check if adding (u, v) creates a K3,3
            # A K3,3 is formed if there are x,y,z in N(u) ∩ N(v) such that 
            # |N(x) ∩ N(y) ∩ N(z)| >= 3 (including u and v? No, u,v are in N(x)...)
            # Let S = N(u) ∩ N(v)
            # If we add (u,v), then u and v become common neighbors of any x,y,z in S.
            # So for any x,y,z in S, the new common neighborhood size is 
            # old_size(x,y,z) + (1 if u not in old) + (1 if v not in old).
            # Since u,v are in N(x) for x in S, u and v are already in N(x).
            # Wait, if x in N(u) ∩ N(v), then u is in N(x) and v is in N(x).
            # So u and v are ALREADY common neighbors of x,y,z.
            # Adding (u,v) does NOT change N(x) for x in S.
            # It DOES change N(u) and N(v).
            # Does it change the common neighborhood of any triple?
            # The triple must include u or v?
            # If the triple is {u, a, b}, then N(u) changes.
            # If the triple is {x, y, z} not containing u or v, N(x), N(y), N(z) don't change.
            # So only triples containing u or v are affected.
            
            # Case 1: Triple contains u. {u, a, b}.
            # New N(u) = Old N(u) + {v}.
            # Common neighbors of {u, a, b} are N(u) ∩ N(a) ∩ N(b).
            # New intersection = (Old N(u) ∩ N(a) ∩ N(b)) ∪ ({v} ∩ N(a) ∩ N(b)).
            # So we add v to the common neighbors if v is in N(a) and N(b).
            # So we must check if adding v to the common neighbors of {u, a, b} makes size > 2.
            # This happens if v is in N(a) ∩ N(b) AND |Old N(u) ∩ N(a) ∩ N(b)| >= 2.
            
            # Case 2: Triple contains v. Symmetric.
            
            # Case 3: Triple contains both u and v. {u, v, a}.
            # New N(u) has v, New N(v) has u.
            # Common neighbors = N(u) ∩ N(v) ∩ N(a).
            # Old common neighbors = (N(u)\{v}) ∩ (N(v)\{u}) ∩ N(a).
            # New common neighbors includes u? No, u is not in N(u). v is not in N(v).
            # So u and v are NOT in the common neighborhood of {u, v, a}.
            # So the common neighborhood does not change?
            # Wait. N(u) contains v. N(v) contains u.
            # x is in N(u) ∩ N(v) ∩ N(a) if x is neighbor of u, v, a.
            # This set is independent of the edge (u,v) itself?
            # Yes, because u is not a neighbor of itself, v is not a neighbor of itself.
            # So adding (u,v) does NOT change the common neighbors of triples containing u or v?
            # Let's re-evaluate.
            # Common neighbors of {u, a, b} are vertices x such that (u,x), (a,x), (b,x) are edges.
            # If we add (u,v), does v become a common neighbor of {u, a, b}?
            # v is a neighbor of u (new edge).
            # Is v a neighbor of a? Check adj[v][a].
            # Is v a neighbor of b? Check adj[v][b].
            # If yes, then v is a new common neighbor.
            # So we need to check if |Old Common Neighbors| + (1 if v in N(a)∩N(b) else 0) > 2.
            
            safe = True
            # Check triples containing u
            for a in range(n):
                if a == u or not adj[u][a]: continue
                for b in range(a+1, n):
                    if b == u or not adj[u][b]: continue
                    # Check if v is in N(a) and N(b)
                    if adj[v][a] and adj[v][b]:
                        # Count old common neighbors of u, a, b
                        cnt = 0
                        for x in range(n):
                            if x == v: continue # v is the new one, check old ones
                            if adj[u][x] and adj[a][x] and adj[b][x]:
                                cnt += 1
                        if cnt >= 2:
                            safe = False
                            break
                if not safe: break
            if not safe: continue
            
            # Check triples containing v
            for a in range(n):
                if a == v or not adj[v][a]: continue
                for b in range(a+1, n):
                    if b == v or not adj[v][b]: continue
                    # Check if u is in N(a) and N(b)
                    if adj[u][a] and adj[u][b]:
                        cnt = 0
                        for x in range(n):
                            if x == u: continue
                            if adj[v][x] and adj[a][x] and adj[b][x]:
                                cnt += 1
                        if cnt >= 2:
                            safe = False
                            break
                if not safe: break
            
            if safe:
                adj[u][v] = 1
                adj[v][u] = 1
        
        # Extract edges
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    edges.append([u, v])
        return edges

    # Run multiple random restarts
    num_restarts = 0
    while time.time() < deadline - 5.0:
        seed = random.randint(0, 10**6)
        edges = greedy_construction(seed)
        cnt = len(edges)
        if cnt > best_count:
            best_count = cnt
            best_edges = edges
        num_restarts += 1
        # If we found a very good solution, we can try local search
        # But greedy with random shuffle is usually quite good for this size
        
    return best_edges
