import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n = 16
    # Seed graph
    edges = set()
    A = [0, 1, 2]
    B = list(range(3, 16))
    for a in A:
        for b in B:
            edges.add((a, b))
    for i in range(3):
        for j in range(i + 1, 3):
            edges.add((i, j))
            
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def creates_q3_with_edge(u, v):
        # Check if adding (u, v) creates a Q3 containing (u, v)
        # Assume u -> 0, v -> 1 in Q3
        # Neighbors of u in G: N_u
        # Neighbors of v in G: N_v
        # Note: (u, v) is not in G yet, but we are checking if adding it allows a Q3.
        # So N_u should include v? No, we are looking for other vertices.
        # The other vertices must be in V \ {u, v}.
        
        N_u = adj[u]
        N_v = adj[v]
        
        # Candidates for 2 and 4: neighbors of u
        # Candidates for 3 and 5: neighbors of v
        
        # We need distinct x2, x4 from N_u \ {v}
        # We need distinct x3, x5 from N_v \ {u}
        # And all 6 distinct from u, v and each other.
        
        cand_24 = [x for x in N_u if x != v]
        cand_35 = [x for x in N_v if x != u]
        
        for x2 in cand_24:
            for x4 in cand_24:
                if x2 == x4: continue
                for x3 in cand_35:
                    if x3 in (x2, x4): continue
                    for x5 in cand_35:
                        if x5 in (x2, x4, x3): continue
                        
                        # Check x2 ~ x3
                        if x3 not in adj[x2]: continue
                        # Check x4 ~ x5
                        if x5 not in adj[x4]: continue
                        
                        # Find x6: neighbor of x2 and x4
                        # x6 must be distinct from u, v, x2, x3, x4, x5
                        # x6 in N(x2) & N(x4)
                        # x7 in N(x3) & N(x5)
                        # And x6 ~ x7
                        
                        common_6 = adj[x2] & adj[x4]
                        common_7 = adj[x3] & adj[x5]
                        
                        # Filter out used vertices
                        used = {u, v, x2, x3, x4, x5}
                        cand_6 = [x for x in common_6 if x not in used]
                        cand_7 = [x for x in common_7 if x not in used]
                        
                        for x6 in cand_6:
                            for x7 in cand_7:
                                if x6 == x7: continue
                                if x7 in adj[x6]:
                                    return True
        return False

    candidates = [(u, v) for u in B for v in B if u < v and (u, v) not in edges]
    random.shuffle(candidates)
    
    for u, v in candidates:
        if time.time() > deadline - 10:
            break
        if not creates_q3_with_edge(u, v):
            edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
            
    return [list(e) for e in edges]
