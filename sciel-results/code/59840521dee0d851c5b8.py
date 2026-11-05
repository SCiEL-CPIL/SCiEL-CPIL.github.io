import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    n = 16
    
    # Fast Q3 check: does adding edge (u,v) to graph adj create a Q3?
    # In Q3 with edge (0,1)->(u,v):
    #   u's other Q3-neighbors: 2,3 (must be in adj[u], distinct from v)
    #   v's other Q3-neighbors: 4,5 (must be in adj[v], distinct from u)
    #   6 is neighbor of 2,4,7; 7 is neighbor of 3,5,6
    #   Edges: (2,3)no, (2,4)yes, (2,6)yes, (3,5)yes, (3,6)yes, (4,5)no, (4,7)yes, (5,7)yes, (6,7)yes
    #   Also (0,2),(0,3),(1,4),(1,5) already satisfied by u,v connections
    #   And (0,1) is the new edge
    #
    # So we need: a,b in adj[u]\{v}, c,d in adj[v]\{u}, all distinct
    #   adj[a][c] (2-3? No! 2 and 3 are not adjacent in Q3)
    #   Wait, let me recheck Q3 edges:
    #   Q3 vertices 0-7, edges: (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
    #   
    #   So if 0->u, 1->v:
    #   0's neighbors: 1(v), 2, 4 -> so img(2), img(4) in adj[u]
    #   1's neighbors: 0(u), 3, 5 -> so img(3), img(5) in adj[v]
    #   Edges to check:
    #   (2,3): img(2)-img(3)
    #   (2,6): img(2)-img(6)
    #   (3,7): img(3)-img(7)
    #   (4,5): img(4)-img(5)
    #   (4,6): img(4)-img(6)
    #   (5,7): img(5)-img(7)
    #   (6,7): img(6)-img(7)
    #
    #   So: a=img(2), b=img(4), c=img(3), d=img(5), f=img(6), g=img(7)
    #   a in adj[u], b in adj[u], c in adj[v], d in adj[v]
    #   a,b,c,d,f,g all distinct, not u,v
    #   adj[a][c], adj[a][f], adj[c][g], adj[b][d], adj[b][f], adj[d][g], adj[f][g]
    
    def creates_q3(adj, u, v):
        """Check if adding edge (u,v) creates a Q3 subgraph."""
        # Get candidate neighbors
        # a, b must be in adj[u] (excluding v since not yet added)
        # c, d must be in adj[v] (excluding u)
        
        neu = [x for x in adj[u] if x != v]
        nev = [x for x in adj[v] if x != u]
        
        if len(neu) < 2 or len(nev) < 2:
            return False
        
        # Try all pairs (a,b) from neu and (c,d) from nev
        # a=img(2), b=img(4), c=img(3), d=img(5)
        for a in neu:
            for b in neu:
                if a == b:
                    continue
                for c in nev:
                    if c in (a, b):
                        continue
                    if not adj[a][c]:  # edge (2,3)
                        continue
                    for d in nev:
                        if d in (a, b, c):
                            continue
                        if not adj[b][d]:  # edge (4,5)
                            continue
                        # Now need f=img(6), g=img(7)
                        # f in adj[a] and adj[b] (edges 2-6, 4-6)
                        # g in adj[c] and adj[d] and adj[f] (edges 3-7, 5-7, 6-7)
                        # f,g not in {u,v,a,b,c,d}
                        
                        # Candidates for f: common neighbors of a and b
                        for f in adj[a]:
                            if f in (u, v, a, b, c, d):
                                continue
                            if not adj[b][f]:
                                continue
                            # Check if exists g in adj[c] & adj[d] & adj[f], not forbidden
                            for g in adj[c]:
                                if g in (u, v, a, b, c, d, f):
                                    continue
                                if adj[d][g] and adj[f][g]:
                                    return True
        return False
    
    def count_edges(adj):
        return sum(len(s) for s in adj) // 2
    
    def make_edge_list(adj):
        result = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    result.append([u, v])
        return result
    
    best_edges = 0
    best_adj = None
    
    # Strategy 1: Start from K_{3,13} base and add edges greedily
    # Part A: {0,1,2}, Part B: {3,...,15}
    def try_k313_base():
        adj = [set() for _ in range(n)]
        for u in range(3):
            for v in range(3, 16):
                adj[u].add(v)
                adj[v].add(u)
        
        # Missing edges: within A (3 edges) and within B (78 edges)
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if j not in adj[i]:
                    missing.append((i, j))
        
        # Sort by some heuristic: prefer edges within B first (more "room")
        # Actually, let's try multiple orderings
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            
            # Create a shuffled version
            shuffled = missing[:]
            if trial == 0:
                # Deterministic: within B first, then within A
                shuffled.sort(key=lambda e: (e[0] >= 3, e[1] >= 3, e[0], e[1]))
            elif trial == 1:
                # Within A first
                shuffled.sort(key=lambda e: (e[0] < 3, e[1] < 3, e[0], e[1]))
            else:
                random.shuffle(shuffled)
            
            # Work on a copy
            work_adj = [set(s) for s in adj]
            
            for u, v in shuffled:
                if time.time() > deadline - 5:
                    break
                if not creates_q3(work_adj, u, v):
                    work_adj[u].add(v)
                    work_adj[v].add(u)
            
            ne = count_edges(work_adj)
            if ne > best_edges:
                best_edges = ne
                best_adj = [set(s) for s in work_adj]
    
    # Strategy 2: Start from empty, random greedy (current approach, improved)
    def try_random_greedy():
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        for trial in range(100):
            if time.time() > deadline - 5:
                break
            
            adj = [set() for _ in range(n)]
            shuffled = all_edges[:]
            random.shuffle(shuffled)
            
            for u, v in shuffled:
                if time.time() > deadline - 5:
                    break
                if not creates_q3(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
            
            ne = count_edges(adj)
            if ne > best_edges:
                best_edges = ne
                best_adj = [set(s) for s in adj]
    
    # Strategy 3: Start from a random dense-ish graph and try to add edges
    def try_dense_start():
        for trial in range(30):
            if time.time() > deadline - 5:
                break
            
            # Start with a random graph of moderate density
            adj = [set() for _ in range(n)]
            all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(all_edges)
            
            # Add ~60% of edges randomly
            num_to_add = int(len(all_edges) * 0.6)
            for i in range(num_to_add):
                u, v = all_edges[i]
                adj[u].add(v)
                adj[v].add(u)
            
            # Now try to add remaining edges greedily
            remaining = all_edges[num_to_add:]
            random.shuffle(remaining)
            
            for u, v in remaining:
                if time.time() > deadline - 5:
                    break
                if not creates_q3(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
            
            ne = count_edges(adj)
            if ne > best_edges:
                best_edges = ne
                best_adj = [set(s) for s in adj]
    
    # Run strategies
    try_k313_base()
    try_random_greedy()
    try_dense_start()
    
    # Final: try local search - remove an edge, try to add multiple others
    if best_adj is not None and time.time() < deadline - 10:
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            work_adj = [set(s) for s in best_adj]
            ne = count_edges(work_adj)
            
            # Try removing random edges and see if we can add more
            all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
            present_edges = [(u, v) for u, v in all_edges if v in work_adj[u]]
            missing_edges = [(u, v) for u, v in all_edges if v not in work_adj[u]]
            
            # Try removing 1-2 edges and re-adding greedily
            for num_remove in [1, 2]:
                if time.time() > deadline - 5:
                    break
                if len(present_edges) < num_remove:
                    continue
                
                to_remove = random.sample(present_edges, num_remove)
                for u, v in to_remove:
                    work_adj[u].discard(v)
                    work_adj[v].discard(u)
                
                # Now greedily add all possible missing edges
                shuffled_missing = missing_edges[:]
                random.shuffle(shuffled_missing)
                
                for u, v in shuffled_missing:
                    if time.time() > deadline - 5:
                        break
                    if v not in work_adj[u] and not creates_q3(work_adj, u, v):
                        work_adj[u].add(v)
                        work_adj[v].add(u)
                
                new_ne = count_edges(work_adj)
                if new_ne > best_edges:
                    best_edges = new_ne
                    best_adj = [set(s) for s in work_adj]
    
    if best_adj is None:
        # Fallback
        return [[a, b] for a in range(3) for b in range(3, 16)]
    
    return make_edge_list(best_adj)
