"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start = time.time()
    deadline = start + 100
    n = 16
    
    # Q3 edges in canonical form (8 vertices, 12 edges)
    q3e = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    
    # Q3 adjacency
    q3_adj = {v: [] for v in range(8)}
    for a, b in q3e:
        q3_adj[a].append(b)
        q3_adj[b].append(a)
    
    # Start with K_{3,13}
    edges = set()
    for a in range(3):
        for b in range(3, 16):
            edges.add((a, b))
    
    def creates_q3(edge_set, new_edge):
        """Check if adding new_edge to edge_set creates a Q3 subgraph."""
        u, v = new_edge
        # For each Q3 edge, try both orientations
        for a, b in q3e:
            for perm in [(a, b), (b, a)]:
                pa, pb = perm
                assign = {pa: u, pb: v}
                used = {u, v}
                remaining = [i for i in range(8) if i not in (pa, pb)]
                
                def bt(idx):
                    if time.time() > deadline:
                        return None
                    if idx == len(remaining):
                        for ea, eb in q3e:
                            x, y = assign[ea], assign[eb]
                            if (x, y) == new_edge or (y, x) == new_edge:
                                continue
                            if (min(x,y), max(x,y)) not in edge_set:
                                return None
                        return True
                    qv = remaining[idx]
                    for cand in range(n):
                        if cand in used:
                            continue
                        ok = True
                        for nb in q3_adj[qv]:
                            if nb in assign:
                                x = assign[nb]
                                if (cand, x) == new_edge or (x, cand) == new_edge:
                                    continue
                                if (min(cand, x), max(cand, x)) not in edge_set:
                                    ok = False
                                    break
                        if ok:
                            assign[qv] = cand
                            used.add(cand)
                            res = bt(idx + 1)
                            if res:
                                return True
                            del assign[qv]
                            used.remove(cand)
                    return False
                
                if bt(0):
                    return True
        return False
    
    # Try to add missing edges
    all_edges = set((i, j) for i in range(n) for j in range(i+1, n))
    missing = list(all_edges - edges)
    random.seed(42)
    random.shuffle(missing)
    
    for e in missing:
        if time.time() > deadline - 5:
            break
        if creates_q3(edges, e):
            continue
        edges.add(e)
    
    return [[u, v] for u, v in edges]
# EVOLVE-BLOCK-END
