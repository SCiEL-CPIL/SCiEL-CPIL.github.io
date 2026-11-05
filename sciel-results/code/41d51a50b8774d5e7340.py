import time
import random
import sys
import numpy as np

def construct():
    n = 16
    target = 65
    deadline = time.time() + 110.0  # safety margin under 120s
    rng = random.Random(12345)
    
    # Precompute all triangles
    triangles = []
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                triangles.append((u, v, w))
    num_tri = len(triangles)
    
    def _score(adj, deg):
        best = 0
        for (u, v, w) in triangles:
            if adj[u][v] and adj[v][w] and adj[u][w]:
                s = deg[u] + deg[v] + deg[w]
                if s > best:
                    best = s
        return best
    
    def _edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    edges.append([u, v])
        return edges
    
    def _valid(edges, n, target):
        if len(edges) != target:
            return False
        seen = set()
        for u, v in edges:
            if u == v or u < 0 or v < 0 or u >= n or v >= n:
                return False
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in seen:
                return False
            seen.add((a, b))
        return True
    
    def _seed_edges(n):
        # K_{8,8} + one intra edge
        edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
        return edges
    
    best_edges = _seed_edges(n)
    best_score = 100
    
    def try_improve(edges, score):
        nonlocal best_edges, best_score
        if score < best_score:
            best_score = score
            best_edges = list(edges)
            return True
        return False
    
    def local_search(adj, deg, score):
        nonlocal best_score, best_edges
        try_improve(_edges_from_adj(adj), score)
        
        max_iter = 500
        for _ in range(max_iter):
            if time.time() >= deadline:
                break
            
            # Find worst triangle
            worst_tri = None
            worst_s = 0
            for (u, v, w) in triangles:
                if adj[u][v] and adj[v][w] and adj[u][w]:
                    s = deg[u] + deg[v] + deg[w]
                    if s > worst_s:
                        worst_s = s
                        worst_tri = (u, v, w)
            
            if worst_tri is None:
                break
            
            u, v, w = worst_tri
            improved = False
            
            # Try removing each edge of the worst triangle and adding a new one
            for a, b in [(u, v), (v, w), (u, w)]:
                # Find candidates: non-adjacent pairs (c,d) where adding (c,d) and removing (a,b) improves
                # c must not be adjacent to a, d must not be adjacent to b (to keep it a swap)
                # Actually, just try all non-edges (c,d) where c!=a,b and d!=a,b
                best_new_score = score
                best_swap = None
                
                # Only try a subset for speed
                candidates = []
                for c in range(n):
                    if c == a or c == b:
                        continue
                    if adj[a][c]:  # c is already adjacent to a, so removing (a,b) doesn't help c
                        continue
                    for d in range(c + 1, n):
                        if d == a or d == b:
                            continue
                        if adj[c][d]:
                            continue
                        # Also ensure (c,d) doesn't create an immediate problem
                        # Quick check: new degrees
                        nd = deg[:]
                        nd[a] -= 1
                        nd[b] -= 1
                        nd[c] += 1
                        nd[d] += 1
                        
                        # Check if this creates any triangle with sum > worst_s
                        # Only check triangles involving the new edge (c,d)
                        ok = True
                        for x in range(n):
                            if x == c or x == d:
                                continue
                            if adj[c][x] and adj[d][x]:
                                s_new = nd[c] + nd[d] + nd[x]
                                if s_new > worst_s:
                                    ok = False
                                    break
                        if not ok:
                            continue
                        
                        candidates.append((c, d))
                
                # Shuffle and try a few
                rng.shuffle(candidates)
                for c, d in candidates[:5]:
                    nd = deg[:]
                    nd[a] -= 1
                    nd[b] -= 1
                    nd[c] += 1
                    nd[d] += 1
                    
                    na = [row[:] for row in adj]
                    na[a][b] = na[b][a] = False
                    na[c][d] = na[d][c] = True
                    
                    ns = _score(na, nd)
                    if ns < best_new_score:
                        best_new_score = ns
                        best_swap = (a, b, c, d, nd, na)
                
                if best_swap:
                    a2, b2, c2, d2, nd2, na2 = best_swap
                    adj = na2
                    deg = nd2
                    score = best_new_score
                    improved = True
                    try_improve(_edges_from_adj(adj), score)
                    break
            
            if not improved:
                # Try random edge swaps
                all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
                rng.shuffle(all_pairs)
                found = False
                for (a, b) in all_pairs:
                    if not adj[a][b]:
                        continue
                    for (c, d) in all_pairs:
                        if adj[c][d] or (c, d) == (a, b) or (c, d) == (b, a):
                            continue
                        if a in (c, d) or b in (c, d):
                            continue
                        nd = deg[:]
                        nd[a] -= 1
                        nd[b] -= 1
                        nd[c] += 1
                        nd[d] += 1
                        na = [row[:] for row in adj]
                        na[a][b] = na[b][a] = False
                        na[c][d] = na[d][c] = True
                        ns = _score(na, nd)
                        if ns < score:
                            adj = na
                            deg = nd
                            score = ns
                            try_improve(_edges_from_adj(adj), score)
                            found = True
                            break
                    if found:
                        break
                if not found:
                    break
        
        return adj, deg, score
    
    # Start with seed
    adj = [[False] * n for _ in range(n)]
    deg = [0] * n
    for u, v in best_edges:
        adj[u][v] = adj[v][u] = True
        deg[u] += 1
        deg[v] += 1
    score = _score(adj, deg)
    adj, deg, score = local_search(adj, deg, score)
    
    # Try diverse initializations
    # 1. K_{7,9} + 2 intra edges
    for _ in range(5):
        if time.time() >= deadline:
            break
        
        # Random bipartition + intra edges
        part_a = list(range(n))
        rng.shuffle(part_a)
        split = rng.randint(6, 10)
        a = part_a[:split]
        b = part_a[split:]
        
        adj = [[False] * n for _ in range(n)]
        deg = [0] * n
        edges = []
        
        # Cross edges
        for u in a:
            for v in b:
                adj[u][v] = adj[v][u] = True
                deg[u] += 1
                deg[v] += 1
                edges.append([u, v])
        
        # Add intra edges to reach target
        intra_needed = target - len(edges)
        intra_candidates = []
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                intra_candidates.append((a[i], a[j]))
        for i in range(len(b)):
            for j in range(i + 1, len(b)):
                intra_candidates.append((b[i], b[j]))
        
        rng.shuffle(intra_candidates)
        for u, v in intra_candidates:
            if intra_needed <= 0:
                break
            if not adj[u][v]:
                adj[u][v] = adj[v][u] = True
                deg[u] += 1
                deg[v] += 1
                edges.append([u, v])
                intra_needed -= 1
        
        if len(edges) != target:
            continue
        
        score = _score(adj, deg)
        adj, deg, score = local_search(adj, deg, score)
    
    # 2. Random dense graphs
    for _ in range(10):
        if time.time() >= deadline:
            break
        
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_pairs)
        edges = [list(p) for p in all_pairs[:target]]
        
        adj = [[False] * n for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            adj[u][v] = adj[v][u] = True
            deg[u] += 1
            deg[v] += 1
        
        score = _score(adj, deg)
        adj, deg, score = local_search(adj, deg, score)
    
    if not _valid(best_edges, n, target):
        best_edges = _seed_edges(n)
        best_score = 100
        adj = [[False] * n for _ in range(n)]
        deg = [0] * n
        for u, v in best_edges:
            adj[u][v] = adj[v][u] = True
            deg[u] += 1
            deg[v] += 1
        best_score = _score(adj, deg)
    
    return {'n': n, 'edges': best_edges}
