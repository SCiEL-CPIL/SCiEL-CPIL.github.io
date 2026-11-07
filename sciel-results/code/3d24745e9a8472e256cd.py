import time
import random

_DEADLINE = 140.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_hit():
    return _elapsed() > _DEADLINE

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency
        adj_c = [0] * k
        adj_i = [0] * (n - k)
        
        for (u, v) in edges:
            if u < k and v < k:
                adj_c[u] |= (1 << v)
                adj_c[v] |= (1 << u)
            elif u < k and v >= k:
                idx_v = v - k
                adj_i[idx_v] |= (1 << u)
            elif u >= k and v < k:
                idx_v = u - k
                adj_i[idx_v] |= (1 << v)

        # Track which edges are covered
        # For core edges: core_edge_covered[u][v] for u < v < k
        # For cross edges: cross_edge_covered[idx][u] for idx in I, u in C
        core_covered = [[False] * k for _ in range(k)]
        cross_covered = [[False] * k for _ in range(n - k)]
        
        cliques = []
        
        def add_clique(clique):
            """Add a clique and mark its edges as covered."""
            cliques.append(clique)
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    if a < k and b < k:
                        if a > b: a, b = b, a
                        core_covered[a][b] = True
                    elif a < k and b >= k:
                        cross_covered[b - k][a] = True
                    elif a >= k and b < k:
                        cross_covered[a - k][b] = True
        
        def is_core_edge_uncovered(u, v):
            if u > v: u, v = v, u
            return not core_covered[u][v]
        
        def is_cross_edge_uncovered(idx, u):
            return not cross_covered[idx][u]
        
        def count_uncovered_core_edges():
            cnt = 0
            for u in range(k):
                for v in range(u + 1, k):
                    if not core_covered[u][v]:
                        cnt += 1
            return cnt
        
        def count_uncovered_cross_edges(idx):
            cnt = 0
            for u in range(k):
                if not cross_covered[idx][u] and ((adj_i[idx] >> u) & 1):
                    cnt += 1
            return cnt
        
        def total_uncovered_edges():
            cnt = count_uncovered_core_edges()
            for idx in range(n - k):
                cnt += count_uncovered_cross_edges(idx)
            return cnt
        
        # Greedy: repeatedly find the best clique to add
        # A good heuristic: find a clique that covers the most uncovered edges
        # Since k is small (<=30), we can enumerate candidate cliques
        
        def find_best_clique():
            """Find a clique that covers the most uncovered edges."""
            best_clique = None
            best_gain = 0
            
            # Try all pairs as seeds, then extend
            # For efficiency, only consider pairs with uncovered edges
            for u in range(k):
                for v in range(u + 1, k):
                    if not is_core_edge_uncovered(u, v):
                        continue
                    
                    # Try to extend {u, v} to a maximal clique
                    # A clique must be a clique in the original graph
                    # Check if u and v are adjacent (they are, since it's a core edge)
                    
                    # Build candidate set: vertices w such that w is adjacent to both u and v
                    # and adding w doesn't create a non-clique
                    cand = (1 << u) | (1 << v)
                    # w must be adjacent to all in cand
                    for w in range(k):
                        if w == u or w == v:
                            continue
                        if (adj_c[u] >> w) & 1 and (adj_c[v] >> w) & 1:
                            # Check w is adjacent to all other candidates
                            ok = True
                            temp = cand
                            while temp:
                                lsb = temp & -temp
                                x = lsb.bit_length() - 1
                                temp ^= lsb
                                if x != u and x != v and not ((adj_c[w] >> x) & 1):
                                    ok = False
                                    break
                            if ok:
                                cand |= (1 << w)
                    
                    # Now cand is a maximal clique containing u,v
                    # Check how many uncovered edges it covers
                    gain = 0
                    verts = []
                    temp = cand
                    while temp:
                        lsb = temp & -temp
                        x = lsb.bit_length() - 1
                        verts.append(x)
                        temp ^= lsb
                    
                    for i in range(len(verts)):
                        for j in range(i + 1, len(verts)):
                            a, b = verts[i], verts[j]
                            if a < k and b < k:
                                if a > b: a, b = b, a
                                if not core_covered[a][b]:
                                    gain += 1
                    
                    # Also consider adding independent vertices
                    # For each independent vertex idx, check if it's adjacent to all in cand
                    # and if adding it covers more uncovered cross edges
                    for idx in range(n - k):
                        # Check if idx is adjacent to all vertices in cand
                        ok = True
                        temp = cand
                        while temp:
                            lsb = temp & -temp
                            x = lsb.bit_length() - 1
                            temp ^= lsb
                            if not ((adj_i[idx] >> x) & 1):
                                ok = False
                                break
                        if not ok:
                            continue
                        
                        # Count uncovered cross edges from idx to cand
                        cross_gain = 0
                        temp = cand
                        while temp:
                            lsb = temp & -temp
                            x = lsb.bit_length() - 1
                            temp ^= lsb
                            if not cross_covered[idx][x]:
                                cross_gain += 1
                        
                        total_gain = gain + cross_gain
                        if total_gain > best_gain:
                            best_gain = total_gain
                            best_clique = verts + [k + idx]
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_clique = verts[:]
            
            return best_clique, best_gain
        
        # Also try cliques involving only independent vertices (size 2: {idx, u})
        # These are needed if no larger clique covers the cross edge
        
        # Process: repeatedly find best clique until no more uncovered edges
        max_iterations = 10000
        for _ in range(max_iterations):
            if _deadline_hit():
                break
            
            if total_uncovered_edges() == 0:
                break
            
            clique, gain = find_best_clique()
            
            if clique is None or gain == 0:
                # No clique found via the above method; use fallback
                # Find any uncovered edge and make it a 2-clique
                found = False
                # Try core edges
                for u in range(k):
                    for v in range(u + 1, k):
                        if not core_covered[u][v] and ((adj_c[u] >> v) & 1):
                            add_clique([u, v])
                            found = True
                            break
                    if found:
                        break
                if found:
                    continue
                
                # Try cross edges
                for idx in range(n - k):
                    for u in range(k):
                        if not cross_covered[idx][u] and ((adj_i[idx] >> u) & 1):
                            add_clique([k + idx, u])
                            found = True
                            break
                    if found:
                        break
                if found:
                    continue
                
                # Should not reach here
                break
            
            add_clique(clique)
        
        # Fallback: cover any remaining uncovered edges with 2-cliques
        if not _deadline_hit() or True:
            # Core edges
            for u in range(k):
                for v in range(u + 1, k):
                    if not core_covered[u][v] and ((adj_c[u] >> v) & 1):
                        add_clique([u, v])
            
            # Cross edges
            for idx in range(n - k):
                for u in range(k):
                    if not cross_covered[idx][u] and ((adj_i[idx] >> u) & 1):
                        add_clique([k + idx, u])
        
        # Verify all edges are covered
        for u in range(k):
            for v in range(u + 1, k):
                if ((adj_c[u] >> v) & 1) and not core_covered[u][v]:
                    # Should not happen
                    pass
        
        # Convert to proper format
        result = []
        for c in cliques:
            result.append([int(x) for x in c])
        return result

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    out = []
    for spec in specs:
        if _deadline_hit():
            out.append([list(e) for e in instance(*spec)])
        else:
            out.append(_solve_split(*spec))
    return out
