import time
import random

_DEADLINE = 130.0
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
        adj_core = [0] * k
        adj_cross = [0] * (n - k)
        
        core_edges = []
        cross_edges = []
        
        for (u, v) in edges:
            if u < k and v < k:
                adj_core[u] |= (1 << v)
                adj_core[v] |= (1 << u)
                core_edges.append((u, v))
            else:
                if u >= k:
                    u, v = v, u
                idx_v = v - k
                adj_cross[idx_v] |= (1 << u)
                cross_edges.append((u, v))

        # Strategy:
        # 1. For each independent vertex v, find the largest clique S in the core subgraph induced by N(v).
        # 2. Form clique S ∪ {v}.
        # 3. Cover remaining cross edges (u, v) with 2-cliques.
        # 4. Cover remaining core edges with 2-cliques.
        
        covered_core = set()
        covered_cross = set()
        
        cliques = []
        
        # Sort independent vertices by potential (number of core edges they can cover)
        vs_sorted = sorted(range(n - k), key=lambda idx: sum(1 for e in core_edges if (adj_cross[idx] >> e[0]) & 1 and (adj_cross[idx] >> e[1]) & 1), reverse=True)
        
        for idx_v in vs_sorted:
            if _deadline_hit():
                break
            mask = adj_cross[idx_v]
            
            # Find a maximal clique in the core subgraph induced by N(v)
            # Vertices in N(v) are those core vertices adjacent to v
            # Edges are core edges between two vertices in N(v)
            
            # Get the set of core vertices in N(v)
            n_v = []
            for u in range(k):
                if (mask >> u) & 1:
                    n_v.append(u)
            
            # If |N(v)| < 2, we can't form a clique of size >= 2 in the core
            if len(n_v) < 2:
                # Just cover cross edges with 2-cliques
                for u in n_v:
                    if (u, k + idx_v) not in covered_cross:
                        cliques.append([u, k + idx_v])
                        covered_cross.add((u, k + idx_v))
                continue
            
            # Find a maximal clique in the subgraph induced by n_v
            # Use a greedy approach: start with the vertex with highest degree in the induced subgraph
            # and iteratively add vertices that are adjacent to all current clique members
            
            # Compute degrees in the induced subgraph
            deg = {u: 0 for u in n_v}
            for u in n_v:
                for v in n_v:
                    if u != v and (adj_core[u] >> v) & 1:
                        deg[u] += 1
            
            # Start with the vertex of highest degree
            if not n_v:
                continue
                
            start = max(n_v, key=lambda u: deg[u])
            clique = [start]
            remaining = set(n_v) - {start}
            
            # Iteratively add vertices
            while remaining:
                # Find a vertex in remaining that is adjacent to all vertices in clique
                candidates = []
                for u in remaining:
                    is_adj = True
                    for c in clique:
                        if not ((adj_core[u] >> c) & 1):
                            is_adj = False
                            break
                    if is_adj:
                        candidates.append(u)
                
                if not candidates:
                    break
                
                # Pick the candidate with highest degree (in the original induced subgraph)
                best = max(candidates, key=lambda u: deg[u])
                clique.append(best)
                remaining.remove(best)
            
            # If the clique has size >= 2, form the big clique
            if len(clique) >= 2:
                big_clique = clique + [k + idx_v]
                cliques.append(big_clique)
                
                # Mark all core edges within clique as covered
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        u, v = clique[i], clique[j]
                        if u > v:
                            u, v = v, u
                        covered_core.add((u, v))
                
                # Mark all cross edges from clique vertices to v as covered
                for u in clique:
                    covered_cross.add((u, k + idx_v))
            
            # Cover remaining cross edges with 2-cliques
            for u in n_v:
                if (u, k + idx_v) not in covered_cross:
                    cliques.append([u, k + idx_v])
                    covered_cross.add((u, k + idx_v))
        
        # Cover remaining core edges with 2-cliques
        for (u, v) in core_edges:
            if (u, v) not in covered_core:
                cliques.append([u, v])
                covered_core.add((u, v))
        
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
