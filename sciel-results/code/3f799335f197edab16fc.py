"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split(n, k, p, seed):
        """
        Minimum clique partition for a split graph using greedy set packing.
        
        Strategy:
        1. For each I-vertex v, compute its C-neighbors N(v)
        2. A "bundled" clique for v covers all edges (v,u) for u in N(v) AND 
           all C-C edges among N(v)
        3. Select a set of I-vertices to bundle using greedy set packing
           (maximize C-C edges covered)
        4. Cover remaining C-C edges with additional cliques
        5. Non-bundled I-vertices get individual edge cliques
        """
        rng = Random(seed)
        
        # Determine neighbors in C for each vertex in I
        neighbors = {}
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            neighbors[v] = nbrs
        
        # All C-C edges
        all_cc_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                all_cc_edges.add((u, v))
        
        # For each I-vertex with >= 2 C-neighbors, compute which C-C edges it covers
        candidates = []
        for v in range(k, n):
            neigh = neighbors[v]
            if len(neigh) >= 2:
                neigh_list = sorted(neigh)
                edges_covered = set()
                for i in range(len(neigh_list)):
                    for j in range(i + 1, len(neigh_list)):
                        a, b = neigh_list[i], neigh_list[j]
                        e = (a, b) if a < b else (b, a)
                        edges_covered.add(e)
                # Weight: number of C-C edges covered (this saves pure C-cliques)
                weight = len(edges_covered)
                candidates.append((v, neigh_list, edges_covered, weight))
        
        # Sort by weight descending
        candidates.sort(key=lambda x: x[3], reverse=True)
        
        # Greedy set packing: select non-overlapping bundles
        selected = {}  # v -> (neigh_list, edges_covered)
        covered_cc = set()
        
        for v, neigh_list, edges_covered, weight in candidates:
            if (time.time() - t0) > DEADLINE:
                break
            # Check if any of its C-C edges are already covered
            if not (edges_covered & covered_cc):
                selected[v] = (neigh_list, edges_covered)
                covered_cc.update(edges_covered)
        
        # Build cliques
        cliques = []
        
        # Add bundled cliques: {v} ∪ N(v)
        for v in sorted(selected.keys()):
            neigh_list, _ = selected[v]
            clique = [v] + neigh_list
            cliques.append(clique)
        
        # Add non-bundled I-vertex edges as pairs
        for v in range(k, n):
            if v in selected:
                continue
            for u in neighbors[v]:
                cliques.append([u, v])
        
        # Cover remaining C-C edges
        remaining_cc = all_cc_edges - covered_cc
        
        if remaining_cc:
            # Greedy: repeatedly find largest clique in the remaining C-C edge subgraph
            # Since C is a clique, any subset of C is a clique.
            # We just need to partition the remaining C-C edges into cliques.
            # Greedy: pick a vertex, extend to maximal clique in remaining subgraph.
            
            remaining = set(remaining_cc)
            vertices_in_remaining = set()
            for (a, b) in remaining:
                vertices_in_remaining.add(a)
                vertices_in_remaining.add(b)
            
            while remaining:
                if (time.time() - t0) > DEADLINE:
                    # Fallback: add each remaining edge as a pair
                    for e in remaining:
                        cliques.append(list(e))
                    break
                
                # Find vertex with max degree in remaining subgraph
                deg = {v: 0 for v in vertices_in_remaining}
                for (a, b) in remaining:
                    deg[a] += 1
                    deg[b] += 1
                
                if not deg or max(deg.values()) == 0:
                    break
                
                start_v = max(deg.keys(), key=lambda x: deg[x])
                
                # Greedy max clique in remaining subgraph containing start_v
                # Candidates: neighbors of start_v in remaining subgraph
                candidates_v = set()
                for (a, b) in remaining:
                    if a == start_v:
                        candidates_v.add(b)
                    elif b == start_v:
                        candidates_v.add(a)
                
                # Sort candidates by degree in remaining subgraph
                def v_degree(v):
                    d = 0
                    for (a, b) in remaining:
                        if a == v or b == v:
                            d += 1
                    return d
                
                candidates_v = sorted(candidates_v, key=v_degree, reverse=True)
                
                clique = [start_v]
                selected_set = {start_v}
                
                for v in candidates_v:
                    # Check if v is connected to all in selected_set via remaining edges
                    can_add = True
                    for w in selected_set:
                        e = (min(v, w), max(v, w))
                        if e not in remaining:
                            can_add = False
                            break
                    if can_add:
                        clique.append(v)
                        selected_set.add(v)
                
                if len(clique) < 2:
                    # Just take one edge
                    e = next(iter(remaining))
                    clique = list(e)
                
                # Remove covered edges
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        u, v = clique[i], clique[j]
                        e = (min(u, v), max(u, v))
                        remaining.discard(e)
                
                cliques.append(clique)
        
        # Validate the solution
        adj = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u + 1, k):
                adj[u].add(v)
                adj[v].add(u)
        for v in range(k, n):
            for u in neighbors[v]:
                adj[u].add(v)
                adj[v].add(u)
        
        all_edges = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    all_edges.add((u, v))
        
        edge_to_clique = {}
        for i, clique in enumerate(cliques):
            if len(clique) < 2:
                return None
            for a_idx in range(len(clique)):
                for b_idx in range(a_idx + 1, len(clique)):
                    u, v = clique[a_idx], clique[b_idx]
                    # Check it's a real edge
                    if v not in adj[u]:
                        return None
                    edge = (min(u, v), max(u, v))
                    if edge in edge_to_clique:
                        return None
                    edge_to_clique[edge] = i
        
        if len(edge_to_clique) != len(all_edges):
            return None
        
        return cliques

    def solve_simple(n, k, p, seed):
        """Fallback: big C clique + one clique per non-isolated I-vertex."""
        rng = Random(seed)
        neighbors = {}
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            neighbors[v] = nbrs
        
        cliques = []
        if k >= 2:
            cliques.append(list(range(k)))
        
        for v in range(k, n):
            if neighbors[v]:
                cliques.append([v] + sorted(neighbors[v]))
        
        # Validate
        adj = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u + 1, k):
                adj[u].add(v)
                adj[v].add(u)
        for v in range(k, n):
            for u in neighbors[v]:
                adj[u].add(v)
                adj[v].add(u)
        
        all_edges = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    all_edges.add((u, v))
        
        edge_to_clique = {}
        for i, clique in enumerate(cliques):
            if len(clique) < 2:
                return None
            for a_idx in range(len(clique)):
                for b_idx in range(a_idx + 1, len(clique)):
                    u, v = clique[a_idx], clique[b_idx]
                    if v not in adj[u]:
                        return None
                    edge = (min(u, v), max(u, v))
                    if edge in edge_to_clique:
                        return None
                    edge_to_clique[edge] = i
        
        if len(edge_to_clique) != len(all_edges):
            return None
        
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if (time.time() - t0) > DEADLINE:
            break
        n, k, p, seed = spec
        result = solve_split(n, k, p, seed)
        if result is None:
            result = solve_simple(n, k, p, seed)
        if result is None:
            # Last resort: individual edges
            rng = Random(seed)
            edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
            for v in range(k, n):
                for u in range(k):
                    if rng.random() < p:
                        edges.append((u, v))
            result = [list(e) for e in edges]
        # Ensure types are correct
        clean_result = []
        for clique in result:
            clean_clique = [int(v) for v in clique]
            clean_result.append(clean_clique)
        results.append(clean_result)

    return results
# EVOLVE-BLOCK-END
