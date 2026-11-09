import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)

        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        C = list(range(k))
        I = list(range(k, n))
        C_set = set(C)

        # Internal C edges
        internal_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                if v in adj[u]:
                    internal_edges.add((u, v))

        # Edges from I to C
        i_c_edges = {}
        for v in I:
            nbrs = adj[v] & C_set
            i_c_edges[v] = nbrs

        # Strategy: Model by adjacency bitmasks, group I vertices by shadow (intersection of neighborhoods)
        # Then cover cross edges with cliques that include I vertices and their shadow in C

        best_partition = None
        best_count = float('inf')

        # Approach: Group I vertices by their exact neighborhood in C
        # For each group, the shadow is the intersection (which is the same for all in group)
        # We can form cliques: {v1, v2, ..., vm} ∪ shadow, but shadow must be a clique in C
        # Actually, we need to partition edges, not vertices.
        
        # Better approach: 
        # 1. First, handle I vertices by grouping those with identical neighborhoods
        # 2. For each group with neighborhood N, we can form cliques that include some I vertices and subsets of N
        # 3. The key insight: if multiple I vertices have the same neighborhood N, and N is a clique in C,
        #    we can put all of them plus N in one clique
        # 4. But N may not be a clique, so we need to find maximal cliques within N
        
        # Let's try a different structural approach:
        # Model as: each I vertex v has a "shadow" S_v = N(v) ∩ C
        # Cross edges are (v, u) for u in S_v
        # We want to partition all cross edges and C-C edges into cliques
        
        # Key idea: If we have I vertices v1, v2, ..., vm with S_v1 ⊇ S_v2 ⊇ ... ⊇ S_vm,
        # then we can potentially nest them. But more practically:
        
        # Let's group I vertices by their exact neighborhood
        from collections import defaultdict
        groups = defaultdict(list)
        for v in I:
            if i_c_edges[v]:
                groups[frozenset(i_c_edges[v])].append(v)
        
        # For each group, the shadow is the neighborhood itself
        # We need to cover:
        # 1. C-C edges (internal to C)
        # 2. Cross edges: for each I vertex v, edges (v, u) for u in S_v
        
        # Strategy: 
        # - Process groups in order of decreasing shadow size
        # - For each group, find a large clique within the shadow (using unused C-C edges)
        # - Form clique: {all I vertices in group} ∪ {clique vertices}
        # - This covers cross edges from all I vertices in group to the clique vertices
        # - Remaining cross edges become 2-cliques or are covered by other cliques
        
        # But wait: if we put multiple I vertices in one clique, we need them to be connected to all clique members
        # Since they all have the same neighborhood, they're connected to all vertices in the shadow
        # So {v1, v2, ..., vm} ∪ Q where Q ⊆ shadow and Q is a clique works!
        
        # Let's implement this:
        
        # Sort groups by shadow size descending
        sorted_groups = sorted(groups.items(), key=lambda x: -len(x[0]))
        
        for strategy in range(2):
            partition = []
            used_c_edges = set()
            
            # Track which C-C edges are used
            # For each group, find a clique within its shadow using unused C-C edges
            
            if strategy == 0:
                group_order = sorted_groups
            else:
                group_order = sorted(sorted_groups, key=lambda x: len(x[0]))
            
            for shadow, i_vertices in group_order:
                if not i_vertices:
                    continue
                
                # Find a large clique within shadow using unused C-C edges
                # Greedy: pick vertices from shadow that have unused edges to each other
                available = [u for u in shadow]
                
                # Sort by number of unused C-C edges to other available vertices
                def unused_degree(u):
                    cnt = 0
                    for w in available:
                        if u != w and (min(u, w), max(u, w)) in internal_edges and (min(u, w), max(u, w)) not in used_c_edges:
                            cnt += 1
                    return cnt
                
                available.sort(key=lambda u: -unused_degree(u))
                
                clique = []
                for u in available:
                    ok = True
                    for w in clique:
                        e = (min(u, w), max(u, w))
                        if e not in internal_edges or e in used_c_edges:
                            ok = False
                            break
                    if ok:
                        clique.append(u)
                
                # Form the clique: all I vertices in group + the C clique
                if len(clique) >= 1:
                    new_clique = sorted(i_vertices + clique)
                    partition.append(new_clique)
                    # Mark C-C edges as used
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            used_c_edges.add((min(clique[i], clique[j]), max(clique[i], clique[j])))
                
                # Cover remaining cross edges for this group's I vertices
                # For each I vertex v, edges to C vertices not in clique
                for v in i_vertices:
                    remaining_nbrs = shadow - set(clique)
                    for u in remaining_nbrs:
                        partition.append(sorted([v, u]))
            
            # Cover remaining C-C edges
            rem_c_edges = internal_edges - used_c_edges
            rem_set = set(rem_c_edges)
            
            while rem_set:
                rem_adj = {u: set() for u in range(k)}
                for (a, b) in rem_set:
                    rem_adj[a].add(b)
                    rem_adj[b].add(a)
                
                max_deg = -1
                start_v = None
                for u in range(k):
                    if len(rem_adj[u]) > max_deg:
                        max_deg = len(rem_adj[u])
                        start_v = u
                if start_v is None:
                    break
                
                clique = [start_v]
                candidates = sorted(rem_adj[start_v], key=lambda x: -len(rem_adj[x]))
                for w in candidates:
                    ok = True
                    for u in clique:
                        if w not in rem_adj[u]:
                            ok = False
                            break
                    if ok:
                        clique.append(w)
                
                if len(clique) >= 2:
                    partition.append(sorted(clique))
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            rem_set.discard((min(clique[i], clique[j]), max(clique[i], clique[j])))
            
            cnt = len(partition)
            if cnt < best_count:
                best_count = cnt
                best_partition = partition
        
        if best_partition is None:
            return []
        
        cleaned = []
        for clique in best_partition:
            cleaned.append([int(x) for x in clique])
        return cleaned

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                results.append([])
            return results
        res = solve_instance(*spec)
        results.append(res)

    return results
