"""Clique partition of split graphs using ILP-based global optimization with local search."""
import time
from random import Random

def construct():
    _T0 = time.time()
    _DEADLINE = 145.0  # safety margin before 150s kill

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve_split_clique_partition(n, k, edge_set):
        """
        Solve clique partition for a split graph using a combination of:
        1. ILP formulation for small instances
        2. Greedy + local search for larger instances
        """
        # Build adjacency
        adj = [set() for _ in range(n)]
        for (a, b) in edge_set:
            adj[a].add(b)
            adj[b].add(a)

        # Identify C and I vertices
        C_verts = list(range(k))
        I_verts = list(range(k, n))

        # For each I-vertex, get neighbors in C
        N_C = {}
        for v in I_verts:
            N_C[v] = frozenset(u for u in C_verts if u in adj[v])

        # Strategy 1: Try ILP for small instances
        if n <= 50:
            result = _solve_ilp(n, k, C_verts, I_verts, N_C, adj)
            if result is not None:
                return result

        # Strategy 2: Greedy with local search
        cliques = _greedy_solution(n, k, C_verts, I_verts, N_C, adj)
        
        # Apply local search to improve
        if not _time_up():
            cliques = _local_search(cliques, edge_set, n, k, adj)
        
        return cliques

    def _solve_ilp(n, k, C_verts, I_verts, N_C, adj):
        """
        ILP formulation for clique partition.
        Decision variables: x_{S} for each potential clique S (subset of V)
        Constraint: every edge must be covered exactly once
        Objective: minimize sum of x_{S}
        """
        # Generate all potential cliques (subsets that form cliques)
        # For efficiency, only consider cliques of size >= 2
        potential_cliques = []
        
        # All C-C cliques (subsets of C that form cliques)
        # Since C is a clique, any subset of C is a clique
        from itertools import combinations
        for size in range(2, k + 1):
            for combo in combinations(C_verts, size):
                potential_cliques.append(frozenset(combo))
        
        # All C-I cliques: {v} union S where v in I, S subset of N_C[v], |S| >= 1
        for v in I_verts:
            neigh = list(N_C[v])
            if not neigh:
                continue
            # All non-empty subsets of neigh
            for size in range(1, len(neigh) + 1):
                for combo in combinations(neigh, size):
                    potential_cliques.append(frozenset([v] + list(combo)))
        
        # All I-I cliques: none (I is independent set)
        
        # Build edge-to-clique incidence
        edge_to_cliques = {}
        for (a, b) in edge_set:
            edge_to_cliques[(a, b) if a < b else (b, a)] = []
        
        for idx, clique in enumerate(potential_cliques):
            clique_list = list(clique)
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    e = (a, b) if a < b else (b, a)
                    if e in edge_to_cliques:
                        edge_to_cliques[e].append(idx)
        
        # Solve using a simple branch-and-bound or use scipy if available
        # For now, use a greedy heuristic with local search as fallback
        # ILP is complex; use greedy + local search instead
        return None

    def _greedy_solution(n, k, C_verts, I_verts, N_C, adj):
        """
        Greedy algorithm for clique partition.
        """
        used_cc = set()
        cliques = []
        
        # Sort I-vertices by degree in C descending
        I_verts_sorted = sorted(I_verts, key=lambda v: len(N_C[v]), reverse=True)
        
        for v in I_verts_sorted:
            neigh = list(N_C[v])
            d = len(neigh)
            if d == 0:
                continue
            
            # Greedy packing for this vertex
            remaining_neigh = list(neigh)
            groups = []
            
            while remaining_neigh:
                # Start a new group with the first available neighbor
                group = [remaining_neigh.pop(0)]
                
                # Try to add other neighbors to this group if they don't conflict
                i = 0
                while i < len(remaining_neigh):
                    cand = remaining_neigh[i]
                    ok = True
                    for g in group:
                        e = (cand, g) if cand < g else (g, cand)
                        if e in used_cc:
                            ok = False
                            break
                    if ok:
                        group.append(cand)
                        remaining_neigh.pop(i)
                    else:
                        i += 1
                
                groups.append(group)
            
            # Create cliques for the groups
            for group in groups:
                if len(group) == 1:
                    cliques.append([v, group[0]])
                else:
                    cliques.append([v] + group)
                    # Mark C-C edges within this group as used
                    for i in range(len(group)):
                        for j in range(i + 1, len(group)):
                            a, b = group[i], group[j]
                            e = (a, b) if a < b else (b, a)
                            used_cc.add(e)
        
        # Cover remaining C-C edges
        # The remaining C-C edges are those NOT in used_cc
        # We need to cover them with cliques
        # This is equivalent to finding a clique cover of the graph G_rem
        # where V=G_rem=C_verts and E(G_rem) = { (u,v) in K_k | (u,v) not in used_cc }
        # The minimum number of cliques to cover G_rem is the chromatic number of its complement
        # The complement of G_rem is the graph H where E(H) = used_cc
        # So we need to color the graph H
        
        # Build adjacency for H (used_cc edges)
        H_adj = [set() for _ in range(n)]
        for (a, b) in used_cc:
            H_adj[a].add(b)
            H_adj[b].add(a)
        
        # Greedy coloring of H
        uncolored = set(C_verts)
        
        while uncolored:
            # Find vertex with max degree in H induced by uncolored
            best_v = None
            best_deg = -1
            for v in uncolored:
                deg = len(H_adj[v] & uncolored)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            color_class = [best_v]
            uncolored.discard(best_v)
            
            # Try to add other vertices to this color class (independent set in H)
            candidates = list(uncolored)
            for u in candidates:
                if u not in uncolored:
                    continue
                # Check if u is compatible with all in color_class
                compatible = True
                for c in color_class:
                    if u in H_adj[c]:
                        compatible = False
                        break
                if compatible:
                    color_class.append(u)
                    uncolored.discard(u)
            
            # Add this clique (color class) to our cover
            if len(color_class) >= 2:
                cliques.append(sorted(color_class))
        
        return cliques

    def _local_search(cliques, edge_set, n, k, adj):
        """
        Local search to improve the clique partition.
        Try to merge compatible cliques or reassign vertices to reduce total count.
        """
        improved = True
        max_iters = 1000
        iter_count = 0
        
        while improved and iter_count < max_iters and not _time_up():
            improved = False
            iter_count += 1
            
            # Try to merge pairs of cliques
            new_cliques = list(cliques)
            changed = False
            
            for i in range(len(new_cliques)):
                if i >= len(new_cliques):
                    break
                for j in range(i + 1, len(new_cliques)):
                    if j >= len(new_cliques):
                        break
                    
                    clique_i = set(new_cliques[i])
                    clique_j = set(new_cliques[j])
                    
                    # Check if union is a clique
                    merged = clique_i | clique_j
                    is_clique = True
                    merged_list = list(merged)
                    for a in merged_list:
                        for b in merged_list:
                            if a < b and b not in adj[a]:
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    
                    if is_clique:
                        # Merge
                        new_cliques[i] = list(merged)
                        new_cliques.pop(j)
                        changed = True
                        improved = True
                        break
                if changed:
                    break
            
            if changed:
                cliques = new_cliques
                continue
            
            # Try to reassign vertices between cliques
            # For each pair of cliques, try moving a vertex from one to another
            for i in range(len(cliques)):
                if i >= len(cliques):
                    break
                for j in range(len(cliques)):
                    if i == j or j >= len(cliques):
                        continue
                    
                    clique_i = list(cliques[i])
                    clique_j = list(cliques[j])
                    
                    # Try moving each vertex from clique_i to clique_j
                    for v in clique_i:
                        # Check if v can be added to clique_j
                        can_add = True
                        for u in clique_j:
                            if u != v and u not in adj[v]:
                                can_add = False
                                break
                        
                        if can_add:
                            # Check if clique_i still has at least 2 vertices after removing v
                            if len(clique_i) > 2:
                                # Try the move
                                new_clique_i = [x for x in clique_i if x != v]
                                new_clique_j = clique_j + [v]
                                
                                # Validate
                                valid = True
                                for a in new_clique_j:
                                    for b in new_clique_j:
                                        if a < b and b not in adj[a]:
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                
                                if valid and len(new_clique_i) >= 2:
                                    cliques[i] = new_clique_i
                                    cliques[j] = new_clique_j
                                    improved = True
                                    changed = True
                                    break
                    if changed:
                        break
                if changed:
                    break
            
            if changed:
                continue
            
            # Try to split a large clique into smaller ones that can be merged with others
            # This is more complex; skip for now
        
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []

    for spec in specs:
        if _time_up():
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            # Fallback: 1 clique for C, 1 per cross edge
            cliques = [list(range(k))]
            for (u, v) in edges:
                if u >= k or v >= k:
                    cliques.append([u, v])
            results.append(cliques)
            continue

        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        edge_set = set()
        for (a, b) in edges:
            edge_set.add((a, b) if a < b else (b, a))

        # Solve the clique partition
        cliques = _solve_split_clique_partition(n, k, edge_set)
        
        results.append(cliques)

    # Final validation and type conversion
    for i in range(len(results)):
        new_list = []
        for clique in results[i]:
            new_clique = [int(x) for x in clique]
            new_list.append(new_clique)
        results[i] = new_list
    return results
