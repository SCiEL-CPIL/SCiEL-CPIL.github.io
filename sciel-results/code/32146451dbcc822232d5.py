import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit

    def instance(n, k, p, seed):
        rng = random.Random(seed)
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
        
        C_vertices = list(range(k))
        I_vertices = list(range(k, n))
        
        # 1. Initialize I-cliques
        # Each i gets a "bag" of subsets of N(i). Initially one subset: N(i).
        # We will split these subsets if conflicts arise.
        i_cliques = {}
        for i in I_vertices:
            N_i = adj[i] & set(C_vertices)
            if N_i:
                i_cliques[i] = [set(N_i)]
            else:
                i_cliques[i] = []
        
        # 2. Identify Conflicts
        # A conflict is a C-edge (c1, c2) that is in the same subset for multiple i's.
        # We track for each C-edge, which i's currently cover it.
        # edge_to_owners: {(c1, c2): set of i's}
        edge_to_owners = defaultdict(set)
        
        def rebuild_owner_map():
            nonlocal edge_to_owners
            edge_to_owners = defaultdict(set)
            for i, subsets in i_cliques.items():
                for S in subsets:
                    # Add all edges within S
                    s_list = list(S)
                    for u_idx in range(len(s_list)):
                        for v_idx in range(u_idx + 1, len(s_list)):
                            u, v = s_list[u_idx], s_list[v_idx]
                            if u < v:
                                edge_to_owners[(u, v)].add(i)
                            else:
                                edge_to_owners[(v, u)].add(i)
        
        rebuild_owner_map()
        
        # 3. Resolve Conflicts
        # While there are edges with > 1 owner:
        #   Pick an edge (u, v) with owners O.
        #   Pick an owner i in O.
        #   Find the subset S in i_cliques[i] containing both u and v.
        #   Split S into S1 and S2 such that u in S1, v in S2.
        #   To minimize new cliques, we want S1 and S2 to be as "clique-friendly" as possible?
        #   Actually, we just need to separate u and v.
        #   Simplest: S1 = {u}, S2 = S \ {u}.
        #   This increases clique count for i by 1.
        #   Repeat.
        
        max_iterations = 10000
        iter_count = 0
        while True:
            if time.time() > DEADLINE - 5:
                break
            if iter_count > max_iterations:
                break
            iter_count += 1
            
            # Find an edge with conflict
            conflict_edge = None
            conflict_owners = None
            for e, owners in edge_to_owners.items():
                if len(owners) > 1:
                    conflict_edge = e
                    conflict_owners = owners
                    break
            
            if conflict_edge is None:
                break
            
            u, v = conflict_edge
            # Pick an owner to split
            # Heuristic: Pick the owner with the largest subset containing u, v?
            # Or just the first one.
            i = next(iter(conflict_owners))
            
            # Find the subset in i_cliques[i] that contains both u and v
            target_subset_idx = -1
            for idx, S in enumerate(i_cliques[i]):
                if u in S and v in S:
                    target_subset_idx = idx
                    break
            
            if target_subset_idx == -1:
                # This shouldn't happen if owner map is correct
                continue
            
            S = i_cliques[i][target_subset_idx]
            
            # Split S into S1 (containing u) and S2 (containing v)
            # To keep S1 and S2 as large as possible (to cover more edges efficiently later?),
            # we can try to keep S1 = S \ {v} and S2 = {v}?
            # Or S1 = {u}, S2 = S \ {u}?
            # Let's try to balance or just separate.
            # Separating v into its own clique might be good if v is "popular".
            # Let's just do S1 = S - {v}, S2 = {v}.
            # Wait, if S1 still contains other conflicts, we'll handle them later.
            
            S1 = S - {v}
            S2 = {v}
            
            # Replace S with S1 and S2 in i_cliques[i]
            i_cliques[i][target_subset_idx] = S1
            i_cliques[i].append(S2)
            
            # Rebuild owner map
            rebuild_owner_map()
        
        # 4. Construct Final Cliques
        result_cliques = []
        
        # Add I-cliques
        for i, subsets in i_cliques.items():
            for S in subsets:
                if S:
                    clique = [i] + list(S)
                    if len(clique) >= 2:
                        result_cliques.append(clique)
        
        # 5. Cover Remaining C-edges
        # The edges covered by I-cliques are exactly the keys in edge_to_owners.
        # We need to cover E(C) \ U.
        
        all_C_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                all_C_edges.add((u, v))
        
        covered_C_edges = set(edge_to_owners.keys())
        remaining_C_edges = all_C_edges - covered_C_edges
        
        if remaining_C_edges:
            # Find clique cover of G_R (graph defined by remaining_C_edges)
            # Vertices are C_vertices
            # Greedy Max Clique
            
            remaining = set(remaining_C_edges)
            while remaining:
                # Build graph
                v_set = set()
                for u, v in remaining:
                    v_set.add(u)
                    v_set.add(v)
                
                adj_sub = {v: set() for v in v_set}
                for u, v in remaining:
                    adj_sub[u].add(v)
                    adj_sub[v].add(u)
                
                # Greedy Max Clique
                candidates = list(v_set)
                candidates.sort(key=lambda x: len(adj_sub[x]), reverse=True)
                
                def max_clique_search(cands, current):
                    if not cands:
                        return current
                    v = cands[0]
                    new_cands = [u for u in cands[1:] if u in adj_sub[v]]
                    res1 = max_clique_search(new_cands, current + [v])
                    res2 = max_clique_search(cands[1:], current)
                    return res1 if len(res1) > len(res2) else res2
                
                best_clique = max_clique_search(candidates, [])
                
                if not best_clique:
                    u, v = next(iter(remaining))
                    best_clique = [u, v]
                
                result_cliques.append(best_clique)
                
                # Remove edges
                for u in best_clique:
                    for v in best_clique:
                        if u < v:
                            remaining.discard((u, v))
                        else:
                            remaining.discard((v, u))
        
        # Convert to lists of ints
        final_result = []
        for clique in result_cliques:
            final_clique = [int(v) for v in clique]
            if len(final_clique) >= 2:
                final_result.append(final_clique)
        
        return final_result

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > DEADLINE:
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            # Fallback: Trivial solution
            trivial = [[int(u), int(v)] for u, v in edges]
            results.append(trivial)
        else:
            res = solve_instance(*spec)
            results.append(res)
            
    return results
