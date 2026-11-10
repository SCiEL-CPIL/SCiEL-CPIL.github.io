"""Clique partition of split graphs with Lagrangian-relaxation-inspired search."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 138.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, p, seed):
        edges = instance(n, k, p, seed)

        # Build cross-edge adjacency: N[i] = set of C-vertices adjacent to I-vertex i
        N = {i: set() for i in range(k, n)}
        for u, v in edges:
            if u < k and v >= k:
                N[v].add(u)
            elif v < k and u >= k:
                N[u].add(v)

        # All C-edges (the clique on first k vertices)
        C_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                C_edges.add((u, v))

        def greedy_partition(order, max_clique_size=None):
            """Greedy partition for a given order of I-vertices.
            Each I-vertex i gets one or more cliques {i} ∪ S where S ⊆ N(i).
            S is chosen to cover as many uncovered C-edges as possible without
            creating conflicts (i.e., no C-edge appears in two cliques).
            """
            covered_C_edges = set()
            cliques = []

            for i in order:
                if not N[i]:
                    continue

                remaining_N = set(N[i])
                while remaining_N:
                    # Build S greedily: try to form a large set of C-vertices
                    # such that all pairwise edges are uncovered
                    S = set()
                    # Sort by degree in remaining uncovered edges for better packing
                    candidates = sorted(remaining_N, key=lambda u: len(
                        [v for v in remaining_N if (min(u, v), max(u, v)) not in covered_C_edges]
                    ), reverse=True)

                    for u in candidates:
                        if u not in remaining_N:
                            continue
                        # Check if u can be added to S
                        can_add = True
                        for v in S:
                            e = (min(u, v), max(u, v))
                            if e in covered_C_edges:
                                can_add = False
                                break
                        if can_add:
                            S.add(u)

                    if not S:
                        S = {min(remaining_N)}

                    clique = sorted([i] + list(S))
                    cliques.append(clique)

                    # Mark C-edges within S as covered
                    S_list = sorted(S)
                    for a in range(len(S_list)):
                        for b in range(a + 1, len(S_list)):
                            covered_C_edges.add((S_list[a], S_list[b]))

                    remaining_N -= S

            # Cover remaining C-edges with C-only cliques
            remaining_C_edges = C_edges - covered_C_edges
            if remaining_C_edges:
                # Greedy: repeatedly find largest clique among remaining edges
                rem_adj = defaultdict(set)
                for u, v in remaining_C_edges:
                    rem_adj[u].add(v)
                    rem_adj[v].add(u)

                while remaining_C_edges:
                    # Find the edge with maximum common neighbors
                    best_edge = None
                    best_common = -1
                    for u, v in remaining_C_edges:
                        common = rem_adj[u] & rem_adj[v]
                        if len(common) > best_common:
                            best_common = len(common)
                            best_edge = (u, v)

                    if best_edge is None:
                        u, v = next(iter(remaining_C_edges))
                        S = {u, v}
                    else:
                        u, v = best_edge
                        S = {u, v} | (rem_adj[u] & rem_adj[v])

                    S_list = sorted(S)
                    cliques.append(S_list)

                    for a in range(len(S_list)):
                        for b in range(a + 1, len(S_list)):
                            remaining_C_edges.discard((S_list[a], S_list[b]))
                    for node in S_list:
                        rem_adj.pop(node, None)
                        for w in list(rem_adj.keys()):
                            rem_adj[w].discard(node)
                    rem_adj = {u2: v2 for u2, v2 in rem_adj.items() if v2}

            return cliques

        def validate(cliques, n, k):
            """Check that the partition is valid."""
            seen_edges = set()
            for clique in cliques:
                if len(clique) < 2:
                    return False
                # Check all pairs are edges
                verts = sorted(clique)
                for a in range(len(verts)):
                    for b in range(a + 1, len(verts)):
                        e = (verts[a], verts[b])
                        if e in seen_edges:
                            return False
                        seen_edges.add(e)
            # Check all edges are covered
            if seen_edges != C_edges:
                return False
            # Check cross edges are covered
            for i in range(k, n):
                for u in N[i]:
                    e = (min(u, i), max(u, i))
                    if e not in seen_edges:
                        return False
            return True

        def local_search(cliques, max_iters=5000):
            """Try to reduce clique count by merging or restructuring.
            
            Strategy: 
            1. Identify C-only cliques and cross cliques
            2. Try to absorb C-only clique edges into existing cross cliques
            3. Try to split cross cliques to allow better packing
            """
            best = list(cliques)
            best_count = len(best)
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                    
                # Parse current structure
                cross_cliques = []  # (i, S_set)
                conly_cliques = []  # S_set
                for clique in best:
                    c_verts = [v for v in clique if v < k]
                    i_verts = [v for v in clique if v >= k]
                    if len(i_verts) == 1:
                        cross_cliques.append((i_verts[0], set(c_verts)))
                    else:
                        conly_cliques.append(set(c_verts))
                
                # Try move 1: Absorb a C-only clique into a cross clique
                # If a C-only clique S has all its edges not yet covered by cross cliques,
                # and there exists a cross clique (i, S2) such that S2 ∪ S' is a valid set
                # (all pairwise edges in S' are uncovered by other cross cliques), we can
                # merge.
                
                # Build covered_by_cross
                covered_by_cross = set()
                for i, S in cross_cliques:
                    S_list = sorted(S)
                    for a in range(len(S_list)):
                        for b in range(a + 1, len(S_list)):
                            covered_by_cross.add((S_list[a], S_list[b]))
                
                improved = False
                for idx, S_conly in enumerate(conly_cliques):
                    S_conly_list = sorted(S_conly)
                    # Check if all edges in S_conly are not covered by cross
                    all_uncovered = True
                    for a in range(len(S_conly_list)):
                        for b in range(a + 1, len(S_conly_list)):
                            if (S_conly_list[a], S_conly_list[b]) in covered_by_cross:
                                all_uncovered = False
                                break
                        if not all_uncovered:
                            break
                    if not all_uncovered:
                        continue
                    
                    # Try to absorb into a cross clique
                    for cidx, (i, S_cross) in enumerate(cross_cliques):
                        # Check if S_cross ∪ S_conly is a valid set (all pairwise edges exist and are uncovered)
                        combined = S_cross | S_conly
                        combined_list = sorted(combined)
                        valid = True
                        for a in range(len(combined_list)):
                            for b in range(a + 1, len(combined_list)):
                                e = (combined_list[a], combined_list_list[b])
                                if e in covered_by_cross and e not in [(combined_list[x], combined_list[y]) for x in range(len(combined_list)) for y in range(x+1, len(combined_list)) if x < a or (x == a and y < b)]:
                                    # This edge is covered by another cross clique - conflict
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            # Perform merge
                            cross_cliques[cidx] = (i, combined)
                            conly_cliques.pop(idx)
                            improved = True
                            break
                    if improved:
                        break
                
                if improved:
                    # Rebuild
                    best = []
                    for i, S in cross_cliques:
                        if S:
                            best.append(sorted([i] + list(S)))
                    for S in conly_cliques:
                        if S:
                            best.append(sorted(S))
                    if len(best) < best_count:
                        best_count = len(best)
                else:
                    # Try move 2: Randomly re-shuffle one I-vertex's C-vertices
                    # Pick a cross clique and try to redistribute its S
                    if cross_cliques:
                        cidx = random.randrange(len(cross_cliques))
                        i, S = cross_cliques[cidx]
                        # Try removing one vertex from S and see if it can go elsewhere
                        if len(S) > 1:
                            v = random.choice(list(S))
                            S_new = S - {v}
                            # Check if v can be added to another cross clique
                            for oidx, (oi, OS) in enumerate(cross_cliques):
                                if oidx == cidx:
                                    continue
                                combined = OS | {v}
                                combined_list = sorted(combined)
                                # Check validity
                                valid = True
                                for a in range(len(combined_list)):
                                    for b in range(a + 1, len(combined_list)):
                                        e = (combined_list[a], combined_list[b])
                                        if e in covered_by_cross:
                                            # Check if this edge is from the current clique being modified
                                            # Simplified: just reject
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                if valid:
                                    # Perform move
                                    cross_cliques[cidx] = (i, S_new)
                                    cross_cliques[oidx] = (oi, combined)
                                    improved = True
                                    break
                            
                            if not improved and len(S_new) >= 1:
                                # Just shrink - this increases clique count, so skip
                                pass
                
                if not improved and it > 100:
                    # Stuck, try a different approach
                    break

            return best

        # Generate initial solution
        I_vertices = list(range(k, n))
        
        # Try multiple orderings
        best_cliques = None
        best_count = float('inf')
        
        # Ordering 1: by degree descending
        order1 = sorted(I_vertices, key=lambda i: len(N[i]), reverse=True)
        cliques1 = greedy_partition(order1)
        if len(cliques1) < best_count:
            best_count = len(cliques1)
            best_cliques = cliques1
        
        # Ordering 2: by degree ascending
        order2 = sorted(I_vertices, key=lambda i: len(N[i]))
        cliques2 = greedy_partition(order2)
        if len(cliques2) < best_count:
            best_count = len(cliques2)
            best_cliques = cliques2
        
        # Ordering 3: random orderings
        rng = random.Random(seed + 1000)
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            rand_order = rng.sample(I_vertices, len(I_vertices))
            c = greedy_partition(rand_order)
            if len(c) < best_count:
                best_count = len(c)
                best_cliques = c
        
        # Apply local search
        if best_cliques is not None and time.time() < deadline - 5:
            improved = local_search(best_cliques)
            if len(improved) < best_count:
                best_cliques = improved
                best_count = len(improved)
        
        return best_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
            continue
        cliques = partition_split_graph(*spec)
        # Ensure all are Python ints and lists
        results.append([[int(v) for v in clique] for clique in cliques])

    return results
# EVOLVE-BLOCK-END
