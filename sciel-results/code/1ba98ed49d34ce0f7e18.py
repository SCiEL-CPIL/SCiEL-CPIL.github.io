import time
import random
import numpy as np

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    start = time.time()
    deadline = start + 85  # 5s safety margin before 90s kill

    n = P['n']
    edges = P['edges']

    # Build adjacency sets and edge set
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))

    total_edges = len(edge_set)

    def make_edge(u, v):
        return (min(u, v), max(u, v))

    def expand_biclique(L, R, remaining):
        """Expand biclique to maximal size."""
        L = set(L)
        R = set(R)
        changed = True
        while changed:
            changed = False
            # Try to add to L
            new_L = set()
            for w in range(n):
                if w in L or w in R:
                    continue
                valid = True
                for r in R:
                    if make_edge(w, r) not in remaining:
                        valid = False
                        break
                if valid:
                    new_L.add(w)
            if new_L:
                L |= new_L
                changed = True

            # Try to add to R
            new_R = set()
            for w in range(n):
                if w in L or w in R:
                    continue
                valid = True
                for l in L:
                    if make_edge(w, l) not in remaining:
                        valid = False
                        break
                if valid:
                    new_R.add(w)
            if new_R:
                R |= new_R
                changed = True
        return L, R

    def find_max_biclique_via_vertex(remaining, n, rng=None):
        """For each vertex u, find max biclique with u on left side."""
        if not remaining:
            return None

        best_biclique = None
        best_size = 0

        # Precompute for each vertex which neighbors are in remaining
        rem_adj = [set() for _ in range(n)]
        for (u, v) in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)

        # Get list of vertices with remaining edges
        vertices_with_edges = [u for u in range(n) if rem_adj[u]]
        if not vertices_with_edges:
            return None

        # Shuffle for randomization
        if rng:
            rng.shuffle(vertices_with_edges)

        for u in vertices_with_edges:
            candidates = list(rem_adj[u])
            if not candidates:
                continue

            # Build intersection graph degrees
            deg_in_inter = {c: 0 for c in candidates}
            for i, a in enumerate(candidates):
                for b in candidates[i+1:]:
                    if make_edge(a, b) in remaining:
                        deg_in_inter[a] += 1
                        deg_in_inter[b] += 1

            # Try multiple orderings
            orderings = [
                sorted(candidates, key=lambda x: -deg_in_inter[x]),
                sorted(candidates, key=lambda x: deg_in_inter[x]),
                sorted(candidates, key=lambda x: x),
                sorted(candidates, key=lambda x: -x),
            ]

            if rng:
                # Add a random ordering
                random_order = candidates[:]
                rng.shuffle(random_order)
                orderings.append(random_order)

            best_clique = set()

            for ordering in orderings:
                clique = set()
                for v in ordering:
                    valid = True
                    for w in clique:
                        if make_edge(v, w) not in remaining:
                            valid = False
                            break
                    if valid:
                        clique.add(v)
                if len(clique) > len(best_clique):
                    best_clique = clique

            # The biclique is ({u}, best_clique)
            L = {u}
            R = set(best_clique)

            # Expand both sides
            L, R = expand_biclique(L, R, remaining)

            # Verify
            valid = True
            for l in L:
                for r in R:
                    if l == r:
                        continue
                    if make_edge(l, r) not in remaining:
                        valid = False
                        break
                if not valid:
                    break

            if valid and len(L) * len(R) > best_size:
                best_size = len(L) * len(R)
                best_biclique = (L, R)

        return best_biclique

    def find_max_biclique_via_edge(remaining, n, rng=None):
        """Try each edge as seed and expand."""
        if not remaining:
            return None

        best_biclique = None
        best_size = 0

        edge_list = list(remaining)
        if rng:
            rng.shuffle(edge_list)

        for (u, v) in edge_list:
            L, R = expand_biclique({u}, {v}, remaining)

            valid = True
            for l in L:
                for r in R:
                    if l == r:
                        continue
                    if make_edge(l, r) not in remaining:
                        valid = False
                        break
                if not valid:
                    break

            if valid and len(L) * len(R) > best_size:
                best_size = len(L) * len(R)
                best_biclique = (L, R)

        return best_biclique

    def greedy_partition(remaining, n, use_vertex_strategy=True, rng=None):
        """Greedy: repeatedly find largest biclique, remove its edges."""
        bicliques = []
        remaining = set(remaining)

        while remaining:
            if time.time() > deadline:
                break

            best_bc = None
            best_size = 0

            if use_vertex_strategy:
                bc = find_max_biclique_via_vertex(remaining, n, rng)
                if bc is not None:
                    L, R = bc
                    size = len(L) * len(R)
                    if size > best_size:
                        best_size = size
                        best_bc = bc

            bc2 = find_max_biclique_via_edge(remaining, n, rng)
            if bc2 is not None:
                L, R = bc2
                size = len(L) * len(R)
                if size > best_size:
                    best_size = size
                    best_bc = bc2

            if best_bc is None:
                for (u, v) in remaining:
                    bicliques.append((frozenset([u]), frozenset([v])))
                remaining = set()
                break

            L, R = best_bc
            bicliques.append((frozenset(L), frozenset(R)))

            to_remove = set()
            for l in L:
                for r in R:
                    if l != r:
                        e = make_edge(l, r)
                        if e in remaining:
                            to_remove.add(e)
            remaining -= to_remove

        return bicliques, remaining

    def try_merge(bicliques, edge_set):
        """Try to merge pairs of bicliques. Returns merged list or None."""
        n_bics = len(bicliques)
        for i in range(n_bics):
            for j in range(i + 1, n_bics):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]

                candidates = []

                # Option 1: Union both sides
                newL = L1 | L2
                newR = R1 | R2
                if not (newL & newR):
                    candidates.append((newL, newR))

                # Option 2: Cross merge
                newL = L1 | R2
                newR = R1 | L2
                if not (newL & newR):
                    candidates.append((newL, newR))

                # Option 3: L2 subset of L1
                if L2 <= L1:
                    newR = R1 | R2
                    if not (L1 & newR):
                        candidates.append((L1, newR))

                # Option 4: R2 subset of R1
                if R2 <= R1:
                    newL = L1 | L2
                    if not (newL & R1):
                        candidates.append((newL, R1))

                # Option 5: L1 subset of L2
                if L1 <= L2:
                    newR = R1 | R2
                    if not (L2 & newR):
                        candidates.append((L2, newR))

                # Option 6: R1 subset of R2
                if R1 <= R2:
                    newL = L1 | L2
                    if not (newL & R2):
                        candidates.append((newL, R2))

                for newL, newR in candidates:
                    if not newL or not newR:
                        continue
                    if newL & newR:
                        continue

                    valid = True
                    for l in newL:
                        for r in newR:
                            if l == r:
                                continue
                            if make_edge(l, r) not in edge_set:
                                valid = False
                                break
                        if not valid:
                            break

                    if valid:
                        result = list(bicliques)
                        result.pop(j)
                        result.pop(i)
                        result.append((frozenset(newL), frozenset(newR)))
                        return result

        return None

    def local_search_improve(bicliques, edge_set, max_iters=50):
        """Try to move edges between bicliques to enable merges."""
        bicliques = [list(bc) for bc in bicliques]
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try moving an edge from one biclique to another
            for i in range(len(bicliques)):
                L1, R1 = bicliques[i]
                if len(L1) * len(R1) <= 1:
                    continue
                
                # Get edges in this biclique
                edges_in_bc = set()
                for l in L1:
                    for r in R1:
                        if l != r:
                            edges_in_bc.add(make_edge(l, r))
                
                # Try removing one edge and see if we can improve
                for edge in list(edges_in_bc)[:5]:  # Limit for speed
                    u, v = edge
                    
                    # Try to find which other biclique could take this edge
                    for j in range(len(bicliques)):
                        if j == i:
                            continue
                        L2, R2 = bicliques[j]
                        
                        # Check if adding edge to biclique j is valid
                        # The edge (u,v) must be a cross-edge in biclique j
                        # Either u in L2 and v in R2, or v in L2 and u in R2
                        
                        # Option A: u in L2, v in R2
                        if u in L2 and v not in R2:
                            # Check if v can be added to R2
                            can_add = True
                            for l in L2:
                                if make_edge(v, l) not in edge_set:
                                    can_add = False
                                    break
                            if can_add and v not in L2:
                                # This is just expanding, not a move
                                pass
                        
                        # Option B: v in L2, u in R2
                        if v in L2 and u not in R2:
                            can_add = True
                            for l in L2:
                                if make_edge(u, l) not in edge_set:
                                    can_add = False
                                    break
                            if can_add and u not in L2:
                                pass
                        
                        # Option C: u in R2, v in L2 - already covered
                        # Option D: v in R2, u in L2 - already covered
                        
                        # Try swapping: move u from L1 to R2 or vice versa
                        # This is complex; skip for now
            
            # Try merging after any potential changes
            merged = try_merge(bicliques, edge_set)
            if merged is not None:
                bicliques = [list(bc) for bc in merged]
                improved = True
            
            if not improved:
                break
        
        return bicliques

    def validate_result(result, edge_set):
        """Validate the result."""
        covered_edges = set()
        for bc in result:
            if not isinstance(bc, list) or len(bc) != 2:
                return False
            left, right = bc
            if not isinstance(left, list) or not isinstance(right, list):
                return False
            if len(left) == 0 or len(right) == 0:
                return False
            if set(left) & set(right):
                return False
            for l in left:
                if not isinstance(l, int) or isinstance(l, bool):
                    return False
                for r in right:
                    if not isinstance(r, int) or isinstance(r, bool):
                        return False
                    if l == r:
                        return False
                    e = make_edge(l, r)
                    if e not in edge_set:
                        return False
                    if e in covered_edges:
                        return False
                    covered_edges.add(e)
        return covered_edges == edge_set

    # Run multiple strategies with randomization
    best_result = None
    best_count = total_edges + 1

    # Strategy 1: Deterministic vertex-based greedy
    bicliques, leftover = greedy_partition(edge_set, n, use_vertex_strategy=True, rng=None)
    if leftover:
        for (u, v) in leftover:
            bicliques.append((frozenset([u]), frozenset([v])))
    if len(bicliques) < best_count:
        best_count = len(bicliques)
        best_result = list(bicliques)

    # Strategy 2: Deterministic edge-based greedy
    bicliques2, leftover2 = greedy_partition(edge_set, n, use_vertex_strategy=False, rng=None)
    if leftover2:
        for (u, v) in leftover2:
            bicliques2.append((frozenset([u]), frozenset([v])))
    if len(bicliques2) < best_count:
        best_count = len(bicliques2)
        best_result = list(bicliques2)

    # Strategy 3-8: Randomized runs
    rng = random.Random(42)
    for trial in range(6):
        if time.time() > deadline - 5:
            break
        
        rng_trial = random.Random(42 + trial)
        
        # Alternate between vertex and edge strategies
        use_vertex = (trial % 2 == 0)
        
        bicliques_t, leftover_t = greedy_partition(edge_set, n, use_vertex_strategy=use_vertex, rng=rng_trial)
        if leftover_t:
            for (u, v) in leftover_t:
                bicliques_t.append((frozenset([u]), frozenset([v])))
        
        if len(bicliques_t) < best_count:
            best_count = len(bicliques_t)
            best_result = list(bicliques_t)

    # Try to improve via merging
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        merged = try_merge(best_result, edge_set)
        if merged is not None:
            best_result = merged
            best_count = len(merged)
            improved = True

    # Try local search
    if time.time() < deadline - 1:
        local_result = local_search_improve(best_result, edge_set, max_iters=20)
        if len(local_result) < best_count:
            best_count = len(local_result)
            best_result = local_result

    # Final conversion to required format
    result = []
    for L, R in best_result:
        result.append([list(L), list(R)])

    # Validate the result
    if not validate_result(result, edge_set):
        # Fallback: one biclique per edge
        result = []
        for u, v in edge_set:
            result.append([[u], [v]])

    return result
