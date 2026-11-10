def construct():
    import time
    from random import Random
    from itertools import combinations

    deadline = time.time() + 142.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return cross_edges

    def solve_instance(n, k, p, seed, rng=None):
        cross_edges = instance(n, k, p, seed)

        # Build neighbor sets
        N = {v: set() for v in range(k, n)}
        for u, v in cross_edges:
            N[v].add(u)

        # C-edges set
        C_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                C_edges.add((u, w))

        # Generate candidate mixed cliques for each I vertex
        # A mixed clique is {v} ∪ S where S ⊆ N[v] and S is an independent set in (C, C_edges)
        # We want S to be "large" to cover many C-edges
        candidates = {}  # v -> list of frozenset S
        for v in range(k, n):
            if not N[v]:
                continue
            neighbors = list(N[v])
            candidates[v] = []

            # Generate maximal independent sets in the graph induced by neighbors with edges from C_edges
            # Use a greedy approach to generate several candidates
            # Candidate 1: Greedy by degree
            for attempt in range(3):
                if rng:
                    order = neighbors[:]
                    rng.shuffle(order)
                    # Sort by degree in remaining graph
                    remaining = set(order)
                    S = []
                    while remaining:
                        # Pick vertex with max degree in remaining
                        best = None
                        best_deg = -1
                        for w in remaining:
                            deg = sum(1 for x in remaining if x != w and (min(w,x), max(w,x)) in C_edges)
                            if deg > best_deg:
                                best_deg = deg
                                best = w
                        if best is None:
                            break
                        S.append(best)
                        remaining.discard(best)
                    S_set = frozenset(S)
                    if S_set not in candidates[v]:
                        candidates[v].append(S_set)

            # Candidate: Just pick a few high-degree vertices
            if rng:
                shuffled = neighbors[:]
                rng.shuffle(shuffled)
                S = []
                for w in shuffled:
                    can_add = True
                    for x in S:
                        if (min(w,x), max(w,x)) in C_edges:
                            can_add = False
                            break
                    if can_add:
                        S.append(w)
                S_set = frozenset(S)
                if S_set not in candidates[v]:
                    candidates[v].append(S_set)

            # Ensure at least one candidate
            if not candidates[v]:
                candidates[v].append(frozenset([neighbors[0]]))

        # Local search for maximum matching
        # State: assignment[v] = S (frozenset) or None
        # Objective: minimize number of uncovered C-edges

        def compute_objective(assignment):
            covered = set()
            for v, S in assignment.items():
                if S is None:
                    continue
                S_list = list(S)
                for i in range(len(S_list)):
                    for j in range(i+1, len(S_list)):
                        e = (min(S_list[i], S_list[j]), max(S_list[i], S_list[j]))
                        covered.add(e)
            return len(C_edges - covered)

        def is_valid(assignment):
            # Check that no C-edge is covered by two different cliques
            edge_to_clique = {}
            for v, S in assignment.items():
                if S is None:
                    continue
                S_list = list(S)
                for i in range(len(S_list)):
                    for j in range(i+1, len(S_list)):
                        e = (min(S_list[i], S_list[j]), max(S_list[i], S_list[j]))
                        if e in edge_to_clique:
                            return False
                        edge_to_clique[e] = v
            return True

        # Initialize: greedy assignment
        assignment = {}
        covered_edges = set()
        I_vertices = sorted(range(k, n), key=lambda x: len(N[x]), reverse=True)

        for v in I_vertices:
            if not N[v]:
                assignment[v] = None
                continue
            # Pick the candidate S that maximizes newly covered edges
            best_S = None
            best_gain = -1
            for S in candidates[v]:
                new_covered = 0
                S_list = list(S)
                for i in range(len(S_list)):
                    for j in range(i+1, len(S_list)):
                        e = (min(S_list[i], S_list[j]), max(S_list[i], S_list[j]))
                        if e not in covered_edges:
                            new_covered += 1
                if new_covered > best_gain:
                    best_gain = new_covered
                    best_S = S
            assignment[v] = best_S
            if best_S:
                S_list = list(best_S)
                for i in range(len(S_list)):
                    for j in range(i+1, len(S_list)):
                        e = (min(S_list[i], S_list[j]), max(S_list[i], S_list[j]))
                        covered_edges.add(e)

        current_obj = compute_objective(assignment)

        # Local search: try to improve by changing assignments
        if rng:
            for iteration in range(200):
                if time.time() > deadline:
                    break
                v = rng.choice([x for x in I_vertices if N[x]])
                old_S = assignment[v]
                # Try a random different candidate
                other_candidates = [S for S in candidates[v] if S != old_S]
                if other_candidates:
                    new_S = rng.choice(other_candidates)
                    # Check validity
                    # Temporarily remove old_S's edges
                    old_edges = set()
                    if old_S:
                        S_list = list(old_S)
                        for i in range(len(S_list)):
                            for j in range(i+1, len(S_list)):
                                old_edges.add((min(S_list[i], S_list[j]), max(S_list[i], S_list[j])))
                    new_edges = set()
                    S_list = list(new_S)
                    for i in range(len(S_list)):
                        for j in range(i+1, len(S_list)):
                            new_edges.add((min(S_list[i], S_list[j]), max(S_list[i], S_list[j])))
                    # Check no conflict
                    conflict = False
                    for e in new_edges:
                        if e in covered_edges and e not in old_edges:
                            # Check if this edge is covered by another clique
                            for v2, S2 in assignment.items():
                                if v2 == v or S2 is None:
                                    continue
                                S2_list = list(S2)
                                for i in range(len(S2_list)):
                                    for j in range(i+1, len(S2_list)):
                                        if (min(S2_list[i], S2_list[j]), max(S2_list[i], S2_list[j])) == e:
                                            conflict = True
                                            break
                                if conflict:
                                    break
                            if conflict:
                                break
                    if not conflict:
                        # Apply change
                        assignment[v] = new_S
                        covered_edges -= old_edges
                        covered_edges |= new_edges
                        new_obj = compute_objective(assignment)
                        if new_obj <= current_obj:
                            current_obj = new_obj
                        else:
                            # Revert
                            assignment[v] = old_S
                            covered_edges -= new_edges
                            covered_edges |= old_edges
        else:
            # Deterministic local search
            for iteration in range(100):
                if time.time() > deadline:
                    break
                improved = False
                for v in I_vertices:
                    if not N[v]:
                        continue
                    old_S = assignment[v]
                    old_edges = set()
                    if old_S:
                        S_list = list(old_S)
                        for i in range(len(S_list)):
                            for j in range(i+1, len(S_list)):
                                old_edges.add((min(S_list[i], S_list[j]), max(S_list[i], S_list[j])))
                    for new_S in candidates[v]:
                        if new_S == old_S:
                            continue
                        new_edges = set()
                        S_list = list(new_S)
                        for i in range(len(S_list)):
                            for j in range(i+1, len(S_list)):
                                new_edges.add((min(S_list[i], S_list[j]), max(S_list[i], S_list[j])))
                        # Check conflict
                        conflict = False
                        for e in new_edges:
                            if e in covered_edges and e not in old_edges:
                                for v2, S2 in assignment.items():
                                    if v2 == v or S2 is None:
                                        continue
                                    S2_list = list(S2)
                                    for i in range(len(S2_list)):
                                        for j in range(i+1, len(S2_list)):
                                            if (min(S2_list[i], S2_list[j]), max(S2_list[i], S2_list[j])) == e:
                                                conflict = True
                                                break
                                    if conflict:
                                        break
                                if conflict:
                                    break
                        if conflict:
                            continue
                        # Evaluate
                        temp_covered = (covered_edges - old_edges) | new_edges
                        new_obj = len(C_edges - temp_covered)
                        if new_obj < current_obj:
                            assignment[v] = new_S
                            covered_edges = temp_covered
                            current_obj = new_obj
                            improved = True
                            break
                if not improved:
                    break

        # Build final cliques
        cliques = []
        for v in I_vertices:
            S = assignment.get(v)
            if S and len(S) >= 1:
                clique = [int(v)] + [int(u) for u in sorted(S)]
                if len(clique) >= 2:
                    cliques.append(clique)

        # Cover remaining C-edges
        remaining_C = C_edges - covered_edges
        while remaining_C:
            if time.time() > deadline:
                for e in remaining_C:
                    cliques.append([int(e[0]), int(e[1])])
                break
            # Find a maximal clique in the remaining graph
            vertices = set()
            for u, w in remaining_C:
                vertices.add(u)
                vertices.add(w)
            S = []
            cand = set(vertices)
            while cand:
                best_v = None
                best_deg = -1
                for w in cand:
                    deg = sum(1 for x in cand if x != w and (min(w,x), max(w,x)) in remaining_C)
                    if deg > best_deg:
                        best_deg = deg
                        best_v = w
                if best_v is None:
                    break
                S.append(best_v)
                new_cand = set()
                for w in cand:
                    if w != best_v and (min(best_v, w), max(best_v, w)) in remaining_C:
                        new_cand.add(w)
                cand = new_cand
            if S and len(S) >= 2:
                cliques.append([int(u) for u in sorted(S)])
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        remaining_C.discard((min(S[i], S[j]), max(S[i], S[j])))
            else:
                e = next(iter(remaining_C))
                cliques.append([int(e[0]), int(e[1])])
                remaining_C.discard(e)

        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []

    for (n, k, p, seed) in specs:
        if time.time() > deadline - 10:
            # Fallback
            ce = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for (u, v) in ce:
                cliques.append([int(u), int(v)])
            result.append(cliques)
            continue

        best_cliques = None
        best_count = float('inf')

        # Try deterministic
        cliques = solve_instance(n, k, p, seed, rng=None)
        count = len(cliques)
        if count < best_count:
            best_count = count
            best_cliques = cliques

        # Try stochastic variants
        for i in range(5):
            if time.time() > deadline - 5:
                break
            rng = Random(seed + 5000 + i)
            cliques = solve_instance(n, k, p, seed, rng=rng)
            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = cliques

        result.append(best_cliques)

    # Ensure types
    final = []
    for partition in result:
        clean = []
        for clique in partition:
            clean.append([int(x) for x in clique])
        final.append(clean)
    return final
