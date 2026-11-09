<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    n = P['n']
    adj, edge_set = _build_adj(n, P['edges'])

    bicliques = []
    remaining_edges = set(edge_set)

    while remaining_edges:
        if time.time() > deadline:
            break

        left, right, count = _find_max_biclique(adj, n, edge_set, deadline)

        if left is None or right is None or count == 0:
            # Fallback: pick any remaining edge as a 1x1 biclique
            e = next(iter(remaining_edges))
            u, v = tuple(e)
            left = [u]
            right = [v]
            count = 1
        else:
            # Verify that all cross pairs are actually edges (they should be by construction)
            # Remove covered edges from remaining
            covered = set()
            for u in left:
                for v in right:
                    fs = frozenset((u, v))
                    if fs in remaining_edges:
                        covered.add(fs)

            if not covered:
                # Shouldn't happen, but safety
                e = next(iter(remaining_edges))
                u, v = tuple(e)
                left = [u]
                right = [v]
                covered = {frozenset((u, v))}

            remaining_edges -= covered

        # Update adjacency for residual
        for u in left:
            for v in right:
                if v in adj[u]:
                    adj[u].discard(v)
                if u in adj[v]:
                    adj[v].discard(u)

        bicliques.append([list(left), list(right)])

    # Fallback: if we ran out of time with edges remaining, cover them as single edges
    if remaining_edges and time.time() < deadline:
        for e in list(remaining_edges):
            if time.time() > deadline:
                break
            u, v = tuple(e)
            bicliques.append([[u], [v]])
            remaining_edges.discard(e)

    # Ensure types are correct: Python int, list
    result = []
    for bic in bicliques:
        L = [int(x) for x in bic[0]]
        R = [int(x) for x in bic[1]]
        result.append([L, R])

    return result
=======
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    n = P['n']
    adj, edge_set = _build_adj(n, P['edges'])

    bicliques = []
    remaining_edges = set(edge_set)

    while remaining_edges:
        if time.time() > deadline:
            break

        left, right, count = _find_max_biclique(adj, n, edge_set, deadline)

        if left is None or right is None or count == 0:
            # Fallback: pick any remaining edge as a 1x1 biclique
            e = next(iter(remaining_edges))
            u, v = tuple(e)
            left = [u]
            right = [v]
            count = 1
        else:
            # Remove covered edges from remaining
            covered = set()
            for u in left:
                for v in right:
                    fs = frozenset((u, v))
                    if fs in remaining_edges:
                        covered.add(fs)

            if not covered:
                e = next(iter(remaining_edges))
                u, v = tuple(e)
                left = [u]
                right = [v]
                covered = {frozenset((u, v))}

            remaining_edges -= covered

        # Update adjacency for residual
        for u in left:
            for v in right:
                if v in adj[u]:
                    adj[u].discard(v)
                if u in adj[v]:
                    adj[v].discard(u)

        bicliques.append([list(left), list(right)])

    # Merge pass: try to merge bicliques that are compatible
    # Two bicliques (L1,R1) and (L2,R2) can be merged if:
    # - L1 and L2 are disjoint, R1 and R2 are disjoint
    # - All cross pairs between L1 and R2 are edges in original graph
    # - All cross pairs between L2 and R1 are edges in original graph
    # This is expensive, so only try if we have time
    if time.time() < deadline - 10:
        merged = True
        while merged:
            merged = False
            for i in range(len(bicliques)):
                if time.time() > deadline - 5:
                    break
                if merged:
                    break
                for j in range(i + 1, len(bicliques)):
                    if time.time() > deadline - 5:
                        break
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]
                    # Check disjointness
                    if set(L1) & set(L2) or set(R1) & set(R2):
                        continue
                    # Check cross edges exist in original graph
                    can_merge = True
                    for u in L1:
                        for v in R2:
                            if frozenset((u, v)) not in edge_set:
                                can_merge = False
                                break
                        if not can_merge:
                            break
                    if can_merge:
                        for u in L2:
                            for v in R1:
                                if frozenset((u, v)) not in edge_set:
                                    can_merge = False
                                    break
                            if not can_merge:
                                break
                    if can_merge:
                        bicliques[i] = [L1 + L2, R1 + R2]
                        del bicliques[j]
                        merged = True
                        break

    # Fallback: if we ran out of time with edges remaining, cover them as single edges
    if remaining_edges and time.time() < deadline:
        for e in list(remaining_edges):
            if time.time() > deadline:
                break
            u, v = tuple(e)
            bicliques.append([[u], [v]])
            remaining_edges.discard(e)

    # Ensure types are correct: Python int, list
    result = []
    for bic in bicliques:
        L = [int(x) for x in bic[0]]
        R = [int(x) for x in bic[1]]
        result.append([L, R])

    return result
>>>>>>> REPLACE
