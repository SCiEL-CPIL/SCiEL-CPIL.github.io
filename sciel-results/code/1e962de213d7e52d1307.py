import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin for 120s limit

    # Precompute all 8-vertex subsets and their Q3 checks
    # Q3 has 8 vertices, 12 edges, bipartite 4+4
    # Pattern edges (bipartite): (0,0),(0,1),(0,3),(1,0),(1,1),(1,2),(2,0),(2,2),(2,3),(3,1),(3,2),(3,3)
    pattern_edges = [(0,0),(0,1),(0,3),(1,0),(1,1),(1,2),(2,0),(2,2),(2,3),(3,1),(3,2),(3,3)]

    perms24 = list(itertools.permutations(range(4)))
    splits8 = list(itertools.combinations(range(8), 4))
    splits8 = [(a, tuple(i for i in range(8) if i not in a)) for a in splits8]

    # Precompute required connections for each permutation
    perm_required = []
    for p in perms24:
        req = []
        for (ar, bp) in pattern_edges:
            req.append((ar, p[bp]))
        perm_required.append(tuple(sorted(req)))

    all_8_subsets = list(itertools.combinations(range(16), 8))

    def check_q3(adj_mask, verts):
        """Check if 8 vertices form Q3."""
        for (A_idx, B_idx) in splits8:
            A = [verts[i] for i in A_idx]
            B = [verts[i] for i in B_idx]
            for req in perm_required:
                ok = True
                for (ar, bi) in req:
                    if not (adj_mask[A[ar]] >> B[bi]) & 1:
                        ok = False
                        break
                if ok:
                    return True
            for req in perm_required:
                ok = True
                for (ar, bi) in req:
                    if not (adj_mask[B[ar]] >> A[bi]) & 1:
                        ok = False
                        break
                if ok:
                    return True
        return False

    def is_q3_free(adj_mask):
        """Check entire graph for Q3."""
        for combo in all_8_subsets:
            if check_q3(adj_mask, combo):
                return False
        return True

    # Build edge list for K16
    all_pairs = list(itertools.combinations(range(16), 2))
    edge_to_idx = {e: i for i, e in enumerate(all_pairs)}
    idx_to_edge = all_pairs

    def edges_to_adj(edge_set):
        adj = [0] * 16
        for (u, v) in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    # Start with K16
    current_edges = set(all_pairs)
    current_adj = edges_to_adj(current_edges)

    # Find all Q3 subgraphs in K16
    q3_list = []
    for combo in all_8_subsets:
        if check_q3(current_adj, combo):
            q3_list.append(combo)

    # For each Q3, find all 12 edges that form it
    # For a given 8-vertex set, find all Q3 edge sets
    def get_q3_edges(adj_mask, verts):
        """Return list of edge sets (each 12 edges) that form Q3 on these 8 vertices."""
        result = []
        for (A_idx, B_idx) in splits8:
            A = [verts[i] for i in A_idx]
            B = [verts[i] for i in B_idx]
            for req in perm_required:
                # Check if this orientation works
                ok = True
                for (ar, bi) in req:
                    if not (adj_mask[A[ar]] >> B[bi]) & 1:
                        ok = False
                        break
                if ok:
                    edges = set()
                    for (ar, bi) in req:
                        u, v = A[ar], B[bi]
                        edges.add((min(u,v), max(u,v)))
                    result.append(edges)
            for req in perm_required:
                ok = True
                for (ar, bi) in req:
                    if not (adj_mask[B[ar]] >> A[bi]) & 1:
                        ok = False
                        break
                if ok:
                    edges = set()
                    for (ar, bi) in req:
                        u, v = B[ar], A[bi]
                        edges.add((min(u,v), max(u,v)))
                    result.append(edges)
        return result

    # Precompute Q3 edge sets
    q3_edge_sets = []
    for combo in q3_list:
        es = get_q3_edges(current_adj, combo)
        for e in es:
            q3_edge_sets.append(frozenset(e))

    # Deduplicate
    q3_edge_sets = list(set(q3_edge_sets))

    # Greedy hitting set: remove minimum edges to hit all Q3s
    # Start with all edges, remove edges that are in many Q3s
    edge_q3_count = {e: 0 for e in all_pairs}
    for q3es in q3_edge_sets:
        for e in q3es:
            edge_q3_count[e] += 1

    # Greedy: repeatedly remove edge in most remaining Q3s
    remaining_q3s = list(q3_edge_sets)
    removed = set()
    while remaining_q3s and time.time() < deadline - 40:
        # Find edge in most remaining Q3s
        best_edge = None
        best_count = -1
        for e in all_pairs:
            if e in removed:
                continue
            cnt = sum(1 for q3es in remaining_q3s if e in q3es)
            if cnt > best_count:
                best_count = cnt
                best_edge = e
        if best_edge is None or best_count <= 0:
            break
        removed.add(best_edge)
        # Remove Q3s that contain this edge
        remaining_q3s = [q3es for q3es in remaining_q3s if best_edge not in q3es]

    current_edges = set(all_pairs) - removed
    current_adj = edges_to_adj(current_edges)
    best_edges = set(current_edges)
    best_adj = current_adj[:]
    best_count = len(current_edges)

    # Local search: try to add back edges
    def try_add_edge(adj_mask, edge_set, u, v):
        """Try adding edge (u,v). Return True if safe."""
        edge_set.add((u, v))
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

        # Check all 8-subsets containing both u and v
        for combo in all_8_subsets:
            if u not in combo or v not in combo:
                continue
            if check_q3(adj_mask, combo):
                edge_set.remove((u, v))
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                return False
        return True

    # Phase 1: Greedy addition
    non_edges = [e for e in all_pairs if e not in current_edges]
    for _round in range(20):
        if time.time() > deadline - 15:
            break
        changed = False
        non_edges.sort(key=lambda e: bin(current_adj[e[0]]).count('1') + bin(current_adj[e[1]]).count('1'), reverse=True)
        for e in non_edges:
            if time.time() > deadline - 15:
                break
            if e in current_edges:
                continue
            u, v = e
            if try_add_edge(current_adj, current_edges, u, v):
                changed = True
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = set(current_edges)
                    best_adj = current_adj[:]
        if not changed:
            break

    # Phase 2: Random local search with add/remove
    random.seed(42)
    no_improve = 0
    max_no_improve = 500

    while time.time() < deadline - 5 and no_improve < max_no_improve:
        move_type = random.random()
        improved = False

        if move_type < 0.6:
            # Try adding a random non-edge
            non_edges_now = [e for e in all_pairs if e not in current_edges]
            if not non_edges_now:
                no_improve += 1
                continue
            weights = [bin(current_adj[e[0]]).count('1') + bin(current_adj[e[1]]).count('1') + 1 for e in non_edges_now]
            e = random.choices(non_edges_now, weights=weights, k=1)[0]
            u, v = e
            if try_add_edge(current_adj, current_edges, u, v):
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = set(current_edges)
                    best_adj = current_adj[:]
                    improved = True
                else:
                    if random.random() < 0.05:
                        improved = True
                    else:
                        current_edges.remove((u, v))
                        current_adj[u] &= ~(1 << v)
                        current_adj[v] &= ~(1 << u)
            no_improve += 0 if improved else 1

        elif move_type < 0.85:
            # Remove a random edge, then try to add back
            if not current_edges:
                no_improve += 1
                continue
            e = random.choice(list(current_edges))
            u, v = e
            current_edges.remove((u, v))
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)

            # Try to add some non-edges
            non_edges_now = [x for x in all_pairs if x not in current_edges]
            if non_edges_now:
                weights = [bin(current_adj[x[0]]).count('1') + bin(current_adj[x[1]]).count('1') + 1 for x in non_edges_now]
                for _ in range(5):
                    e2 = random.choices(non_edges_now, weights=weights, k=1)[0]
                    if try_add_edge(current_adj, current_edges, e2[0], e2[1]):
                        break
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = set(current_edges)
                best_adj = current_adj[:]
                improved = True
            no_improve += 0 if improved else 1

        else:
            # Swap
            if not current_edges:
                no_improve += 1
                continue
            e_remove = random.choice(list(current_edges))
            u1, v1 = e_remove
            current_edges.remove((u1, v1))
            current_adj[u1] &= ~(1 << v1)
            current_adj[v1] &= ~(1 << u1)

            non_edges_now = [x for x in all_pairs if x not in current_edges]
            if non_edges_now:
                weights = [bin(current_adj[x[0]]).count('1') + bin(current_adj[x[1]]).count('1') + 1 for x in non_edges_now]
                e_add = random.choices(non_edges_now, weights=weights, k=1)[0]
                if try_add_edge(current_adj, current_edges, e_add[0], e_add[1]):
                    if len(current_edges) > best_count:
                        best_count = len(current_edges)
                        best_edges = set(current_edges)
                        best_adj = current_adj[:]
                        improved = True
                else:
                    current_edges.add((u1, v1))
                    current_adj[u1] |= (1 << v1)
                    current_adj[v1] |= (1 << u1)
            else:
                current_edges.add((u1, v1))
                current_adj[u1] |= (1 << v1)
                current_adj[v1] |= (1 << u1)

            no_improve += 0 if improved else 1

    # Final greedy addition
    final_edges = set(best_edges)
    final_adj = best_adj[:]
    non_edges_final = [e for e in all_pairs if e not in final_edges]
    non_edges_final.sort(key=lambda e: bin(final_adj[e[0]]).count('1') + bin(final_adj[e[1]]).count('1'), reverse=True)

    for e in non_edges_final:
        if time.time() > deadline - 2:
            break
        u, v = e
        try_add_edge(final_adj, final_edges, u, v)

    return [list(e) for e in final_edges]
