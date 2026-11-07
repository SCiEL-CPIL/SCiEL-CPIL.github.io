import random
import time

def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill

    n = 15
    all_possible = set((i, j) for i in range(n) for j in range(i + 1, n))
    all_edges_list = list(all_possible)
    mask_all = (1 << n) - 1

    def bits_to_list(mask):
        res = []
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            res.append(idx)
            mask ^= lsb
        return res

    def check_valid(adj, deg):
        """Fast validity: no two same-degree vertices at distance <= 3."""
        # dist<=1: direct neighbors
        for u in range(n):
            neighbors = bits_to_list(adj[u])
            for v in neighbors:
                if v > u and deg[u] == deg[v]:
                    return False
        # dist<=2: common neighbor
        for u in range(n):
            neighbors_u = bits_to_list(adj[u])
            for v in neighbors_u:
                if v <= u:
                    continue
                if deg[u] != deg[v]:
                    continue
                if adj[u] & adj[v]:
                    return False
        # dist<=3: for each middle edge (u,v), check if any same-degree pair
        # is one step from u (excluding v) and one step from v (excluding u)
        for u in range(n):
            neighbors_u = bits_to_list(adj[u])
            for v in neighbors_u:
                if v <= u:
                    continue
                N_u = adj[u] & ~(1 << v)
                N_v = adj[v] & ~(1 << u)
                if N_u == 0 or N_v == 0:
                    continue
                list_w = bits_to_list(N_u)
                list_x = bits_to_list(N_v)
                # Check if any degree in list_w matches any degree in list_x
                degs_in_x = set(deg[x] for x in list_x)
                for w in list_w:
                    if deg[w] in degs_in_x:
                        return False
        return True

    def is_valid(edges_set):
        adj = [0] * n
        deg = [0] * n
        for u, v in edges_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return check_valid(adj, deg)

    def local_search(initial_edges):
        edges = set(tuple(sorted(e)) for e in initial_edges)
        if not is_valid(edges):
            # Remove edges until valid
            attempts = 0
            while not is_valid(edges) and attempts < 200:
                e = random.choice(list(edges))
                edges.remove(e)
                attempts += 1
            if not is_valid(edges):
                return None, 0

        current_score = len(edges)

        # Phase 1: Greedy addition
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            missing = [e for e in all_possible if e not in edges]
            random.shuffle(missing)
            for e in missing:
                if (time.time() - start_time) > deadline:
                    return list(edges), current_score
                new_edges = edges | {e}
                if is_valid(new_edges):
                    edges = new_edges
                    current_score = len(edges)
                    improved = True
                    break

        # Phase 2: 1-swap moves
        max_swap_rounds = 25
        for _ in range(max_swap_rounds):
            if (time.time() - start_time) > deadline:
                break
            improved = False
            missing = [e for e in all_possible if e not in edges]
            random.shuffle(missing)
            existing = list(edges)
            random.shuffle(existing)

            for e_rm in existing:
                if (time.time() - start_time) > deadline:
                    break
                edges_wo = edges - {e_rm}
                for e_add in missing:
                    new_edges = edges_wo | {e_add}
                    if is_valid(new_edges):
                        edges = new_edges
                        current_score = len(edges)
                        improved = True
                        break
                if improved:
                    # Greedy add after swap
                    while True:
                        if (time.time() - start_time) > deadline:
                            return list(edges), current_score
                        miss2 = [e for e in all_possible if e not in edges]
                        random.shuffle(miss2)
                        added = False
                        for e in miss2:
                            ne = edges | {e}
                            if is_valid(ne):
                                edges = ne
                                current_score = len(edges)
                                added = True
                                break
                        if not added:
                            break
                    break
            if not improved:
                break

        # Phase 3: 2-for-1 swaps
        for _ in range(10):
            if (time.time() - start_time) > deadline:
                break
            improved = False
            missing = [e for e in all_possible if e not in edges]
            random.shuffle(missing)
            existing = list(edges)
            random.shuffle(existing)
            limit = min(len(existing), 40)

            for i in range(limit):
                if (time.time() - start_time) > deadline:
                    break
                for j in range(i + 1, limit):
                    edges_wo = edges - {existing[i], existing[j]}
                    for e_add in missing[:30]:
                        if e_add == existing[i] or e_add == existing[j]:
                            continue
                        ne = edges_wo | {e_add}
                        if is_valid(ne):
                            edges = ne
                            current_score = len(edges)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break

        # Phase 4: 1-for-2 swaps (remove 1, add 2)
        for _ in range(8):
            if (time.time() - start_time) > deadline:
                break
            improved = False
            existing = list(edges)
            random.shuffle(existing)
            for e_rm in existing[:30]:
                if (time.time() - start_time) > deadline:
                    break
                edges_wo = edges - {e_rm}
                missing = [e for e in all_possible if e not in edges]
                random.shuffle(missing)
                # Try adding 2 edges
                for e1 in missing[:25]:
                    for e2 in missing[25:50]:
                        if e1 == e2:
                            continue
                        ne = edges_wo | {e1, e2}
                        if is_valid(ne):
                            edges = ne
                            current_score = len(edges)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break

        return list(edges), current_score

    best_edges = None
    best_score = -1

    # Strategy 1: Diverse random starts with varying densities
    for i in range(40):
        if (time.time() - start_time) > deadline:
            break
        if i < 15:
            density = 0.7 + 0.25 * random.random()
        elif i < 30:
            density = 0.5 + 0.35 * random.random()
        else:
            density = 0.3 + 0.5 * random.random()

        num_initial = int(len(all_edges_list) * density)
        initial_edges = random.sample(all_edges_list, num_initial)
        res_edges, score = local_search(initial_edges)
        if res_edges is not None and score > best_score:
            best_score = score
            best_edges = res_edges

    # Strategy 2: Structured starts targeting specific degree distributions
    # Target: all 15 vertices have distinct degrees (0..14 or some subset)
    # This is the strongest valid configuration
    for trial in range(20):
        if (time.time() - start_time) > deadline:
            break

        if trial % 4 == 0:
            # Start with a "ladder" structure: vertex i connected to i+1, i+2, ...
            # This naturally creates distinct degrees
            initial_edges = set()
            perm = list(range(n))
            random.shuffle(perm)
            # Build a path first
            for i in range(n - 1):
                u, v = perm[i], perm[i + 1]
                initial_edges.add((min(u, v), max(u, v)))
            # Add chords to make degrees more distinct
            for i in range(n):
                for j in range(i + 2, min(i + 5, n)):
                    if random.random() < 0.5:
                        u, v = perm[i], perm[j]
                        initial_edges.add((min(u, v), max(u, v)))
            initial_edges = list(initial_edges)

        elif trial % 4 == 1:
            # Start with a "star + extras" structure
            initial_edges = set()
            center = random.randint(0, n - 1)
            for v in range(n):
                if v != center and random.random() < 0.6:
                    initial_edges.add((min(center, v), max(center, v)))
            # Add random edges among non-center vertices
            others = [v for v in range(n) if v != center]
            random.shuffle(others)
            for i in range(len(others)):
                for j in range(i + 1, len(others)):
                    if random.random() < 0.3:
                        u, v = others[i], others[j]
                        initial_edges.add((min(u, v), max(u, v)))
            initial_edges = list(initial_edges)

        elif trial % 4 == 2:
            # Two-group structure: 7 vertices degree ~2-3, 8 vertices degree ~5-7
            # Cross edges only between groups, controlled density
            initial_edges = set()
            group_a = list(range(7))
            group_b = list(range(7, 15))
            random.shuffle(group_a)
            random.shuffle(group_b)
            # Within-group edges (sparse)
            for i in range(len(group_a) - 1):
                u, v = group_a[i], group_a[i + 1]
                if random.random() < 0.5:
                    initial_edges.add((min(u, v), max(u, v)))
            for i in range(len(group_b) - 1):
                u, v = group_b[i], group_b[i + 1]
                if random.random() < 0.5:
                    initial_edges.add((min(u, v), max(u, v)))
            # Cross edges (denser)
            for a in group_a:
                for b in group_b:
                    if random.random() < 0.4:
                        initial_edges.add((min(a, b), max(a, b)))
            initial_edges = list(initial_edges)

        else:
            # Random regular-ish: target degree ~7 for all
            initial_edges = set()
            for i in range(n):
                candidates = [j for j in range(n) if j != i and (min(i, j), max(i, j)) not in initial_edges]
                random.shuffle(candidates)
                for j in candidates:
                    if len([e for e in initial_edges if i in e]) >= 7:
                        break
                    initial_edges.add((min(i, j), max(i, j)))
            initial_edges = list(initial_edges)

        res_edges, score = local_search(initial_edges)
        if res_edges is not None and score > best_score:
            best_score = score
            best_edges = res_edges

    # Strategy 3: Perturbation from best
    if best_edges is not None:
        for i in range(40):
            if (time.time() - start_time) > deadline:
                break
            pert_size = random.randint(2, 8)
            edges = set(tuple(sorted(e)) for e in best_edges)
            to_remove = random.sample(list(edges), min(pert_size, len(edges)))
            for e in to_remove:
                edges.remove(e)
            res_edges, score = local_search(list(edges))
            if res_edges is not None and score > best_score:
                best_score = score
                best_edges = res_edges

    # Final squeeze
    if best_edges is not None:
        for _ in range(5):
            if (time.time() - start_time) > deadline:
                break
            res_edges, score = local_search(best_edges)
            if res_edges is not None and score > best_score:
                best_score = score
                best_edges = res_edges
            else:
                break

    if best_edges is None:
        return []

    result = [[int(u), int(v)] for u, v in best_edges]
    return result

if __name__ == "__main__":
    pass
