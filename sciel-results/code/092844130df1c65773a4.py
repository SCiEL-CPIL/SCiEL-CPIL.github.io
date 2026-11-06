"""Search-based construction for 16-vertex 65-edge graph minimizing max triangle degree sum.

Strategy:
- Start from K_{8,8} + one intra-A edge (base score 24).
- Use bitmask adjacency for fast triangle enumeration.
- Local search: edge-swap moves (remove one edge, add one non-edge) that strictly reduce
  the maximum triangle degree sum.
- Multiple random restarts with perturbation to escape local optima.
- Tight deadline guard with safety margin.
"""
import time
import random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n = 16
    target_edges = 65

    def make_base():
        """K_{8,8} + edge (0,1). Part A: 0-7, Part B: 8-15."""
        edges = set()
        for u in range(8):
            for v in range(8, 16):
                edges.add((u, v))
        edges.add((0, 1))
        return edges

    def compute_score(es):
        """Compute (max_triangle_deg_sum, deg_list, adj_bitmask_list)."""
        adj = [0] * n
        deg = [0] * n
        for (u, v) in es:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1

        best = 0
        for (u, v) in es:
            common = adj[u] & adj[v]
            c = common
            while c:
                w = (c & -c).bit_length() - 1
                c &= c - 1
                s = deg[u] + deg[v] + deg[w]
                if s > best:
                    best = s
        return best, deg, adj

    def compute_score_from_state(deg, adj, es):
        """Same as compute_score but reuses deg/adj if consistent; here just recompute."""
        return compute_score(es)

    def local_search(es, score, deg, adj, max_iters=30000):
        """
        Edge-swap local search.
        A move: remove edge (a,b), add non-edge (c,d).
        Accept if new max triangle degree sum < current.
        """
        current_es = set(es)
        current_score = score
        current_deg = list(deg)
        current_adj = list(adj)

        def get_non_edges(es_set):
            ne = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) not in es_set:
                        ne.append((u, v))
            return ne

        non_edges = get_non_edges(current_es)
        improved = True
        iters = 0

        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1

            # Find worst triangle
            worst_sum = 0
            worst_tri = None
            for (u, v) in current_es:
                common = current_adj[u] & current_adj[v]
                c = common
                while c:
                    w = (c & -c).bit_length() - 1
                    c &= c - 1
                    s = current_deg[u] + current_deg[v] + current_deg[w]
                    if s > worst_sum:
                        worst_sum = s
                        worst_tri = (u, v, w)

            if worst_tri is None:
                break

            u, v, w = worst_tri
            tri_edges = [(min(u, v), max(u, v)), (min(u, w), max(u, w)), (min(v, w), max(v, w))]
            # Sort by endpoint degree sum descending (attack high-degree edges first)
            tri_edges.sort(key=lambda e: current_deg[e[0]] + current_deg[e[1]], reverse=True)

            found_swap = False
            for (a, b) in tri_edges:
                if (a, b) not in current_es:
                    continue
                # Try removing (a,b)
                new_deg_a = current_deg[a] - 1
                new_deg_b = current_deg[b] - 1

                # Candidate non-edges sorted by endpoint degree sum ascending
                candidates = []
                for (c, d) in non_edges:
                    if (c, d) == (a, b):
                        continue
                    candidates.append((current_deg[c] + current_deg[d], c, d))
                candidates.sort()

                for (_, c, d) in candidates[:80]:
                    if time.time() > deadline:
                        break
                    trial_es = current_es - {(a, b)} | {(c, d)}
                    trial_score, trial_deg, trial_adj = compute_score(trial_es)
                    if trial_score < current_score:
                        current_es = trial_es
                        current_score = trial_score
                        current_deg = trial_deg
                        current_adj = trial_adj
                        non_edges = get_non_edges(current_es)
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    break

            if not found_swap:
                # Stuck: try a few random perturbations to escape
                if iters % 50 == 0 and random.random() < 0.6 and non_edges:
                    for _ in range(3):
                        if time.time() > deadline:
                            break
                        rem = random.choice(list(current_es))
                        add = random.choice(non_edges)
                        if add in current_es:
                            continue
                        trial_es = current_es - {rem} | {add}
                        trial_score, trial_deg, trial_adj = compute_score(trial_es)
                        if trial_score <= current_score + 1:
                            current_es = trial_es
                            current_score = trial_score
                            current_deg = trial_deg
                            current_adj = trial_adj
                            non_edges = get_non_edges(current_es)
                            improved = True
                            break

        return current_es, current_score, current_deg, current_adj

    # --- Main search ---
    rng = random.Random(12345)

    # Start from base
    best_es = make_base()
    best_score, best_deg, best_adj = compute_score(best_es)

    # First local search from base
    best_es, best_score, best_deg, best_adj = local_search(best_es, best_score, best_deg, best_adj, max_iters=50000)

    # Multiple restarts with perturbation
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break

        # Perturb the best solution
        start_es = set(best_es)
        num_swaps = rng.randint(2, 5)
        for _ in range(num_swaps):
            ne = get_non_edges(start_es)
            if not ne:
                break
            rem = rng.choice(list(start_es))
            add = rng.choice(ne)
            start_es = start_es - {rem} | {add}

        start_score, start_deg, start_adj = compute_score(start_es)

        # Local search from perturbed start
        es_r, score_r, deg_r, adj_r = local_search(start_es, start_score, start_deg, start_adj, max_iters=20000)

        if score_r < best_score:
            best_es = es_r
            best_score = score_r
            best_deg = deg_r
            best_adj = adj_r

    # Final polish: one more local search with more iterations
    best_es, best_score, best_deg, best_adj = local_search(best_es, best_score, best_deg, best_adj, max_iters=50000)

    # Verify
    final_score, _, _ = compute_score(best_es)
    if len(best_es) != target_edges:
        best_es = make_base()
        final_score = 24

    # Convert to list of lists
    edges_list = [[u, v] for (u, v) in sorted(best_es)]

    return {'n': n, 'edges': edges_list}


# Quick self-test
if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    elapsed = time.time() - t0
    assert result['n'] == 16
    assert len(result['edges']) == 65
    n = 16
    deg = [0] * n
    adj = [0] * n
    for (u, v) in result['edges']:
        deg[u] += 1
        deg[v] += 1
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    best = 0
    for (u, v) in result['edges']:
        common = adj[u] & adj[v]
        c = common
        while c:
            w = (c & -c).bit_length() - 1
            c &= c - 1
            s = deg[u] + deg[v] + deg[w]
            if s > best:
                best = s
    print(f"Score: {best}, Time: {elapsed:.1f}s")
