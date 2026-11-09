import time
import random
import math
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    n = 16
    target_edges = 65

    # Precompute all triangles
    all_tris = list(itertools.combinations(range(n), 3))
    num_tris = len(all_tris)

    # Precompute for each edge, the list of triangle indices containing it
    # And for each vertex, the list of triangle indices containing it
    edge_to_tris = {}
    vert_to_tris = {v: [] for v in range(n)}
    for i, (u, v, w) in enumerate(all_tris):
        for e in [(u, v), (u, w), (v, w)]:
            e_sorted = (min(e), max(e))
            edge_to_tris.setdefault(e_sorted, []).append(i)
        for vert in (u, v, w):
            vert_to_tris[vert].append(i)

    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    all_edges_set = set(all_edges)

    def build_initial_graph(seed):
        """Build K_{8,8} + one random intra-partition edge."""
        adj = [0] * n
        deg = [0] * n
        edges = set()

        rng_local = random.Random(seed)

        for u in range(8):
            for v in range(8, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                edges.add((u, v))

        # Add one random intra-partition edge
        while True:
            u = rng_local.randint(0, 7)
            v = rng_local.randint(0, 7)
            if u != v:
                e = (min(u, v), max(u, v))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                edges.add(e)
                break

        return adj, deg, edges

    def compute_tri_sums(adj, deg):
        """Compute triangle sums for all triangles."""
        tri_sum = [0] * num_tris
        for i, (u, v, w) in enumerate(all_tris):
            if (adj[u] >> v) & 1 and (adj[u] >> w) & 1 and (adj[v] >> w) & 1:
                tri_sum[i] = deg[u] + deg[v] + deg[w]
        return tri_sum

    def get_non_edges(adj):
        """Return list of non-edges."""
        non_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if not ((adj[u] >> v) & 1):
                    non_edges.append((u, v))
        return non_edges

    # Run multiple restarts with different strategies
    best_overall_score = float('inf')
    best_overall_edges = None

    # Phase 1: Multiple restarts with simulated annealing (current approach)
    num_restarts = 6
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break

        # Initialize
        adj, deg, current_edges = build_initial_graph(42 + restart * 1000)
        tri_sum = compute_tri_sums(adj, deg)
        current_score = max(tri_sum)

        best_score = current_score
        best_edges = set(current_edges)

        rng = random.Random(42 + restart * 7919)

        T_start = 6.0
        T_end = 0.001

        # Precompute non-edges for faster sampling
        non_edges = get_non_edges(adj)

        i = 0
        n_iter_total = 200000
        restart_deadline = min(time.time() + (deadline - time.time()) / max(num_restarts - restart, 1) - 2, deadline - 30)

        while time.time() < restart_deadline:
            i += 1
            T = T_start * (T_end / T_start) ** (min(i, n_iter_total) / n_iter_total)

            # Move Selection - bias toward edges in high-sum triangles
            if rng.random() < 0.7 and current_score > 0:
                max_val = current_score
                candidates = [idx for idx, s in enumerate(tri_sum) if s == max_val]
                if candidates:
                    idx = rng.choice(candidates)
                    u, v, w = all_tris[idx]
                    e_remove = rng.choice([(u, v), (u, w), (v, w)])
                    e_remove = (min(e_remove), max(e_remove))
                else:
                    e_remove = rng.choice(list(current_edges))
                    e_remove = (min(e_remove), max(e_remove))
            else:
                e_remove = rng.choice(list(current_edges))
                e_remove = (min(e_remove), max(e_remove))

            if not non_edges:
                break
            e_add = rng.choice(non_edges)

            if e_add == e_remove:
                continue

            u1, v1 = e_remove
            u2, v2 = e_add

            prev_score = current_score

            # Apply Move
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            deg[u1] -= 1
            deg[v1] -= 1
            current_edges.remove(e_remove)

            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            deg[u2] += 1
            deg[v2] += 1
            current_edges.add(e_add)

            # Update non_edges list
            non_edges.remove(e_add)
            non_edges.append(e_remove)

            # Update Triangle Sums Incrementally
            update_indices = set()
            if e_remove in edge_to_tris:
                update_indices.update(edge_to_tris[e_remove])
            if e_add in edge_to_tris:
                update_indices.update(edge_to_tris[e_add])
            for vert in (u1, v1, u2, v2):
                update_indices.update(vert_to_tris[vert])

            for idx in update_indices:
                u, v, w = all_tris[idx]
                if (adj[u] >> v) & 1 and (adj[u] >> w) & 1 and (adj[v] >> w) & 1:
                    tri_sum[idx] = deg[u] + deg[v] + deg[w]
                else:
                    tri_sum[idx] = 0

            current_score = max(tri_sum)

            # Acceptance
            delta = current_score - prev_score
            if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                if current_score < best_score:
                    best_score = current_score
                    best_edges = set(current_edges)
            else:
                # Revert
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                deg[u1] += 1
                deg[v1] += 1
                current_edges.add(e_remove)

                adj[u2] &= ~(1 << v2)
                adj[v2] &= ~(1 << u2)
                deg[u2] -= 1
                deg[v2] -= 1
                current_edges.remove(e_add)

                non_edges.remove(e_remove)
                non_edges.append(e_add)

                for idx in update_indices:
                    u, v, w = all_tris[idx]
                    if (adj[u] >> v) & 1 and (adj[u] >> w) & 1 and (adj[v] >> w) & 1:
                        tri_sum[idx] = deg[u] + deg[v] + deg[w]
                    else:
                        tri_sum[idx] = 0

                current_score = prev_score

        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall_edges = best_edges

    # Phase 2: Targeted local search - try to reduce score below current best
    # Use a more aggressive strategy focused on high-degree-sum triangles
    if time.time() < deadline - 10 and best_overall_edges is not None:
        adj = [0] * n
        deg = [0] * n
        current_edges = set(best_overall_edges)
        for u, v in current_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1

        tri_sum = compute_tri_sums(adj, deg)
        current_score = max(tri_sum)
        best_score = current_score
        best_edges = set(current_edges)

        rng = random.Random(99999)
        non_edges = get_non_edges(adj)

        # Aggressive local search: try to find and break high-sum triangles
        polish_deadline = deadline - 5
        no_improve = 0
        max_no_improve = 5000

        while time.time() < polish_deadline and no_improve < max_no_improve:
            # Find the worst triangle
            max_val = current_score
            candidates = [idx for idx, s in enumerate(tri_sum) if s == max_val]
            if not candidates:
                break

            idx = rng.choice(candidates)
            u, v, w = all_tris[idx]

            # Try to break this triangle by removing one of its edges
            # and adding a non-edge that doesn't create worse triangles
            improved = False
            for e_remove in [(u, v), (u, w), (v, w)]:
                e_remove = (min(e_remove), max(e_remove))
                if e_remove not in current_edges:
                    continue

                # Try several random non-edges to add
                for _ in range(20):
                    if not non_edges:
                        break
                    e_add = rng.choice(non_edges)
                    if e_add == e_remove:
                        continue

                    u1, v1 = e_remove
                    u2, v2 = e_add

                    # Apply
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    deg[u1] -= 1
                    deg[v1] -= 1
                    current_edges.remove(e_remove)

                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    deg[u2] += 1
                    deg[v2] += 1
                    current_edges.add(e_add)

                    non_edges.remove(e_add)
                    non_edges.append(e_remove)

                    # Update triangle sums
                    update_indices = set()
                    if e_remove in edge_to_tris:
                        update_indices.update(edge_to_tris[e_remove])
                    if e_add in edge_to_tris:
                        update_indices.update(edge_to_tris[e_add])
                    for vert in (u1, v1, u2, v2):
                        update_indices.update(vert_to_tris[vert])

                    for tri_idx in update_indices:
                        tu, tv, tw = all_tris[tri_idx]
                        if (adj[tu] >> tv) & 1 and (adj[tu] >> tw) & 1 and (adj[tv] >> tw) & 1:
                            tri_sum[tri_idx] = deg[tu] + deg[tv] + deg[tw]
                        else:
                            tri_sum[tri_idx] = 0

                    new_score = max(tri_sum)

                    if new_score < current_score:
                        current_score = new_score
                        no_improve = 0
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_edges = set(current_edges)
                        break
                    else:
                        # Revert
                        adj[u1] |= (1 << v1)
                        adj[v1] |= (1 << u1)
                        deg[u1] += 1
                        deg[v1] += 1
                        current_edges.add(e_remove)

                        adj[u2] &= ~(1 << v2)
                        adj[v2] &= ~(1 << u2)
                        deg[u2] -= 1
                        deg[v2] -= 1
                        current_edges.remove(e_add)

                        non_edges.remove(e_remove)
                        non_edges.append(e_add)

                        for tri_idx in update_indices:
                            tu, tv, tw = all_tris[tri_idx]
                            if (adj[tu] >> tv) & 1 and (adj[tu] >> tw) & 1 and (adj[tv] >> tw) & 1:
                                tri_sum[tri_idx] = deg[tu] + deg[tv] + deg[tw]
                            else:
                                tri_sum[tri_idx] = 0

                        current_score = max_val  # restore

                if improved:
                    break

            if not improved:
                no_improve += 1

        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall_edges = best_edges

    if best_overall_edges is None:
        # Fallback: use the initial construction
        adj, deg, current_edges = build_initial_graph(42)
        best_overall_edges = current_edges

    edges_list = [list(e) for e in best_overall_edges]
    return {'n': n, 'edges': edges_list}
