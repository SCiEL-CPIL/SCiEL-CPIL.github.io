import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    n = 16
    all_pairs = list(combinations(range(n), 2))
    n_pairs = len(all_pairs)  # 120

    # Edge index lookup
    edge_index = {}
    for i, (u, v) in enumerate(all_pairs):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    # Precompute witnesses for triangle-freeness
    # witnesses[i] = list of (w, idx_uw, idx_vw) for edge i=(u,v)
    # Triangle exists iff edge i, idx_uw, idx_vw are all present
    witnesses = [[] for _ in range(n_pairs)]
    for i in range(n_pairs):
        u, v = all_pairs[i]
        for w in range(n):
            if w == u or w == v:
                continue
            # (u,w) and (v,w) must exist in edge_index
            if (u, w) in edge_index and (v, w) in edge_index:
                witnesses[i].append((w, edge_index[(u, w)], edge_index[(v, w)]))

    def is_triangle_free(x):
        for i in range(n_pairs):
            if x[i]:
                for (w, iw, jw) in witnesses[i]:
                    if x[iw] and x[jw]:
                        return False
        return True

    def is_3_colorable(adj_bits):
        """Fast 3-colorability check using bitmasks and degree ordering."""
        degrees = [bin(adj_bits[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        colors = [-1] * n

        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            mask = adj_bits[v]
            for c in range(3):
                conflict = False
                m = mask
                while m:
                    lsb = m & -m
                    u = lsb.bit_length() - 1
                    if colors[u] == c:
                        conflict = True
                        break
                    m -= lsb
                if not conflict:
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False

        return backtrack(0)

    def check_3_colorable(x):
        """Check 3-colorability from edge list x."""
        adj_bits = [0] * n
        for i in range(n_pairs):
            if x[i]:
                u, v = all_pairs[i]
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
        return is_3_colorable(adj_bits)

    def edge_count(x):
        return sum(x)

    def random_maximal_tf(seed_offset=0):
        """Generate a random maximal triangle-free graph."""
        x = [0] * n_pairs
        perm = list(range(n_pairs))
        random.shuffle(perm)
        for i in perm:
            u, v = all_pairs[i]
            ok = True
            for (w, iw, jw) in witnesses[i]:
                if x[iw] and x[jw]:
                    ok = False
                    break
            if ok:
                x[i] = 1
        return x

    def local_search(x, time_limit):
        """
        Local search to maximize edge count while maintaining
        triangle-freeness and non-3-colorability.
        """
        best_x = x[:]
        best_score = edge_count(x)
        current_x = x[:]
        current_score = best_score

        temp = 1.0
        cool_rate = 0.9995

        while time.time() < time_limit:
            # Pick a random edge to flip
            i = random.randint(0, n_pairs - 1)
            new_x = current_x[:]
            new_x[i] = 1 - new_x[i]

            # Check triangle-freeness
            if not is_triangle_free(new_x):
                continue

            new_score = edge_count(new_x)

            # Check 3-colorability (only if score improved or we're exploring)
            if new_score >= current_score:
                if check_3_colorable(new_x):
                    continue

            # Acceptance criterion
            if new_score >= current_score:
                current_x = new_x
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_x = new_x[:]
            else:
                # Simulated annealing: accept worse moves with probability
                delta = current_score - new_score
                if delta > 0 and random.random() < (temp / (1.0 + delta)):
                    current_x = new_x
                    current_score = new_score

            temp *= cool_rate
            if temp < 0.01:
                temp = 0.01

        return best_x

    # Known good fallback: Clebsch graph (40 edges)
    # But first, let's try to construct something better

    best_edges = []
    best_count = 0

    # Strategy 1: Try multiple random maximal triangle-free graphs
    # and use local search to improve them
    num_starts = 50
    per_start_time = 110.0 / num_starts

    for s in range(num_starts):
        if time.time() > deadline:
            break

        x = random_maximal_tf(s)
        cnt = edge_count(x)

        # Check if it's already non-3-colorable
        if not check_3_colorable(x):
            if cnt > best_count:
                best_count = cnt
                # Convert to edge list
                best_edges = []
                for i in range(n_pairs):
                    if x[i]:
                        u, v = all_pairs[i]
                        best_edges.append([u, v])

        # Local search to improve
        if time.time() < deadline - 5:
            x_improved = local_search(x, min(time.time() + per_start_time, deadline - 5))
            cnt_improved = edge_count(x_improved)

            # Verify the improved graph is valid
            if is_triangle_free(x_improved) and not check_3_colorable(x_improved):
                if cnt_improved > best_count:
                    best_count = cnt_improved
                    best_edges = []
                    for i in range(n_pairs):
                        if x_improved[i]:
                            u, v = all_pairs[i]
                            best_edges.append([u, v])

    # Strategy 2: Direct edge addition search
    # Start from a base graph and try to add edges one by one
    if time.time() < deadline - 10:
        # Try starting from Clebsch graph and adding edges
        clebsch_x = [0] * n_pairs
        for u in range(n):
            for bit_idx in range(4):
                v = u ^ (1 << bit_idx)
                if u < v:
                    clebsch_x[edge_index[(u, v)]] = 1
            v = u ^ 15
            if u < v:
                clebsch_x[edge_index[(u, v)]] = 1

        if not check_3_colorable(clebsch_x):
            clebsch_cnt = edge_count(clebsch_x)
            if clebsch_cnt > best_count:
                best_count = clebsch_cnt
                best_edges = []
                for i in range(n_pairs):
                    if clebsch_x[i]:
                        u, v = all_pairs[i]
                        best_edges.append([u, v])

            # Try to add more edges
            x_base = clebsch_x[:]
            for _ in range(200):
                if time.time() > deadline - 5:
                    break
                # Try adding a random non-existent edge
                non_edges = [i for i in range(n_pairs) if not x_base[i]]
                if not non_edges:
                    break
                i = random.choice(non_edges)
                new_x = x_base[:]
                new_x[i] = 1
                if is_triangle_free(new_x) and not check_3_colorable(new_x):
                    x_base = new_x
                    cnt_base = edge_count(x_base)
                    if cnt_base > best_count:
                        best_count = cnt_base
                        best_edges = []
                        for j in range(n_pairs):
                            if x_base[j]:
                                u, v = all_pairs[j]
                                best_edges.append([u, v])

    # Final fallback: return Clebsch graph if nothing better found
    if best_count == 0:
        best_edges = []
        for u in range(n):
            for bit_idx in range(4):
                v = u ^ (1 << bit_idx)
                if u < v:
                    best_edges.append([u, v])
            v = u ^ 15
            if u < v:
                best_edges.append([u, v])
        best_count = len(best_edges)

    return best_edges
