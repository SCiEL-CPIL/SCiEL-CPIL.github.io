import time
import numpy as np

def sign_edges(graph, budget):
    deadline = time.time() + 85.0
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2

    # Identify edge types
    top_edge_indices = []
    bottom_edge_indices = []
    match_edge_indices = []
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_edge_indices.append(idx)
        elif u >= m and v >= m:
            bottom_edge_indices.append(idx)
        else:
            match_edge_indices.append(idx)

    # Symmetry reduction: fix bottom cycle signs = top cycle signs
    # So we only need to assign signs to top cycle (m edges) + matching (m edges) = 2m variables
    num_vars = 2 * m

    def build_full_signs(vars_signs):
        """Expand reduced variables to full edge signs using symmetry."""
        signs = np.ones(num_edges, dtype=np.int8)
        for i, idx in enumerate(top_edge_indices):
            signs[idx] = vars_signs[i]
        for i, idx in enumerate(bottom_edge_indices):
            signs[idx] = vars_signs[i]  # symmetry: bottom = top
        for i, idx in enumerate(match_edge_indices):
            signs[idx] = vars_signs[m + i]
        return signs

    def build_signed_matrix(signs):
        S = np.zeros((n, n), dtype=np.float64)
        for idx, (u, v) in enumerate(edges):
            s = float(signs[idx])
            S[u, v] = s
            S[v, u] = s
        return S

    def spectral_radius(signs):
        S = build_signed_matrix(signs)
        ev = np.linalg.eigvalsh(S)
        return float(np.max(np.abs(ev)))

    def spectral_radius_from_vars(vars_signs):
        full = build_full_signs(vars_signs)
        return spectral_radius(full)

    best_signs = None
    best_obj = float('inf')

    def try_signs(vars_signs):
        nonlocal best_signs, best_obj
        obj = spectral_radius_from_vars(vars_signs)
        if obj < best_obj:
            best_obj = obj
            best_signs = build_full_signs(vars_signs).copy()

    def local_search(vars_start):
        """Local search on reduced 2m-dimensional space with 1-flip and 2-flip moves."""
        cur = vars_start.copy()
        cur_obj = spectral_radius_from_vars(cur)
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_signs = build_full_signs(cur).copy()

        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            # 1-flip moves
            best_flip = -1
            best_flip_obj = cur_obj
            for e in range(num_vars):
                old = cur[e]
                cur[e] = -old
                o = spectral_radius_from_vars(cur)
                cur[e] = old
                if o < best_flip_obj - 1e-12:
                    best_flip_obj = o
                    best_flip = e

            if best_flip >= 0:
                cur[best_flip] = -cur[best_flip]
                cur_obj = best_flip_obj
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best_signs = build_full_signs(cur).copy()
                improved = True
                continue

            # 2-flip moves
            found_pair = False
            for e1 in range(num_vars):
                if time.time() > deadline - 3.0:
                    break
                for e2 in range(e1 + 1, num_vars):
                    if time.time() > deadline - 3.0:
                        break
                    old1, old2 = cur[e1], cur[e2]
                    cur[e1], cur[e2] = -old1, -old2
                    o = spectral_radius_from_vars(cur)
                    cur[e1], cur[e2] = old1, old2
                    if o < cur_obj - 1e-12:
                        cur[e1], cur[e2] = -old1, -old2
                        cur_obj = o
                        if cur_obj < best_obj:
                            best_obj = cur_obj
                            best_signs = build_full_signs(cur).copy()
                        improved = True
                        found_pair = True
                        break
                if found_pair:
                    break

            if not improved:
                # Try 3-flip moves as escape
                found_triple = False
                for e1 in range(num_vars):
                    if time.time() > deadline - 3.0:
                        break
                    for e2 in range(e1 + 1, num_vars):
                        if time.time() > deadline - 3.0:
                            break
                        for e3 in range(e2 + 1, num_vars):
                            if time.time() > deadline - 3.0:
                                break
                            old1, old2, old3 = cur[e1], cur[e2], cur[e3]
                            cur[e1], cur[e2], cur[e3] = -old1, -old2, -old3
                            o = spectral_radius_from_vars(cur)
                            cur[e1], cur[e2], cur[e3] = old1, old2, old3
                            if o < cur_obj - 1e-12:
                                cur[e1], cur[e2], cur[e3] = -old1, -old2, -old3
                                cur_obj = o
                                if cur_obj < best_obj:
                                    best_obj = cur_obj
                                    best_signs = build_full_signs(cur).copy()
                                improved = True
                                found_triple = True
                                break
                        if found_triple:
                            break
                    if found_triple:
                        break

    # Generate structured starting patterns in reduced space
    def pattern_all_plus():
        return np.ones(num_vars, dtype=np.int8)

    def pattern_alt_top():
        v = np.ones(num_vars, dtype=np.int8)
        for i in range(m):
            v[i] = 1 if i % 2 == 0 else -1
        return v

    def pattern_alt_match():
        v = np.ones(num_vars, dtype=np.int8)
        for i in range(m):
            v[m + i] = 1 if i % 2 == 0 else -1
        return v

    def pattern_alt_both():
        v = np.ones(num_vars, dtype=np.int8)
        for i in range(m):
            v[i] = 1 if i % 2 == 0 else -1
            v[m + i] = 1 if i % 2 == 0 else -1
        return v

    def pattern_alt_both_shifted():
        v = np.ones(num_vars, dtype=np.int8)
        for i in range(m):
            v[i] = 1 if i % 2 == 0 else -1
            v[m + i] = -1 if i % 2 == 0 else 1
        return v

    def pattern_all_minus_top():
        v = np.ones(num_vars, dtype=np.int8)
        for i in range(m):
            v[i] = -1
        return v

    def pattern_all_minus_match():
        v = np.ones(num_vars, dtype=np.int8)
        for i in range(m):
            v[m + i] = -1
        return v

    patterns = [
        pattern_all_plus(),
        pattern_alt_top(),
        pattern_alt_match(),
        pattern_alt_both(),
        pattern_alt_both_shifted(),
        pattern_all_minus_top(),
        pattern_all_minus_match(),
    ]

    for p in patterns:
        if time.time() > deadline - 10.0:
            break
        local_search(p)

    # Random restarts in reduced space
    if time.time() < deadline - 15.0:
        rng = np.random.default_rng(3)
        max_restarts = 30
        for _ in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            v = rng.choice([-1, 1], size=num_vars)
            local_search(v)

    # Final polish: exhaustive 1-flip check on best
    if best_signs is not None and time.time() < deadline - 1.0:
        # Convert best back to reduced form for final check
        # Just do a final local search pass
        # Reconstruct reduced vars from best_signs
        cur = np.ones(num_vars, dtype=np.int8)
        for i, idx in enumerate(top_edge_indices):
            cur[i] = best_signs[idx]
        for i, idx in enumerate(match_edge_indices):
            cur[m + i] = best_signs[idx]
        local_search(cur)

    if best_signs is None:
        best_signs = np.array([1] * num_edges, dtype=np.int8)

    return best_signs.astype(np.int8)
