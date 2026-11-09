import numpy as np
import time

from autoresearch_suite.science_tcs_expansion import check_lift


def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit

    n = graph['n']
    edges = graph['edges']
    m = n // 2

    # Classify edges: cycle edges connect vertices within same half,
    # matching edges connect i to i+m
    cycle_edges = []
    matching_edges = []
    for i, (u, v) in enumerate(edges):
        if abs(u - v) == m:
            matching_edges.append(i)
        else:
            cycle_edges.append(i)

    # For the signed cycle, build the circulant signed adjacency
    # Cycle edges: (i, i+1 mod m) for i in 0..m-1
    # We need to map each cycle edge to its position in the cycle

    # Build a mapping from cycle edge index -> position in cycle
    # Edge (i, i+1 mod m) has position i (or (i+1) mod m for the wrap-around)
    cycle_pos = {}
    for idx in cycle_edges:
        u, v = edges[idx]
        if u < m:
            # u is in first cycle
            pos = u  # edge from u to (u+1)%m
        else:
            # v is in first cycle (since u>=m, v<m for matching)
            # Actually for cycle edges, both u,v < m or both >= m
            pos = v
        cycle_pos[idx] = pos

    # lambda cache: for each cycle sign pattern, compute eigvalsh of signed cycle
    lambda_cache = {}

    def get_lambda(signed_cycle):
        key = signed_cycle.tobytes()
        if key in lambda_cache:
            return lambda_cache[key]
        C = np.zeros((m, m), dtype=float)
        for i in range(m):
            j = (i + 1) % m
            C[i, j] = signed_cycle[i]
            C[j, i] = signed_cycle[i]
        lam = np.linalg.eigvalsh(C)
        lambda_cache[key] = lam
        return lam

    def eval_pattern(cycle_signs, match_ones):
        lam = get_lambda(cycle_signs)
        m_negs = m - match_ones
        mu = 2 * (match_ones - m_negs)
        return float(np.max(np.maximum(np.abs(lam + mu), np.abs(lam - mu))))

    best_radius = float('inf')
    best_cycle = None
    best_match_ones = None

    # Phase 1: Search cycle signs (2^m if m small)
    if m <= 16:
        for mask in range(1 << m):
            if time.time() > deadline:
                break
            s = np.ones(m, dtype=int)
            for i in range(m):
                if (mask >> i) & 1:
                    s[i] = -1
            if int(np.prod(s)) == 1:
                continue  # disconnected
            # For this cycle, find best matching
            for match_ones in range(m + 1):
                r = eval_pattern(s, match_ones)
                if r < best_radius - 1e-12:
                    best_radius = r
                    best_cycle = s.copy()
                    best_match_ones = match_ones
    else:
        # Random + local search for cycle signs
        rng = np.random.default_rng(12345)
        for _ in range(50000):
            if time.time() > deadline:
                break
            s = rng.choice([-1, 1], size=m)
            if int(np.prod(s)) == 1:
                continue
            for match_ones in range(m + 1):
                r = eval_pattern(s, match_ones)
                if r < best_radius - 1e-12:
                    best_radius = r
                    best_cycle = s.copy()
                    best_match_ones = match_ones
        # Local search
        if best_cycle is not None:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(m):
                    old = best_cycle[i]
                    best_cycle[i] = -old
                    if int(np.prod(best_cycle)) == -1:
                        for match_ones in range(m + 1):
                            r = eval_pattern(best_cycle, match_ones)
                            if r < best_radius - 1e-12:
                                best_radius = r
                                best_match_ones = match_ones
                                improved = True
                                break
                    best_cycle[i] = old

    if best_cycle is None:
        # Fallback
        best_cycle = np.ones(m, dtype=int)
        best_cycle[0] = -1
        best_match_ones = m
        best_radius = eval_pattern(best_cycle, best_match_ones)

    # Build final signs
    signs = np.ones(2 * m, dtype=int)
    for idx in cycle_edges:
        pos = cycle_pos[idx]
        signs[idx] = best_cycle[pos]

    # Matching edges: need exactly best_match_ones of them to be +1
    # Assign +1 to the first best_match_ones matching edges, -1 to rest
    for k, idx in enumerate(matching_edges):
        if k < best_match_ones:
            signs[idx] = 1
        else:
            signs[idx] = -1

    # Verify
    try:
        radius, details = check_lift(graph, signs)
        return signs
    except Exception:
        # Fallback: try a few simple valid patterns
        for trial in range(20):
            s = np.ones(2 * m, dtype=int)
            s[0] = -1  # ensure odd cycle product
            if trial > 0:
                s[trial % (2 * m)] = -1
            try:
                check_lift(graph, s)
                return s
            except Exception:
                pass
        return signs
