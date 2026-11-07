"""Rewritten program for spectral graph lifts with optimized connectivity check."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product as iproduct


def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift

    deadline = time.time() + 82.0
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)

    # Identify edge types: top cycle, bottom cycle, matching
    top_edges = []  # (edge_idx, u, v) where u,v in 0..m-1
    bot_edges = []  # (edge_idx, u, v) where u,v in m..2m-1
    match_edges = []  # (edge_idx, i) where edge is (i, i+m)

    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_edges.append((idx, u, v))
        elif u >= m and v >= m:
            top_idx_u = u - m
            top_idx_v = v - m
            bot_edges.append((idx, top_idx_u, top_idx_v))
        else:
            i = min(u, v)
            match_edges.append((idx, i))

    # Verify structure
    assert len(top_edges) == m, f"Expected {m} top edges, got {len(top_edges)}"
    assert len(bot_edges) == m, f"Expected {m} bot edges, got {len(bot_edges)}"
    assert len(match_edges) == m, f"Expected {m} match edges, got {len(match_edges)}"

    # Sort matching edges by i
    match_edges.sort(key=lambda x: x[1])

    def build_matrix(a, b, c):
        """Build the 2m x 2m signed adjacency matrix."""
        S = np.zeros((n, n), dtype=np.float64)
        for idx, u, v in top_edges:
            S[u, v] = a[u]
            S[v, u] = a[u]
        for idx, u, v in bot_edges:
            S[m + u, m + v] = b[u]
            S[m + v, m + u] = b[u]
        for idx, i in match_edges:
            S[i, m + i] = c[i]
            S[m + i, i] = c[i]
        return S

    def spectral_radius(a, b, c):
        """Compute spectral radius."""
        S = build_matrix(a, b, c)
        eigvals = np.linalg.eigvalsh(S)
        return np.max(np.abs(eigvals))

    def is_connected(a, b, c):
        """Check connectivity: product around outer cycle must be -1."""
        prod = 1
        for x in a:
            prod *= x
        for x in b:
            prod *= x
        for x in c:
            prod *= x
        return prod == -1

    def signs_from_abc(a, b, c):
        """Convert a, b, c arrays to full signs array."""
        signs = np.zeros(num_edges, dtype=np.int8)
        for idx, u, v in top_edges:
            signs[idx] = a[u]
        for idx, u, v in bot_edges:
            signs[idx] = b[u]
        for idx, i in match_edges:
            signs[idx] = c[i]
        return signs

    def evaluate_abc(a, b, c):
        """Evaluate a, b, c configuration. Returns radius or None if invalid."""
        if not is_connected(a, b, c):
            return None
        return spectral_radius(a, b, c)

    best_signs = None
    best_radius = np.inf

    # For small m (m=4), enumerate exhaustively
    if m <= 4:
        free_vars_count = 2 * m + (m - 2)
        
        if free_vars_count <= 20:
            for combo in iproduct([-1, 1], repeat=free_vars_count):
                if time.time() > deadline:
                    break
                a = np.array(combo[0:m], dtype=np.int8)
                b = np.array(combo[m:2*m], dtype=np.int8)
                c_prefix = list(combo[2*m:])
                
                # Determine c[0] and c[m-1] for connectivity
                # WLOG fix c[0] = +1
                c = np.zeros(m, dtype=np.int8)
                c[0] = 1
                for j, val in enumerate(c_prefix):
                    c[j + 1] = val
                
                # Determine c[m-1]
                prod = 1
                for x in a:
                    prod *= x
                for x in b:
                    prod *= x
                for x in c[:-1]:
                    prod *= x
                c[m - 1] = -prod

                radius = spectral_radius(a, b, c)
                if radius < best_radius:
                    best_radius = radius
                    best_signs = signs_from_abc(a, b, c)

        if best_signs is not None:
            try:
                radius, details = check_lift(graph, best_signs)
                return best_signs
            except:
                pass

    # For larger m or fallback: local search
    rng = np.random.default_rng(3)
    
    # Initial: try seed's random approach
    for _ in range(budget):
        if time.time() > deadline:
            break
        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        a = np.zeros(m, dtype=np.int8)
        b = np.zeros(m, dtype=np.int8)
        c = np.zeros(m, dtype=np.int8)
        for idx, u, v in top_edges:
            a[u] = signs[idx]
        for idx, u, v in bot_edges:
            b[u] = signs[idx]
        for idx, i in match_edges:
            c[i] = signs[idx]
        
        radius = evaluate_abc(a, b, c)
        if radius is not None and radius < best_radius:
            best_radius = radius
            best_signs = signs_from_abc(a, b, c)

    # Also try structured initial solutions
    for i in range(m):
        if time.time() > deadline:
            break
        a = np.ones(m, dtype=np.int8)
        b = np.ones(m, dtype=np.int8)
        c = np.ones(m, dtype=np.int8)
        c[i] = -1
        radius = evaluate_abc(a, b, c)
        if radius is not None and radius < best_radius:
            best_radius = radius
            best_signs = signs_from_abc(a, b, c)

    if best_signs is None:
        a = np.ones(m, dtype=np.int8)
        b = np.ones(m, dtype=np.int8)
        c = np.ones(m, dtype=np.int8)
        c[0] = -1
        best_signs = signs_from_abc(a, b, c)
        return best_signs

    # Local search on a, b, c representation
    current_signs = best_signs.copy()
    current_radius = best_radius

    cur_a = np.zeros(m, dtype=np.int8)
    cur_b = np.zeros(m, dtype=np.int8)
    cur_c = np.zeros(m, dtype=np.int8)
    for idx, u, v in top_edges:
        cur_a[u] = current_signs[idx]
    for idx, u, v in bot_edges:
        cur_b[u] = current_signs[idx]
    for idx, i in match_edges:
        cur_c[i] = current_signs[idx]

    # Local search: flip one variable at a time
    while time.time() < deadline:
        improved = False
        
        # 1-flip moves
        for var_type in ['a', 'b', 'c']:
            if time.time() > deadline:
                break
            for i in range(m):
                if time.time() > deadline:
                    break
                
                a = cur_a.copy()
                b = cur_b.copy()
                c = cur_c.copy()
                
                if var_type == 'a':
                    a[i] = -a[i]
                elif var_type == 'b':
                    b[i] = -b[i]
                else:
                    c[i] = -c[i]
                
                if not is_connected(a, b, c):
                    continue
                
                radius = spectral_radius(a, b, c)
                if radius < current_radius - 1e-10:
                    cur_a, cur_b, cur_c = a, b, c
                    current_radius = radius
                    current_signs = signs_from_abc(a, b, c)
                    best_radius = radius
                    best_signs = current_signs.copy()
                    improved = True

        if improved:
            continue

        # 2-flip moves
        two_flip_improved = False
        for i in range(m):
            if time.time() > deadline:
                break
            for j in range(m):
                if time.time() > deadline:
                    break
                for vi in range(3):
                    if time.time() > deadline:
                        break
                    for vj in range(vi, 3):
                        if time.time() > deadline:
                            break
                        
                        a = cur_a.copy()
                        b = cur_b.copy()
                        c = cur_c.copy()
                        
                        if vi == 0:
                            a[i] = -a[i]
                        elif vi == 1:
                            b[i] = -b[i]
                        else:
                            c[i] = -c[i]
                        
                        if vj == 0:
                            a[j] = -a[j]
                        elif vj == 1:
                            b[j] = -b[j]
                        else:
                            c[j] = -c[j]
                        
                        if not is_connected(a, b, c):
                            continue
                        
                        radius = spectral_radius(a, b, c)
                        if radius < current_radius - 1e-10:
                            cur_a, cur_b, cur_c = a, b, c
                            current_radius = radius
                            current_signs = signs_from_abc(a, b, c)
                            best_radius = radius
                            best_signs = current_signs.copy()
                            two_flip_improved = True
                            break
                    if two_flip_improved:
                        break
                if two_flip_improved:
                    break
            if two_flip_improved:
                break
        
        if two_flip_improved:
            continue

        # Random restarts
        restart_improved = False
        for _ in range(500):
            if time.time() > deadline:
                break
            
            a = rng.choice([-1, 1], size=m).astype(np.int8)
            b = rng.choice([-1, 1], size=m).astype(np.int8)
            c = rng.choice([-1, 1], size=m).astype(np.int8)
            
            if not is_connected(a, b, c):
                continue
            
            radius = spectral_radius(a, b, c)
            if radius < current_radius - 1e-10:
                cur_a, cur_b, cur_c = a, b, c
                current_radius = radius
                current_signs = signs_from_abc(a, b, c)
                best_radius = radius
                best_signs = current_signs.copy()
                restart_improved = True
                break
        
        if not restart_improved:
            for _ in range(2000):
                if time.time() > deadline:
                    break
                
                a = rng.choice([-1, 1], size=m).astype(np.int8)
                b = rng.choice([-1, 1], size=m).astype(np.int8)
                c = rng.choice([-1, 1], size=m).astype(np.int8)
                
                if not is_connected(a, b, c):
                    continue
                
                radius = spectral_radius(a, b, c)
                if radius < current_radius - 1e-10:
                    cur_a, cur_b, cur_c = a, b, c
                    current_radius = radius
                    current_signs = signs_from_abc(a, b, c)
                    best_radius = radius
                    best_signs = current_signs.copy()
                    restart_improved = True
                    break
            
            if not restart_improved:
                break

    # Final verification
    try:
        radius, details = check_lift(graph, best_signs)
    except:
        pass

    return best_signs
# EVOLVE-BLOCK-END
