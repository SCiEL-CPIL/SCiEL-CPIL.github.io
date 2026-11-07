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
    # Top cycle: edges between vertices 0..m-1 (consecutive, wrapping)
    # Bottom cycle: edges between vertices m..2m-1 (consecutive, wrapping)
    # Matching: edges (i, i+m)
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
            # Matching edge
            i = min(u, v)
            match_edges.append((idx, i))

    # Verify structure
    assert len(top_edges) == m, f"Expected {m} top edges, got {len(top_edges)}"
    assert len(bot_edges) == m, f"Expected {m} bot edges, got {len(bot_edges)}"
    assert len(match_edges) == m, f"Expected {m} match edges, got {len(match_edges)}"

    # Sort matching edges by i
    match_edges.sort(key=lambda x: x[1])
    match_idx_by_i = {i: idx for idx, i in match_edges}

    # WLOG: fix c_0 = +1 (global sign flip invariance)
    # Connectivity: product(a) * product(b) * product(c_1..c_{m-1}) = -1
    # So c_{m-1} is determined: c_{m-1} = -1 / (product(a)*product(b)*product(c_1..c_{m-2}))
    # In terms of signs: c_{m-1} = -(product(a) * product(b) * product(c_1..c_{m-2}))

    # Free variables: a[0..m-1], b[0..m-1], c[1..m-2]
    # Determined: c[0] = +1, c[m-1] = f(a, b, c[1..m-2])
    # Total free: 2m + (m-2) = 3m - 2

    def build_matrix(a, b, c):
        """Build the 2m x 2m signed adjacency matrix.
        a: array of m signs for top cycle
        b: array of m signs for bottom cycle  
        c: array of m signs for matching
        """
        S = np.zeros((n, n), dtype=np.float64)
        # Top cycle edges
        for idx, u, v in top_edges:
            S[u, v] = a[u]
            S[v, u] = a[u]
        # Bottom cycle edges
        for idx, u, v in bot_edges:
            S[m + u, m + v] = b[u]
            S[m + v, m + u] = b[u]
        # Matching edges
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
        """Check connectivity: product around outer cycle must be -1.
        Outer cycle: a_0*a_1*...*a_{m-1} * b_{m-1}*b_{m-2}*...*b_0 * c_0*c_1*...*c_{m-1}
        = product(a) * product(b) * product(c)  (since product(b) in reverse = product(b))
        """
        prod = 1
        for x in a:
            prod *= x
        for x in b:
            prod *= x
        for x in c:
            prod *= x
        return prod == -1

    def determine_c_last(a, b, c_prefix):
        """Given c[0..m-2] (with c[0]=+1 fixed), determine c[m-1] for connectivity.
        Need: product(a)*product(b)*c[0]*c[1]*...*c[m-2]*c[m-1] = -1
        So: c[m-1] = -1 / (product(a)*product(b)*c[0]*...*c[m-2])
        Since all are +/-1: c[m-1] = -(product(a)*product(b)*c[0]*...*c[m-2])
        """
        prod = 1
        for x in a:
            prod *= x
        for x in b:
            prod *= x
        for x in c_prefix:
            prod *= x
        return -prod

    # For small m (m=4), enumerate exhaustively
    # Free vars: a[0..3], b[0..3], c[1] (c[0]=1 fixed, c[3] determined)
    # That's 2^9 = 512 possibilities
    if m <= 4:
        best_signs = None
        best_radius = np.inf
        free_vars_count = 2 * m + (m - 2)  # a, b, c[1..m-2]
        
        if free_vars_count <= 20:
            for combo in iproduct([-1, 1], repeat=free_vars_count):
                if time.time() > deadline:
                    break
                a = np.array(combo[0:m], dtype=np.int8)
                b = np.array(combo[m:2*m], dtype=np.int8)
                c_prefix = list(combo[2*m:])  # c[1..m-2]
                c_last = determine_c_last(a, b, c_prefix)
                c = np.zeros(m, dtype=np.int8)
                c[0] = 1
                for j, val in enumerate(c_prefix):
                    c[j + 1] = val
                c[m - 1] = c_last

                radius = spectral_radius(a, b, c)
                if radius < best_radius:
                    best_radius = radius
                    # Build full signs array
                    signs = np.zeros(num_edges, dtype=np.int8)
                    for idx, u, v in top_edges:
                        signs[idx] = a[u]
                    for idx, u, v in bot_edges:
                        signs[idx] = b[u]
                    for idx, i in match_edges:
                        signs[idx] = c[i]
                    best_signs = signs.copy()
        else:
            # Too many to enumerate, fall through to local search
            best_signs = None
            best_radius = np.inf

        if best_signs is not None:
            # Verify with check_lift
            try:
                radius, details = check_lift(graph, best_signs)
                return best_signs
            except:
                pass

    # For larger m or fallback: local search
    rng = np.random.default_rng(3)
    
    best_signs = None
    best_radius = np.inf

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
        radius = spectral_radius(a, b, c)
        return radius

    # Initial: try seed's random approach
    for _ in range(budget):
        if time.time() > deadline:
            break
        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        # Extract a, b, c from signs
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
    # All +1 except one -1 on matching (ensures connectivity if product = -1)
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

    # Try: all +1 on cycles, alternating on matching
    for pattern in range(2):
        if time.time() > deadline:
            break
        a = np.ones(m, dtype=np.int8)
        b = np.ones(m, dtype=np.int8)
        c = np.ones(m, dtype=np.int8)
        for i in range(m):
            if (i + pattern) % 2 == 1:
                c[i] = -1
        # Check connectivity
        if is_connected(a, b, c):
            radius = spectral_radius(a, b, c)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs_from_abc(a, b, c)

    if best_signs is None:
        # Emergency: just return a valid pattern
        a = np.ones(m, dtype=np.int8)
        b = np.ones(m, dtype=np.int8)
        c = np.ones(m, dtype=np.int8)
        c[0] = -1
        best_signs = signs_from_abc(a, b, c)
        return best_signs

    # Local search on a, b, c representation
    current_signs = best_signs.copy()
    current_radius = best_radius

    # Extract current a, b, c
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
    # Variables: a[0..m-1], b[0..m-1], c[0..m-1]
    # But flipping changes connectivity, so we need to check
    # Better: flip and if connectivity breaks, flip another to restore
    
    while time.time() < deadline:
        improved = False
        
        # 1-flip moves: flip one sign, check if still connected
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
        var_types = ['a', 'b', 'c']
        for i in range(m):
            if time.time() > deadline:
                break
            for j in range(m):
                if time.time() > deadline:
                    break
                # Flip (var_type_i, i) and (var_type_j, j)
                # To keep it manageable, try all pairs of positions
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
            # Try more aggressive random restarts
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
        # Shouldn't happen, but fallback
        pass

    return best_signs
# EVOLVE-BLOCK-END
