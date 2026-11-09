"""Sign-edges solver for prism-graph spectral lifts: structured init + local search."""
import time
import numpy as np

_DEADLINE = 85.0


def _build_signed_adj(graph, signs):
    n = graph['n']
    edges = graph['edges']
    A = np.zeros((n, n), dtype=np.float64)
    for (u, v), s in zip(edges, signs):
        sv = float(s)
        A[u, v] = sv
        A[v, u] = sv
    return A


def _is_coboundary(graph, signs):
    """Check if sign assignment is a coboundary (lift is disconnected)."""
    n = graph['n']
    edges = graph['edges']
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    
    f = [-1] * n
    f[0] = 0
    stack = [0]
    while stack:
        u = stack.pop()
        for v, idx in adj[u]:
            s = signs[idx]
            expected_fv = f[u] + (0 if s == 1 else 1)
            if f[v] == -1:
                f[v] = expected_fv % 2
                stack.append(v)
            else:
                if f[v] != expected_fv % 2:
                    return False
    return True


def _lift_radius_fast(graph, signs):
    """Return spectral radius if lift is connected and valid, else inf."""
    if _is_coboundary(graph, signs):
        return float('inf')
    A = _build_signed_adj(graph, signs)
    try:
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))
    except Exception:
        return float('inf')


def _eigenvector_importance(graph, signs):
    """Return per-edge importance based on the top eigenvector."""
    A = _build_signed_adj(graph, signs)
    try:
        eigvals, eigvecs = np.linalg.eigh(A)
        idx = np.argmax(np.abs(eigvals))
        vec = np.abs(eigvecs[:, idx])
        edges = graph['edges']
        importance = np.zeros(len(edges))
        for i, (u, v) in enumerate(edges):
            importance[i] = vec[u] * vec[v]
        return importance
    except Exception:
        return np.ones(len(edges))


def _phase_based_init(graph, m):
    """Generate sign patterns using phase synchronization on the prism structure."""
    n_edges = len(graph['edges'])
    inits = []
    
    # Phase model: z_v = exp(i * theta_v)
    # Edge sign = sign(Re(z_u * conj(z_v))) = sign(cos(theta_u - theta_v))
    # For a cycle, this gives alternating signs when phase difference is ~pi
    
    # Try different phase configurations
    for base_phase in np.linspace(0, np.pi, 8):
        signs = np.ones(n_edges, dtype=np.int64)
        # Bottom cycle: phases 0, base_phase, 2*base_phase, ...
        # Top cycle: same phases
        # Matching: phase difference = 0 (same phase)
        for i in range(m):
            theta_u = i * base_phase
            theta_v = (i + 1) % m * base_phase
            # Bottom edge (i, (i+1)%m)
            diff = theta_u - theta_v
            signs[i] = 1 if np.cos(diff) >= 0 else -1
            # Top edge (m+i, m+(i+1)%m)
            signs[m + i] = 1 if np.cos(diff) >= 0 else -1
        # Matching edges: all +1 (same phase)
        inits.append(signs.copy())
    
    # Spiral phases: theta_i = 2*pi*i/m
    signs = np.ones(n_edges, dtype=np.int64)
    for i in range(m):
        diff = 2 * np.pi * i / m
        signs[i] = 1 if np.cos(diff) >= 0 else -1
        signs[m + i] = 1 if np.cos(diff) >= 0 else -1
    inits.append(signs.copy())
    
    # Twisted: top cycle has opposite phase offset
    signs = np.ones(n_edges, dtype=np.int64)
    for i in range(m):
        diff_bottom = 2 * np.pi * i / m
        diff_top = 2 * np.pi * (i + 0.5) / m
        signs[i] = 1 if np.cos(diff_bottom) >= 0 else -1
        signs[m + i] = 1 if np.cos(diff_top) >= 0 else -1
    inits.append(signs.copy())
    
    return inits


def _generate_structured_inits(graph, m):
    """Generate smart initial sign patterns for prism graphs."""
    n_edges = len(graph['edges'])
    inits = []
    
    # Pattern 1: All +1 (baseline, likely disconnected)
    inits.append(np.ones(n_edges, dtype=np.int64))
    
    # Pattern 2: All -1
    inits.append(-np.ones(n_edges, dtype=np.int64))
    
    # Pattern 3: Alternating on bottom cycle, +1 elsewhere
    signs = np.ones(n_edges, dtype=np.int64)
    for i in range(m):
        signs[i] = 1 if i % 2 == 0 else -1
    inits.append(signs.copy())
    
    # Pattern 4: Alternating on top cycle, +1 elsewhere
    signs = np.ones(n_edges, dtype=np.int64)
    for i in range(m):
        signs[m + i] = 1 if i % 2 == 0 else -1
    inits.append(signs.copy())
    
    # Pattern 5: Alternating on matching
    signs = np.ones(n_edges, dtype=np.int64)
    for i in range(m):
        signs[2*m + i] = 1 if i % 2 == 0 else -1
    inits.append(signs.copy())
    
    # Pattern 6: Random with seed
    rng = np.random.default_rng(42)
    for _ in range(10):
        signs = rng.choice([-1, 1], size=n_edges)
        inits.append(signs)
    
    # Pattern 7: -1 on one edge of bottom cycle (to break coboundary)
    signs = np.ones(n_edges, dtype=np.int64)
    signs[0] = -1
    inits.append(signs.copy())
    
    # Pattern 8: -1 on one edge of top cycle
    signs = np.ones(n_edges, dtype=np.int64)
    signs[m] = -1
    inits.append(signs.copy())
    
    # Pattern 9: -1 on one matching edge
    signs = np.ones(n_edges, dtype=np.int64)
    signs[2*m] = -1
    inits.append(signs.copy())
    
    # Pattern 10: Checkerboard-like
    signs = np.ones(n_edges, dtype=np.int64)
    signs[0] = -1
    signs[m + 0] = -1
    inits.append(signs.copy())
    
    # Add phase-based inits
    inits.extend(_phase_based_init(graph, m))
    
    # For odd m: single -1 on bottom cycle at different positions
    if m % 2 == 1:
        for pos in range(m):
            signs = np.ones(n_edges, dtype=np.int64)
            signs[pos] = -1
            inits.append(signs.copy())
        # Single -1 on top cycle
        for pos in range(m):
            signs = np.ones(n_edges, dtype=np.int64)
            signs[m + pos] = -1
            inits.append(signs.copy())
    
    return inits


def _local_search(graph, signs, radius, deadline):
    """Local search with single and double flips, ordered by eigenvector importance."""
    n_edges = len(graph['edges'])
    signs = signs.copy()
    best_radius = radius
    best_signs = signs.copy()
    
    improved = True
    stall_count = 0
    max_stall = 3
    
    while improved and time.time() < deadline:
        improved = False
        
        # Compute importance ordering
        if time.time() < deadline:
            importance = _eigenvector_importance(graph, best_signs)
            order = np.argsort(-importance)
        else:
            order = np.arange(n_edges)
        
        # Single flips in importance order
        for idx in order:
            i = int(idx)
            if time.time() >= deadline:
                break
            signs = best_signs.copy()
            signs[i] = -signs[i]
            r = _lift_radius_fast(graph, signs)
            if r < best_radius - 1e-12:
                best_radius = r
                best_signs = signs.copy()
                improved = True
        
        if improved:
            stall_count = 0
            continue
        stall_count += 1
        
        if stall_count > max_stall:
            break
        
        # Double flips in importance order
        if time.time() >= deadline:
            break
        for idx_i in order:
            if time.time() >= deadline:
                break
            i = int(idx_i)
            for idx_j in order:
                j = int(idx_j)
                if j <= i:
                    continue
                if time.time() >= deadline:
                    break
                signs = best_signs.copy()
                signs[i] = -signs[i]
                signs[j] = -signs[j]
                r = _lift_radius_fast(graph, signs)
                if r < best_radius - 1e-12:
                    best_radius = r
                    best_signs = signs.copy()
                    improved = True
                    break
            if improved:
                break
        
        # Triple flips if still stuck
        if not improved and stall_count >= 2 and time.time() < deadline:
            for idx_i in order[:min(5, n_edges)]:
                if time.time() >= deadline:
                    break
                i = int(idx_i)
                for idx_j in order[:min(8, n_edges)]:
                    if time.time() >= deadline:
                        break
                    j = int(idx_j)
                    if j <= i:
                        continue
                    for idx_k in order[:min(10, n_edges)]:
                        if time.time() >= deadline:
                            break
                        k = int(idx_k)
                        if k <= j:
                            continue
                        signs = best_signs.copy()
                        signs[i] = -signs[i]
                        signs[j] = -signs[j]
                        signs[k] = -signs[k]
                        r = _lift_radius_fast(graph, signs)
                        if r < best_radius - 1e-12:
                            best_radius = r
                            best_signs = signs.copy()
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            stall_count = 0
    
    return best_signs, best_radius


def _polish(graph, signs, radius, deadline):
    """Final high-precision polish: try all single flips with tight tolerance."""
    n_edges = len(graph['edges'])
    best_signs = signs.copy()
    best_radius = radius
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_edges):
            if time.time() >= deadline:
                break
            candidate = best_signs.copy()
            candidate[i] = -candidate[i]
            r = _lift_radius_fast(graph, candidate)
            if r < best_radius - 1e-14:
                best_radius = r
                best_signs = candidate
                improved = True
    
    return best_signs, best_radius


def sign_edges(graph, budget):
    start = time.time()
    deadline = start + _DEADLINE
    
    n = graph['n']
    m = n // 2
    n_edges = len(graph['edges'])
    
    if n_edges == 0:
        return np.array([], dtype=np.int64)
    
    # Generate initial candidates
    inits = _generate_structured_inits(graph, m)
    
    # Evaluate all inits, keep best
    best_signs = None
    best_radius = float('inf')
    for init in inits:
        if time.time() >= deadline:
            break
        r = _lift_radius_fast(graph, init)
        if r < best_radius:
            best_radius = r
            best_signs = init.copy()
    
    if best_signs is None:
        # Fallback: try random
        rng = np.random.default_rng(3)
        for _ in range(min(budget, 50)):
            if time.time() >= deadline:
                break
            signs = rng.choice([-1, 1], size=n_edges)
            r = _lift_radius_fast(graph, signs)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
    
    if best_signs is None:
        return np.ones(n_edges, dtype=np.int64)
    
    # Local search
    if time.time() < deadline:
        best_signs, best_radius = _local_search(graph, best_signs, best_radius, deadline)
    
    # Final polish
    if time.time() < deadline - 5:
        best_signs, best_radius = _polish(graph, best_signs, best_radius, deadline)
    
    return np.asarray(best_signs, dtype=np.int64)
