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
    """Check if sign assignment is a coboundary (lift is disconnected).
    A sign assignment s is a coboundary iff there exists f: V->{0,1} such that
    s(u,v) = (-1)^(f(u)+f(v)) for all edges (u,v).
    This is equivalent to: the product of signs around every cycle is +1.
    We can check by BFS: assign f[0]=0, then for each edge (u,v), require
    f[v] = f[u] + (0 if s=+1 else 1) mod 2. If consistent, it's a coboundary.
    """
    n = graph['n']
    edges = graph['edges']
    # Build adjacency
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
            # s = (-1)^(f[u]+f[v]) => f[v] = f[u] + (0 if s==1 else 1) mod 2
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
    # First m edges are bottom cycle (0,1),(1,2),...,(m-1,0)
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
    
    return inits


def _local_search(graph, signs, radius, deadline):
    """Local search with single and double flips."""
    n_edges = len(graph['edges'])
    signs = signs.copy()
    best_radius = radius
    best_signs = signs.copy()
    
    improved = True
    stall_count = 0
    while improved and time.time() < deadline:
        improved = False
        # Single flips
        for i in range(n_edges):
            if time.time() >= deadline:
                break
            signs[i] = -signs[i]
            r = _lift_radius_fast(graph, signs)
            if r < best_radius - 1e-12:
                best_radius = r
                best_signs = signs.copy()
                improved = True
            else:
                signs[i] = -signs[i]
        if improved:
            stall_count = 0
            continue
        stall_count += 1
        
        # Double flips
        if time.time() >= deadline:
            break
        for i in range(n_edges):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n_edges):
                if time.time() >= deadline:
                    break
                signs[i] = -signs[i]
                signs[j] = -signs[j]
                r = _lift_radius_fast(graph, signs)
                if r < best_radius - 1e-12:
                    best_radius = r
                    best_signs = signs.copy()
                    improved = True
                    signs[i] = -signs[i]  # revert for consistency
                    signs[j] = -signs[j]
                    break
                else:
                    signs[i] = -signs[i]
                    signs[j] = -signs[j]
            if improved:
                break
        
        # Triple flips if stuck
        if not improved and stall_count >= 2 and time.time() < deadline:
            for i in range(n_edges):
                if time.time() >= deadline:
                    break
                for j in range(i + 1, n_edges):
                    if time.time() >= deadline:
                        break
                    for k in range(j + 1, min(j + 5, n_edges)):
                        if time.time() >= deadline:
                            break
                        signs[i] = -signs[i]
                        signs[j] = -signs[j]
                        signs[k] = -signs[k]
                        r = _lift_radius_fast(graph, signs)
                        if r < best_radius - 1e-12:
                            best_radius = r
                            best_signs = signs.copy()
                            improved = True
                            signs[i] = -signs[i]
                            signs[j] = -signs[j]
                            signs[k] = -signs[k]
                            break
                        else:
                            signs[i] = -signs[i]
                            signs[j] = -signs[j]
                            signs[k] = -signs[k]
                    if improved:
                        break
                if improved:
                    break
            stall_count = 0
    
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
    
    return np.asarray(best_signs, dtype=np.int64)
