"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time as _time
import numpy as np
from collections import deque

# Module-level deadline tracking
_T0 = _time.time()
_GLOBAL_DEADLINE = _T0 + 80.0  # safety margin before 90s kill

# EVOLVE-BLOCK-START
def sign_edges(graph, budget):
    global _GLOBAL_DEADLINE
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n = 2 * m
    
    # Per-call deadline: min(20s, remaining global)
    remaining = _GLOBAL_DEADLINE - _time.time()
    call_deadline = _time.time() + min(20.0, max(1.0, remaining))
    
    # Precompute adjacency (with edge index) for validity BFS
    adj = [[] for _ in range(n)]
    for ei, (u, v) in enumerate(edges):
        adj[u].append((v, ei))
        adj[v].append((u, ei))
    
    def build_A(signs):
        A = np.zeros((n, n), dtype=float)
        for ei, (u, v) in enumerate(edges):
            s = float(signs[ei])
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(A):
        ev = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(ev)))
    
    def is_valid(signs):
        # Connected lift iff NOT a coboundary iff BFS propagation finds a conflict
        phi = [0] * n
        phi[0] = 1
        q = deque([0])
        while q:
            u = q.popleft()
            for v, ei in adj[u]:
                expected = phi[u] * signs[ei]
                if phi[v] == 0:
                    phi[v] = expected
                    q.append(v)
                elif phi[v] != expected:
                    return True
        return False
    
    def make_valid(signs):
        # If invalid (coboundary), flip edges until valid
        if is_valid(signs):
            return signs
        # It's a coboundary. Flipping any edge breaks the coboundary property? 
        # Not necessarily a single flip. But flipping one edge in a cycle makes product -1.
        # Simple: flip a random edge, recheck; repeat.
        rng_local = np.random.default_rng(hash(tuple(signs)) & 0xFFFFFFFF)
        for _ in range(n_edges * 2):
            i = int(rng_local.integers(n_edges))
            trial = signs.copy()
            trial[i] *= -1
            if is_valid(trial):
                return trial
        # Fallback: flip all matching edges
        trial = signs.copy()
        for ei, (u, v) in enumerate(edges):
            if abs(u - v) == m:
                trial[ei] *= -1
        if is_valid(trial):
            return trial
        return None
    
    best_signs = None
    best_radius = float('inf')
    
    rng = np.random.default_rng(3)
    
    # Structured initializations
    inits = []
    # 1. all +1 then make valid
    s = np.ones(n_edges, dtype=int)
    inits.append(s)
    # 2. matching = -1
    s = np.ones(n_edges, dtype=int)
    for ei, (u, v) in enumerate(edges):
        if abs(u - v) == m:
            s[ei] = -1
    inits.append(s)
    # 3. top cycle = -1
    s = np.ones(n_edges, dtype=int)
    for ei, (u, v) in enumerate(edges):
        if u < m and v < m:
            s[ei] = -1
    inits.append(s)
    # 4. checkerboard on matching
    s = np.ones(n_edges, dtype=int)
    for ei, (u, v) in enumerate(edges):
        if abs(u - v) == m:
            if (u // m) % 2 == 0:
                s[ei] = -1
    inits.append(s)
    # Random inits
    for _ in range(30):
        inits.append(rng.choice([-1, 1], size=n_edges).astype(int))
    
    def hill_climb(signs):
        nonlocal best_signs, best_radius
        A = build_A(signs)
        r = spectral_radius(A)
        improved = True
        while improved and _time.time() < call_deadline:
            improved = False
            best_trial = None
            best_trial_r = r
            # 1-flip moves
            for i in range(n_edges):
                trial = signs.copy()
                trial[i] *= -1
                if not is_valid(trial):
                    continue
                At = A.copy()
                u, v = edges[i]
                At[u, v] = -At[u, v]
                At[v, u] = -At[v, u]
                tr = spectral_radius(At)
                if tr < best_trial_r - 1e-12:
                    best_trial_r = tr
                    best_trial = trial
                    best_trial_A = At
            if best_trial is not None:
                signs = best_trial
                A = best_trial_A
                r = best_trial_r
                improved = True
                if r < best_radius:
                    best_radius = r
                    best_signs = signs.copy()
            else:
                # Try 2-flip moves to escape
                for i in range(n_edges):
                    if _time.time() > call_deadline:
                        break
                    for j in range(i + 1, n_edges):
                        trial = signs.copy()
                        trial[i] *= -1
                        trial[j] *= -1
                        if not is_valid(trial):
                            continue
                        At = A.copy()
                        u1, v1 = edges[i]
                        At[u1, v1] *= -1
                        At[v1, u1] *= -1
                        u2, v2 = edges[j]
                        At[u2, v2] *= -1
                        At[v2, u2] *= -1
                        tr = spectral_radius(At)
                        if tr < best_trial_r - 1e-12:
                            best_trial_r = tr
                            best_trial = trial
                            best_trial_A = At
                    if best_trial is not None:
                        break
                if best_trial is not None:
                    signs = best_trial
                    A = best_trial_A
                    r = best_trial_r
                    improved = True
                    if r < best_radius:
                        best_radius = r
                        best_signs = signs.copy()
        return signs, r
    
    for s in inits:
        if _time.time() > call_deadline:
            break
        sv = make_valid(s)
        if sv is None:
            continue
        hill_climb(sv)
    
    if best_signs is None:
        # Fallback: return a valid signing
        sv = make_valid(np.ones(n_edges, dtype=int))
        if sv is None:
            sv = np.ones(n_edges, dtype=int)
            # ensure valid
            for ei, (u, v) in enumerate(edges):
                if abs(u - v) == m:
                    sv[ei] = -1
        best_signs = sv
    
    return best_signs.astype(int)
# EVOLVE-BLOCK-END
