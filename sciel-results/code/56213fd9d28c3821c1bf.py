"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _build_prism(m):
    """Build prism graph with m: two m-cycles + matching. Returns (n, edges)."""
    n = 2 * m
    edges = []
    # First cycle: 0..m-1
    for i in range(m):
        edges.append((i, (i + 1) % m))
    # Second cycle: m..2m-1
    for i in range(m):
        u = m + i
        v = m + (i + 1) % m
        edges.append((u, v))
    # Matching: i <-> i+m
    for i in range(m):
        edges.append((i, i + m))
    return n, edges

def _signed_adj(n, edges, signs):
    """Build signed adjacency matrix."""
    A = np.zeros((n, n), dtype=float)
    for (u, v), s in zip(edges, signs):
        A[u, v] = s
        A[v, u] = s
    return A

def _spectral_radius(n, edges, signs):
    """Compute spectral radius of signed adjacency matrix."""
    A = _signed_adj(n, edges, signs)
    vals = np.linalg.eigvalsh(A)
    return np.max(np.abs(vals))

def _is_valid_lift(n, edges, signs):
    """
    Check if the 2-lift is connected and 3-regular.
    For a prism graph (3-regular), the lift is automatically 3-regular if valid.
    The lift is connected iff the product of signs around every cycle is consistent
    with a single connected component.
    
    Actually, for a 2-lift: vertices are (v, 0) and (v, 1).
    Edge (u,v) with sign +1 connects (u,0)-(v,0) and (u,1)-(v,1).
    Edge (u,v) with sign -1 connects (u,0)-(v,1) and (u,1)-(v,0).
    
    The lift is connected iff there's no non-trivial bipartition of the base vertices
    such that all matching edges have sign +1 and all cycle edges within each part have sign...
    
    Actually, the 2-lift is connected iff the signed graph is "connected" in the sense that
    the only harmonic 1-form is zero, or equivalently, the product of signs around any
    closed walk determines a unique parity.
    
    More concretely: the lift has 2 components iff there exists a function f: V -> {0,1}
    such that for every edge (u,v) with sign s: f(u) XOR f(v) = 0 if s=+1, f(u) XOR f(v) = 1 if s=-1.
    This is a system of XOR equations. The lift is connected iff this system has no solution
    (other than the trivial one which would give two copies).
    
    Wait, let me think again. If such an f exists, then (v, 0) in the lift connects to (w, f(u) XOR f(v))... 
    Actually, the lift is connected iff the system of constraints is inconsistent.
    """
    # Build the constraint system: for each edge (u,v) with sign s,
    # if s = +1: f(u) = f(v)
    # if s = -1: f(u) != f(v)
    # The lift is connected iff this system is inconsistent (no valid 2-coloring exists).
    
    # Use BFS/union-find to check consistency
    parent = list(range(n))
    rank = [0] * n
    # value[i] = value of f(i) relative to root of its component
    # We'll use a parity DSU
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y, parity):
        """Union x and y with constraint f(x) XOR f(y) = parity."""
        rx, ry = find(x), find(y)
        if rx == ry:
            # Check consistency
            # We need to track the parity from each node to its root
            pass
        else:
            parent[ry] = rx
    
    # Simpler approach: BFS to assign values and check for conflicts
    # value[v] = 0 or 1, with constraint: for edge (u,v) sign s,
    # if s=+1: value[u] == value[v]
    # if s=-1: value[u] != value[v]
    
    value = [-1] * n  # -1 = unvisited
    value[0] = 0
    
    from collections import deque
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        s = signs[idx]
        # constraint: value[u] XOR value[v] = 0 if s=+1, 1 if s=-1
        xor_val = 0 if s == 1 else 1
        adj[u].append((v, xor_val))
        adj[v].append((u, xor_val))
    
    queue = deque([0])
    while queue:
        u = queue.popleft()
        for v, xor_val in adj[u]:
            expected = value[u] ^ xor_val
            if value[v] == -1:
                value[v] = expected
                queue.append(v)
            elif value[v] != expected:
                # Inconsistent: no valid 2-coloring, so lift is connected
                return True
    
    # Consistent: there's a valid 2-coloring, so lift has 2 components
    return False

def _evaluate(n, edges, signs):
    """Return spectral radius if valid, else None."""
    if not _is_valid_lift(n, edges, signs):
        return None
    return _spectral_radius(n, edges, signs)

def sign_edges(graph, budget):
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    
    # Determine m
    m = n // 2
    
    deadline = time.time() + 55  # Safety margin before 60s limit
    
    # Start with random initialization
    rng = np.random.default_rng(3)
    signs = rng.choice([-1, 1], size=num_edges)
    
    # Evaluate initial
    best_radius = _evaluate(n, edges, signs)
    best_signs = signs.copy()
    if best_radius is None:
        # Try to fix: flip random edges until valid
        for attempt in range(1000):
            signs = rng.choice([-1, 1], size=num_edges)
            r = _evaluate(n, edges, signs)
            if r is not None:
                best_radius = r
                best_signs = signs.copy()
                break
        if best_radius is None:
            # Fallback: all +1 except one -1 on a matching edge
            signs = np.ones(num_edges, dtype=int)
            # Find a matching edge
            for i, (u, v) in enumerate(edges):
                if v == u + m or u == v + m:
                    signs[i] = -1
                    break
            best_radius = _evaluate(n, edges, signs)
            best_signs = signs.copy()
            if best_radius is None:
                # This shouldn't happen for m >= 4
                return signs
    
    # Local search: single edge flips
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(num_edges):
            if time.time() > deadline:
                break
            signs[i] = -signs[i]
            r = _evaluate(n, edges, signs)
            if r is not None and r < best_radius:
                best_radius = r
                best_signs = signs.copy()
                improved = True
            else:
                signs[i] = -signs[i]  # revert
    
    # Local search: pair flips
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(num_edges):
                if time.time() > deadline:
                    break
                for j in range(i + 1, num_edges):
                    if time.time() > deadline:
                        break
                    signs[i] = -signs[i]
                    signs[j] = -signs[j]
                    r = _evaluate(n, edges, signs)
                    if r is not None and r < best_radius:
                        best_radius = r
                        best_signs = signs.copy()
                        improved = True
                    else:
                        signs[i] = -signs[i]
                        signs[j] = -signs[j]  # revert
    
    # Simulated annealing / random restarts for remaining time
    if time.time() < deadline:
        current_signs = best_signs.copy()
        current_radius = best_radius
        T = 0.5
        cooling = 0.999
        while time.time() < deadline:
            i = rng.integers(num_edges)
            current_signs[i] = -current_signs[i]
            r = _evaluate(n, edges, current_signs)
            if r is not None:
                if r < current_radius:
                    current_radius = r
                    if r < best_radius:
                        best_radius = r
                        best_signs = current_signs.copy()
                else:
                    # Accept with probability exp(-(r - current_radius) / T)
                    if np.log(rng.random()) < -(r - current_radius) / T:
                        current_radius = r
                    else:
                        current_signs[i] = -current_signs[i]  # revert
            else:
                current_signs[i] = -current_signs[i]  # revert invalid
            T *= cooling
            if T < 0.01:
                T = 0.01
    
    return best_signs
# EVOLVE-BLOCK-END
