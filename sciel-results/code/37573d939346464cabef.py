"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = time.time() + 145.0

def _build_circulant_adj(n, S):
    """Build adjacency list for circulant graph on n vertices with connection set S."""
    adj = [set() for _ in range(n)]
    for i in range(n):
        for s in S:
            j = (i + s) % n
            if j != i:
                adj[i].add(j)
                adj[j].add(i)
    return adj

def _has_k4(n, adj):
    """Check if graph has a K4."""
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            common = adj[i] & adj[j]
            for k in common:
                if k <= j:
                    continue
                common2 = common & adj[k]
                for l in common2:
                    if l > k:
                        return True
    return False

def _has_indep6(n, adj):
    """Check if graph has an independent set of size 6."""
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                comp_adj[i].add(j)
                comp_adj[j].add(i)
    
    def _find_clique(candidates, size):
        if size == 0:
            return True
        if not candidates or len(candidates) < size:
            return False
        for v in list(candidates):
            if size == 1:
                return True
            new_cands = candidates & comp_adj[v]
            if len(new_cands) >= size - 1:
                if _find_clique(new_cands, size - 1):
                    return True
            candidates.discard(v)
        return False
    
    return _find_clique(set(range(n)), 6)

def _is_valid(n, adj):
    return not _has_k4(n, adj) and not _has_indep6(n, adj)

def _circulant_to_dict(n, S):
    adj = _build_circulant_adj(n, S)
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
    return {'n': n, 'edges': edges}

def _search_circulant(n, max_attempts=50000):
    """Search for valid circulant graph on n vertices."""
    if n <= 1:
        return set()
    
    # For circulant graphs, connection set S ⊆ {1, 2, ..., n//2}
    # (we can assume symmetry: if s in S then n-s in S)
    max_conn = n // 2
    
    # Try random subsets of possible connection values
    all_conns = list(range(1, max_conn + 1))
    
    for attempt in range(max_attempts):
        if time.time() > _DEADLINE - 10:
            break
        
        # Random subset with bias toward medium density
        # Expected degree ~ 2*|S|, want degree ~ n/2 to n/3
        target_size = random.randint(max(1, n // 6), min(max_conn, n // 3))
        if target_size > len(all_conns):
            target_size = len(all_conns)
        
        S = set(random.sample(all_conns, target_size))
        
        adj = _build_circulant_adj(n, S)
        
        if not _has_k4(n, adj) and not _has_indep6(n, adj):
            return S
    
    return None

def _grow_circulant(n, S, max_attempts=20000):
    """Try to grow a circulant graph from n to n+1 vertices."""
    # This is tricky for circulant graphs - changing n changes the structure
    # Instead, we'll search for circulant graphs of increasing size
    
    return None

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 145.0
    
    random.seed(42)
    
    best_n = 0
    best_S = None
    best_result = None
    
    # Search for circulant graphs of increasing size
    # Start from smaller n and work up
    for n in range(10, 36):
        if time.time() > _DEADLINE - 20:
            break
        
        # Allocate more attempts for larger n (harder to find)
        if n <= 15:
            attempts = 100000
        elif n <= 20:
            attempts = 200000
        elif n <= 25:
            attempts = 300000
        else:
            attempts = 500000
        
        S = _search_circulant(n, attempts)
        
        if S is not None:
            result = _circulant_to_dict(n, S)
            # Verify
            adj = _build_circulant_adj(n, S)
            if _is_valid(n, adj):
                if n > best_n:
                    best_n = n
                    best_S = S
                    best_result = result
    
    # If we didn't find anything, return a small valid graph
    if best_result is None:
        # Return Paley 17 as fallback
        squares = {x * x % 17 for x in range(1, 17)}
        n = 17
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % 17 in squares:
                    adj[i].add(j)
                    adj[j].add(i)
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append([i, j])
        best_result = {'n': n, 'edges': edges}
    
    return best_result
# EVOLVE-BLOCK-END
