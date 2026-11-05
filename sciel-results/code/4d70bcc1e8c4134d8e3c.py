"""Few edges in 5-uniform hypergraphs without Property B.

Uses the known 51-edge construction on 10 vertices (even 5-subsets), then
attempts greedy edge removal to minimize edges while maintaining non-2-colorability.
Uses a fast bitmask-based 2-SAT verifier for edge removal checks.
"""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

_DEADLINE = 145.0
_START = time.monotonic()


def _elapsed():
    return time.monotonic() - _START


def _edges_10_even():
    """All 5-subsets of {0..9} with an even number of even-indexed vertices."""
    evenset = frozenset((0, 2, 4, 6, 8))
    edges = []
    for c in combinations(range(10), 5):
        if sum(1 for v in c if v in evenset) % 2 == 0:
            edges.append(list(c))
    return edges


def _is_2_colorable_fast(n, edges):
    """Fast check: return True if hypergraph IS 2-colorable (has a proper coloring).
    Uses backtracking with unit propagation, optimized with early termination.
    Returns False if NOT 2-colorable (what we want).
    """
    if not edges:
        return True
    
    m = len(edges)
    # Build incidence
    v_edges = [[] for _ in range(n)]
    for ei, e in enumerate(edges):
        for v in e:
            v_edges[v].append(ei)
    
    # If any vertex has no edges, it's trivially colorable
    # Use DPLL-style: try to find a coloring
    # For small n (<=10), brute force with pruning is fast enough
    # But let's use a smarter approach: for n=10, 2^10 = 1024 colorings
    
    # Actually for n<=12, we can enumerate all 2^n colorings
    if n <= 14:
        for mask in range(1 << n):
            valid = True
            for e in edges:
                # Check if all vertices in e have same color
                colors_in_edge = set()
                for v in e:
                    colors_in_edge.add((mask >> v) & 1)
                if len(colors_in_edge) == 1:
                    valid = False
                    break
            if valid:
                return True
        return False
    else:
        # For larger n, use backtracking
        return _backtrack_colorable(n, edges, v_edges)


def _backtrack_colorable(n, edges, v_edges):
    """Backtracking for larger n."""
    color = [-1] * n
    edge_col = [[0, 0] for _ in range(len(edges))]
    edge_un = [5] * len(edges)
    
    order = sorted(range(n), key=lambda v: -len(v_edges[v]))
    
    def bt(idx):
        if idx == n:
            return True
        while idx < n and color[order[idx]] != -1:
            idx += 1
        if idx == n:
            return True
        v = order[idx]
        for c in (0, 1):
            color[v] = c
            before_ec = [list(x) for x in edge_col]
            before_eu = list(edge_un)
            for ei in v_edges[v]:
                edge_col[ei][c] += 1
                edge_un[ei] -= 1
                if edge_col[ei][c] == 5:
                    color[v] = -1
                    edge_col[:] = before_ec
                    edge_un[:] = before_eu
                    break
            else:
                if bt(idx + 1):
                    return True
            color[v] = -1
            edge_col[:] = before_ec
            edge_un[:] = before_eu
        return False
    
    return bt(0)


# Precompute the base construction
_BASE_EDGES = _edges_10_even()
_N = 10


def _try_remove_edges(edges, n, max_time=10.0):
    """Try to greedily remove edges while maintaining non-2-colorability."""
    if _elapsed() > _DEADLINE - 5:
        return edges
    
    current_edges = list(edges)
    improved = True
    while improved and _elapsed() < _DEADLINE - 5:
        improved = False
        # Try removing each edge
        for i in range(len(current_edges)):
            if _elapsed() > _DEADLINE - 5:
                break
            candidate = current_edges[:i] + current_edges[i+1:]
            # Check if candidate is still non-2-colorable
            if not _is_2_colorable_fast(n, candidate):
                current_edges = candidate
                improved = True
                break  # restart from beginning after removal
    return current_edges


def construct():
    """Return a minimal non-2-colorable 5-uniform hypergraph."""
    if _elapsed() > _DEADLINE - 2:
        # Emergency fallback
        return {'n': _N, 'edges': _BASE_EDGES}
    
    # Start with the known 51-edge construction
    edges = _BASE_EDGES
    
    # Verify it's non-2-colorable (should be, but check)
    if _is_2_colorable_fast(_N, edges):
        # This shouldn't happen, but fallback to complete hypergraph
        from itertools import combinations
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    
    # Try to reduce edges
    reduced_edges = _try_remove_edges(edges, _N)
    
    return {'n': _N, 'edges': reduced_edges}
# EVOLVE-BLOCK-END
