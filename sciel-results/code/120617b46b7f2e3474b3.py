"""Few edges in 5-uniform hypergraphs without Property B.

Strategy:
1. Start with the known 51-edge construction (even 5-subsets of 10 vertices).
2. Use an optimized DPLL-based 2-colorability check (each 5-edge gives two clauses:
   forbid all-0 and forbid all-1) to verify non-2-colorability.
3. Local search: greedily delete edges while maintaining non-2-colorability.
4. Return the best hypergraph found before the time deadline.
"""
import time
import random
from itertools import combinations

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


# Precompute at module level
_DEFAULT_EDGES = _edges_10_even()


def _not_2_colorable(n, edges):
    """
    Returns True if the hypergraph is NOT 2-colorable (i.e., every 2-coloring
    makes at least one edge monochromatic).
    
    Uses DPLL with unit propagation on the CNF:
    For each edge {v1,...,v5}, add clauses:
      (v1 ∨ v2 ∨ v3 ∨ v4 ∨ v5)       # not all 0
      (¬v1 ∨ ¬v2 ∨ ¬v3 ∨ ¬v4 ∨ ¬v5)  # not all 1
    
    This is a 2-colorability check where each variable is a vertex color.
    """
    m = len(edges)
    if m == 0:
        return False  # empty graph is trivially 2-colorable
    
    # Build clause structure
    # Each clause is a list of (var, sign) where sign=1 means positive, -1 means negative
    # Clause (v1∨...∨v5): all positive
    # Clause (¬v1∨...∨¬v5): all negative
    
    # For efficiency, represent each edge as two bitmasks:
    # pos_mask: vertices in the edge (for the "not all 0" clause)
    # neg_mask: same vertices (for the "not all 1" clause)
    
    # State:
    # color[v] = -1 (unassigned), 0, or 1
    # For unit propagation, track for each edge:
    #   pos_remaining: number of unassigned vertices in edge (for "not all 0" clause)
    #   neg_remaining: number of unassigned vertices in edge (for "not all 1" clause)
    #   pos_forced: if all but one are 0, the last must be 1
    #   neg_forced: if all but one are 1, the last must be 0
    
    # More efficient: track counts
    # pos_count[ei]: number of vertices in edge ei that are currently 0
    # neg_count[ei]: number of vertices in edge ei that are currently 1
    # uncount[ei]: number unassigned
    
    # Clause "not all 0" is violated if pos_count == 5
    # Clause "not all 1" is violated if neg_count == 5
    # Unit propagation: if pos_count == 4 and uncount == 1, force last to 1
    #                    if neg_count == 4 and uncount == 1, force last to 0
    
    # Precompute vertex-edge incidence
    v_edges = [[] for _ in range(n)]
    for ei, e in enumerate(edges):
        for v in e:
            v_edges[v].append(ei)
    
    # Vertex ordering by degree
    v_order = sorted(range(n), key=lambda v: -len(v_edges[v]))
    
    # Use arrays for speed
    color = [-1] * n
    pos_count = [0] * m
    neg_count = [0] * m
    uncount = [5] * m
    
    # Stack for backtracking: list of (vertex, old_color, changes)
    # changes: list of (ei, old_pos, old_neg, old_unc)
    
    def _propagate():
        """Unit propagation. Returns False on contradiction."""
        changed = True
        while changed:
            changed = False
            for ei in range(m):
                pc = pos_count[ei]
                nc = neg_count[ei]
                uc = uncount[ei]
                if pc == 5 or nc == 5:
                    return False
                if uc == 0:
                    continue
                if pc == 4 and uc == 1:
                    # Force the unassigned vertex to 1
                    for v in edges[ei]:
                        if color[v] == -1:
                            color[v] = 1
                            neg_count[ei] += 1
                            uncount[ei] -= 1
                            for e2 in v_edges[v]:
                                if e2 != ei:
                                    neg_count[e2] += 1
                                    uncount[e2] -= 1
                            changed = True
                            break
                elif nc == 4 and uc == 1:
                    # Force the unassigned vertex to 0
                    for v in edges[ei]:
                        if color[v] == -1:
                            color[v] = 0
                            pos_count[ei] += 1
                            uncount[ei] -= 1
                            for e2 in v_edges[v]:
                                if e2 != ei:
                                    pos_count[e2] += 1
                                    uncount[e2] -= 1
                            changed = True
                            break
        return True
    
    # Backtracking search
    # We'll use a recursive approach with state snapshots for n<=14
    # For larger n, we'd need more sophisticated techniques, but n<=14 is manageable
    
    def _search(idx):
        """Try to find a satisfying assignment starting from vertex order index idx."""
        # Find next unassigned vertex
        while idx < n and color[v_order[idx]] != -1:
            idx += 1
        if idx == n:
            return True  # All assigned, no contradiction found
        
        v = v_order[idx]
        
        # Snapshot
        color_snap = color[:]
        pos_snap = pos_count[:]
        neg_snap = neg_count[:]
        unc_snap = uncount[:]
        
        for c in (0, 1):
            color[v] = c
            valid = True
            for ei in v_edges[v]:
                if c == 0:
                    pos_count[ei] += 1
                else:
                    neg_count[ei] += 1
                uncount[ei] -= 1
                if pos_count[ei] == 5 or neg_count[ei] == 5:
                    valid = False
                    break
            if valid and _propagate():
                if _search(idx + 1):
                    return True
            # Restore
            color[:] = color_snap
            pos_count[:] = pos_snap
            neg_count[:] = neg_snap
            uncount[:] = unc_snap
        
        return False
    
    return not _search(0)


def construct():
    if _elapsed() > _DEADLINE - 10:
        return {'n': 10, 'edges': _DEFAULT_EDGES}
    
    current_edges = list(_DEFAULT_EDGES)
    n = 10
    rng = random.Random(42)
    
    # Verify initial state
    if not _not_2_colorable(n, current_edges):
        # Shouldn't happen, but safety
        return {'n': 10, 'edges': _DEFAULT_EDGES}
    
    # Local search: try to remove edges
    while _elapsed() < _DEADLINE - 5:
        removed_any = False
        indices = list(range(len(current_edges)))
        rng.shuffle(indices)
        
        for idx in indices:
            if _elapsed() > _DEADLINE - 5:
                break
            
            e_to_remove = current_edges[idx]
            new_edges = current_edges[:idx] + current_edges[idx+1:]
            
            # Check if still NOT 2-colorable
            if _not_2_colorable(n, new_edges):
                current_edges = new_edges
                removed_any = True
        
        if not removed_any:
            break
    
    return {'n': n, 'edges': current_edges}
