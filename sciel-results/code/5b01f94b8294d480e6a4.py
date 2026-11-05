"""Few edges in 5-uniform hypergraphs without Property B.

Strategy:
1. Start with the known 51-edge construction (even 5-subsets of 10 vertices).
2. Use a fast 2-SAT-style backtracking verifier (forbid monochromatic edges).
3. Local search: greedily delete edges while maintaining non-2-colorability.
4. Try to find smaller constructions via randomized search on 10-14 vertices.
"""
import time
import random
from itertools import combinations

_DEADLINE = 145.0
_START = time.monotonic()


def _elapsed():
    return time.monotonic() - _START


def _not_2_colorable(n, edges):
    """
    Check if the 5-uniform hypergraph is NOT 2-colorable.
    Uses backtracking with unit propagation on the constraint that
    no edge is monochromatic.
    
    For each edge e = {v1,...,v5}, we need NOT(all 0) AND NOT(all 1).
    This is equivalent to: for each edge, at least one vertex is 0 AND at least one is 1.
    
    We model this as a CSP where variables are vertices (0/1) and constraints
    are: for each edge, not all equal.
    """
    m = len(edges)
    if m == 0:
        return False
    
    # Build incidence: for each vertex, list of edge indices
    v_edges = [[] for _ in range(n)]
    for ei, e in enumerate(edges):
        for v in e:
            v_edges[v].append(ei)
    
    # Vertex ordering by degree (most constrained first)
    v_order = sorted(range(n), key=lambda v: -len(v_edges[v]))
    
    color = [-1] * n
    # For each edge, track counts of 0s and 1s assigned
    zero_count = [0] * m
    one_count = [0] * m
    unassigned = [5] * m
    
    # Constraint: for edge ei, we need 1 <= zero_count[ei] <= 4 and 1 <= one_count[ei] <= 4
    # Violation if zero_count[ei] == 5 or one_count[ei] == 5 (all same color)
    # Also violation if zero_count[ei] == 0 and unassigned[ei] == 0 (all 1s)
    # Actually: violation if zero_count[ei] + unassigned[ei] == 0 (impossible, all 1s)
    #           or one_count[ei] + unassigned[ei] == 0 (impossible, all 0s)
    #           or zero_count[ei] == 5 (all 0s)
    #           or one_count[ei] == 5 (all 1s)
    
    def _propagate():
        """Unit propagation: if an edge has 4 of one color and 1 unassigned, force the last."""
        changed = True
        while changed:
            changed = False
            for ei in range(m):
                zc = zero_count[ei]
                oc = one_count[ei]
                uc = unassigned[ei]
                
                # Check for immediate contradiction
                if zc == 5 or oc == 5:
                    return False
                if uc == 0:
                    # All assigned; check if valid
                    if zc == 0 or oc == 0:
                        return False
                    continue
                
                # If 4 are 0 and 1 unassigned, the last must be 1
                if zc == 4 and uc == 1:
                    for v in edges[ei]:
                        if color[v] == -1:
                            color[v] = 1
                            one_count[ei] += 1
                            unassigned[ei] -= 1
                            for e2 in v_edges[v]:
                                if e2 != ei:
                                    one_count[e2] += 1
                                    unassigned[e2] -= 1
                            changed = True
                            break
                # If 4 are 1 and 1 unassigned, the last must be 0
                elif oc == 4 and uc == 1:
                    for v in edges[ei]:
                        if color[v] == -1:
                            color[v] = 0
                            zero_count[ei] += 1
                            unassigned[ei] -= 1
                            for e2 in v_edges[v]:
                                if e2 != ei:
                                    zero_count[e2] += 1
                                    unassigned[e2] -= 1
                            changed = True
                            break
        return True
    
    def _search(idx):
        # Skip already-assigned vertices
        while idx < n and color[v_order[idx]] != -1:
            idx += 1
        if idx == n:
            # All assigned; verify no edge is monochromatic
            for ei in range(m):
                if zero_count[ei] == 0 or one_count[ei] == 0:
                    return False
            return True
        
        v = v_order[idx]
        
        # Save state
        color_snap = color[:]
        zc_snap = zero_count[:]
        oc_snap = one_count[:]
        uc_snap = unassigned[:]
        
        for c in (0, 1):
            color[v] = c
            valid = True
            for ei in v_edges[v]:
                if c == 0:
                    zero_count[ei] += 1
                else:
                    one_count[ei] += 1
                unassigned[ei] -= 1
                # Check for contradiction
                if zero_count[ei] == 5 or one_count[ei] == 5:
                    valid = False
                    break
            if valid:
                if _propagate():
                    if _search(idx + 1):
                        return True
            
            # Restore
            color[:] = color_snap
            zero_count[:] = zc_snap
            one_count[:] = oc_snap
            unassigned[:] = uc_snap
        
        return False
    
    return not _search(0)


def _known_51_edges():
    """The known 51-edge construction: 5-subsets of 10 vertices with even intersection with a fixed 5-set."""
    fixed = frozenset((0, 2, 4, 6, 8))
    edges = []
    for c in combinations(range(10), 5):
        if sum(1 for v in c if v in fixed) % 2 == 0:
            edges.append(list(c))
    return edges


def _greedy_reduce(n, edges, max_time=10.0):
    """
    Greedily try to remove edges while maintaining non-2-colorability.
    Returns the reduced edge list.
    """
    current = list(edges)
    rng = random.Random(123)
    
    while _elapsed() < max_time and len(current) > 0:
        # Try to remove each edge (in random order)
        indices = list(range(len(current)))
        rng.shuffle(indices)
        removed = False
        
        for idx in indices:
            if _elapsed() >= max_time:
                break
            
            # Try removing edge idx
            trial = current[:idx] + current[idx+1:]
            if _not_2_colorable(n, trial):
                current = trial
                removed = True
                break  # Restart after successful removal
        
        if not removed:
            break
    
    return current


def _random_search(n, target_edges, rng, max_time=5.0):
    """
    Randomly generate 5-uniform hypergraphs and check if non-2-colorable.
    Then try to reduce.
    """
    all_edges = list(combinations(range(n), 5))
    best = None
    
    while _elapsed() < max_time:
        # Sample target_edges edges
        if target_edges > len(all_edges):
            target_edges = len(all_edges)
        
        sample = rng.sample(all_edges, target_edges)
        edges = [list(e) for e in sample]
        
        if _not_2_colorable(n, edges):
            # Found a non-2-colorable hypergraph; try to reduce
            reduced = _greedy_reduce(n, edges, max_time=min(3.0, max_time - _elapsed()))
            if best is None or len(reduced) < len(best):
                best = reduced
    
    return best


def _structured_search(n, rng, max_time=5.0):
    """
    Try structured constructions:
    - Even/odd intersections with fixed subsets
    - Random structured patterns
    """
    best = None
    all_edges = list(combinations(range(n), 5))
    
    while _elapsed() < max_time:
        # Try various fixed subsets and parity conditions
        k = rng.randint(2, n - 2)
        fixed = set(rng.sample(range(n), k))
        
        # Try both parities
        for parity in (0, 1):
            edges = []
            for c in all_edges:
                if sum(1 for v in c if v in fixed) % 2 == parity:
                    edges.append(list(c))
            
            if 10 <= len(edges) <= 60:
                if _not_2_colorable(n, edges):
                    reduced = _greedy_reduce(n, edges, max_time=min(2.0, max_time - _elapsed()))
                    if best is None or len(reduced) < len(best):
                        best = reduced
    
    return best


def construct():
    if _elapsed() > _DEADLINE - 10:
        return {'n': 10, 'edges': _known_51_edges()}
    
    rng = random.Random(42)
    
    # Start with known 51-edge construction
    best_n = 10
    best_edges = _known_51_edges()
    
    # First, try to reduce the known construction
    t0 = time.monotonic()
    reduced = _greedy_reduce(10, best_edges, max_time=min(20.0, _DEADLINE - 20 - _elapsed()))
    if len(reduced) < len(best_edges):
        best_edges = reduced
        best_n = 10
    
    # Try random search on various n values
    for n in [10, 11, 12, 13, 14]:
        if _elapsed() > _DEADLINE - 15:
            break
        
        # Try different edge counts
        for target in [30, 35, 40, 45, 50]:
            if _elapsed() > _DEADLINE - 15:
                break
            
            result = _random_search(n, target, rng, max_time=min(5.0, _DEADLINE - 15 - _elapsed()))
            if result is not None and len(result) < len(best_edges):
                best_edges = result
                best_n = n
    
    # Try structured search
    if _elapsed() < _DEADLINE - 15:
        for n in [10, 11, 12]:
            if _elapsed() > _DEADLINE - 15:
                break
            result = _structured_search(n, rng, max_time=min(5.0, _DEADLINE - 15 - _elapsed()))
            if result is not None and len(result) < len(best_edges):
                best_edges = result
                best_n = n
    
    # Final reduction attempt on best found
    if _elapsed() < _DEADLINE - 5:
        final = _greedy_reduce(best_n, best_edges, max_time=min(5.0, _DEADLINE - 5 - _elapsed()))
        if len(final) < len(best_edges):
            best_edges = final
    
    return {'n': best_n, 'edges': best_edges}
