"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

DEADLINE = 150.0
SAFETY_MARGIN = 5.0


def _generate_norm4_candidates():
    """Generate all nonzero integer vectors with squared norm 4 in R^11.
    These are: weight-2 vectors (±2 on one coord) and weight-4 vectors (±1 on four coords).
    Also includes vectors like (±1, ±1, ±2, 0...) which have norm^2 = 1+1+4=6, no.
    Actually norm^2=4 means:
    - One coordinate is ±2: C(11,1)*2 = 22 vectors
    - Two coordinates are ±1,±1 (same or diff sign): C(11,2)*4 = 55*4 = 220 vectors
    Total: 242 vectors with norm^2 = 4.
    """
    vecs = []
    # Type A: single coordinate ±2
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            vecs.append(v)
    # Type B: two coordinates ±1
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                vecs.append(v)
    return np.array(vecs, dtype=np.int64)


def _greedy_color(conflict_adj, P, n):
    """Greedy coloring of the subgraph induced by P (list of node indices).
    Returns color_classes: list of lists, each list contains nodes with same color.
    The number of colors is an upper bound on the MIS size of G[P].
    """
    color_classes = []
    uncolored = set(P)
    
    while uncolored:
        color_class = []
        for v in list(uncolored):
            # v can get this color if no neighbor in color_class
            can_color = True
            for u in color_class:
                if conflict_adj[v][u]:
                    can_color = False
                    break
            if can_color:
                color_class.append(v)
                uncolored.discard(v)
        if color_class:
            color_classes.append(color_class)
        else:
            break
    
    return color_classes


def _bb_mcs(conflict_adj, P, n, deadline, best_size_ref):
    """Tomita BB-MCS: branch and bound for Maximum Independent Set.
    conflict_adj: n x n boolean array, True means edge (conflict).
    P: list of candidate nodes to choose from.
    Returns the best independent set found (list of node indices).
    best_size_ref: [int] mutable reference to track best size globally.
    """
    best_set = []
    
    def _search(P_list, current_set):
        nonlocal best_set
        if time.time() > deadline:
            return
        
        if not P_list:
            if len(current_set) > len(best_set):
                best_set = list(current_set)
                best_size_ref[0] = len(current_set)
            return
        
        if len(current_set) + len(P_list) <= len(best_set):
            return
        
        # Greedy coloring for upper bound
        color_classes = _greedy_color(conflict_adj, P_list, n)
        
        # Process colors in reverse (highest color number first)
        for ci in range(len(color_classes) - 1, -1, -1):
            if time.time() > deadline:
                return
            # Upper bound: current_set size + number of colors from ci onwards
            if len(current_set) + ci + 1 <= len(best_set):
                break
            
            color_class = color_classes[ci]
            for v in color_class:
                if v not in P_list:
                    continue
                # Branch: include v
                new_set = current_set + [v]
                # New P: neighbors of v that are in P_list and not conflicting with v
                # Since v is in an independent set, we can only add nodes NOT adjacent to v
                new_P = [u for u in P_list if u != v and not conflict_adj[v][u]]
                _search(new_P, new_set)
                if time.time() > deadline:
                    return
                # Remove v from P_list for next iteration
                P_list.remove(v)
    
    _search(list(P), [])
    return best_set


def _bb_mcs_vectorized(conflict_matrix, P, n, deadline, best_size_ref):
    """Optimized BB-MCS using numpy for the conflict checks.
    conflict_matrix: n x n boolean numpy array.
    """
    best_set = []
    cm = conflict_matrix  # boolean matrix
    
    def _search(P_list, current_set):
        nonlocal best_set
        if time.time() > deadline:
            return
        
        if not P_list:
            if len(current_set) > len(best_set):
                best_set = list(current_set)
                best_size_ref[0] = len(current_set)
            return
        
        if len(current_set) + len(P_list) <= len(best_set):
            return
        
        # Greedy coloring
        color_classes = []
        uncolored = set(P_list)
        while uncolored:
            color_class = []
            for v in sorted(uncolored):  # deterministic order
                can = True
                for u in color_class:
                    if cm[v, u]:
                        can = False
                        break
                if can:
                    color_class.append(v)
                    uncolored.discard(v)
            if color_class:
                color_classes.append(color_class)
        
        # Process in reverse color order
        for ci in range(len(color_classes) - 1, -1, -1):
            if time.time() > deadline:
                return
            if len(current_set) + ci + 1 <= len(best_set):
                break
            
            for v in color_classes[ci]:
                if v not in P_list:
                    continue
                new_set = current_set + [v]
                # Nodes compatible with v: not in P_list already handled, 
                # and not conflicting with v
                new_P = [u for u in P_list if u != v and not cm[v, u]]
                _search(new_P, new_set)
                if time.time() > deadline:
                    return
                P_list.remove(v)
    
    _search(list(P), [])
    return best_set


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE - SAFETY_MARGIN

    # Generate norm-4 candidates
    candidates = _generate_norm4_candidates()
    n = len(candidates)
    
    # For norm-4 vectors: constraint is ||u-v||^2 >= 4
    # ||u-v||^2 = ||u||^2 + ||v||^2 - 2*u.v = 4 + 4 - 2*u.v = 8 - 2*u.v >= 4
    # So u.v <= 2
    # Conflict if u.v > 2
    
    # Build conflict matrix
    # Dot products
    dots = candidates @ candidates.T  # n x n
    # Conflict if dot > 2 (and i != j)
    conflict = (dots > 2)
    np.fill_diagonal(conflict, False)
    
    # Convert to list of lists for faster indexing in some cases
    # Actually let's use numpy directly
    
    # First, get a good initial solution via greedy
    rng = np.random.default_rng(42)
    
    # Greedy to get initial lower bound
    def _greedy_init():
        kept = []
        order = rng.permutation(n)
        for idx in order:
            v = idx
            ok = True
            for u in kept:
                if conflict[v, u]:
                    ok = False
                    break
            if ok:
                kept.append(v)
        return kept
    
    best_set = _greedy_init()
    best_size = len(best_set)
    
    # Try multiple greedy restarts for a better initial bound
    for _ in range(20):
        if time.time() > deadline:
            break
        s = _greedy_init()
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
    
    # Now use BB-MCS to try to improve
    best_size_ref = [best_size]
    
    # Use all nodes as candidates
    P = list(range(n))
    
    # Run BB-MCS with remaining time
    remaining_deadline = deadline
    result_set = _bb_mcs_vectorized(conflict, P, n, remaining_deadline, best_size_ref)
    
    if len(result_set) > best_size:
        best_set = result_set
        best_size = len(result_set)
    
    # Final: use best_set
    if best_size == 0:
        # Fallback
        vecs = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vecs.append(v)
        return np.array(vecs, dtype=np.int64)
    
    result = candidates[best_set]
    
    # Verify
    n_res = len(result)
    max_norm_sq = int(np.max(np.sum(result**2, axis=1)))
    valid = True
    for i in range(n_res):
        for j in range(i + 1, n_res):
            diff = result[i] - result[j]
            if np.dot(diff, diff) < max_norm_sq:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        vecs = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vecs.append(v)
        return np.array(vecs, dtype=np.int64)
    
    return result
# EVOLVE-BLOCK-END
