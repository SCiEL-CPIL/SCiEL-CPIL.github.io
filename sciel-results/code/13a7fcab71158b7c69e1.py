"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

_N = 256
_KILL = 90.0
_MARGIN = 8.0

def _build_conflict():
    """Build adjacency lists: conflict[x] = frozenset of y != x that form a 3-AP with x."""
    conflict = [set() for _ in range(_N + 1)]
    for a in range(1, _N + 1):
        for b in range(a + 1, _N + 1):
            # c = 2b - a (a, b, c)
            c = 2 * b - a
            if c <= _N:
                conflict[a].add(c)
                conflict[b].add(c)
                conflict[c].add(a)
                conflict[c].add(b)
                conflict[a].add(b)
                conflict[b].add(a)
            # mid = (a+b)/2 if even
            s = a + b
            if s % 2 == 0:
                m = s // 2
                if 1 <= m <= _N and m != a and m != b:
                    conflict[a].add(m)
                    conflict[b].add(m)
                    conflict[m].add(a)
                    conflict[m].add(b)
                    conflict[a].add(b)
                    conflict[b].add(a)
    return [frozenset(s) for s in conflict]

def _greedy_init(rng, direction='asc'):
    S = []
    order = list(range(1, _N + 1))
    if direction == 'desc':
        order = order[::-1]
    elif direction == 'rand':
        rng.shuffle(order)
    for x in order:
        ok = True
        for y in S:
            if x in _conflict[x] and y in _conflict[x]:
                # check if x and y conflict
                if y in _conflict[x]:
                    ok = False
                    break
        if ok:
            S.append(x)
    return S

def _greedy_init_v2(rng, direction='asc'):
    """Greedy with conflict check."""
    S = set()
    order = list(range(1, _N + 1))
    if direction == 'desc':
        order = order[::-1]
    elif direction == 'rand':
        rng.shuffle(order)
    for x in order:
        # Check x against all in S
        conflict = False
        for y in S:
            if y in _conflict[x]:
                conflict = True
                break
        if not conflict:
            S.add(x)
    return S

def _branch_and_bound(conflict, weights, deadline, max_nodes=5_000_000):
    """
    Maximum weight independent set via branch and bound.
    vertices: 1..N
    weights[v] = 1/v
    conflict[v] = frozenset of conflicting vertices
    """
    N = _N
    # Order vertices by weight descending for better pruning
    verts = list(range(1, N + 1))
    verts.sort(key=lambda v: -weights[v])
    
    best_weight = [0.0]
    best_set = [set()]
    node_count = [0]
    start_time = time.time()
    
    # Precompute for each vertex, which later vertices (in verts order) it conflicts with
    # Use a bitmask or set approach. Since N=256, use Python sets.
    
    # We'll use a recursive approach with a "remaining" set of candidates
    # and a "forbidden" set of vertices that can't be added due to current choices
    
    def bb(candidates, forbidden, current_weight, current_set):
        node_count[0] += 1
        if node_count[0] % 10000 == 0:
            if time.time() > deadline:
                return
        
        if not candidates:
            if current_weight > best_weight[0]:
                best_weight[0] = current_weight
                best_set[0] = set(current_set)
            return
        
        # Upper bound: current_weight + sum of weights of all candidates not in forbidden
        # Quick bound: sum of top weights in candidates
        bound = current_weight
        for v in candidates:
            if v not in forbidden:
                bound += weights[v]
        
        if bound <= best_weight[0] + 1e-12:
            return
        
        # Pick a vertex to branch on - use the one with highest weight among available
        # Find best vertex in candidates not in forbidden
        best_v = None
        best_w = -1.0
        for v in candidates:
            if v not in forbidden and weights[v] > best_w:
                best_w = weights[v]
                best_v = v
        
        if best_v is None:
            # No available vertices
            if current_weight > best_weight[0]:
                best_weight[0] = current_weight
                best_set[0] = set(current_set)
            return
        
        v = best_v
        
        # Branch 1: Include v
        new_candidates = set(candidates)
        new_candidates.discard(v)
        new_forbidden = set(forbidden)
        # Vertices conflicting with v become forbidden
        for u in conflict[v]:
            if u in new_candidates:
                new_candidates.discard(u)
            new_forbidden.add(u)
        
        bb(new_candidates, new_forbidden, current_weight + weights[v], current_set | {v})
        
        if time.time() > deadline:
            return
        
        # Branch 2: Exclude v
        new_candidates2 = set(candidates)
        new_candidates2.discard(v)
        bb(new_candidates2, forbidden, current_weight, current_set)
    
    initial_candidates = set(range(1, N + 1))
    initial_forbidden = set()
    bb(initial_candidates, initial_forbidden, 0.0, set())
    
    return best_set[0], best_weight[0]

def _branch_and_bound_fast(conflict, weights, deadline, max_nodes=2_000_000):
    """
    Faster B&B using list-based approach and tighter bounds.
    Uses the fact that we process vertices in a fixed order.
    """
    N = _N
    verts = list(range(1, N + 1))
    verts.sort(key=lambda v: -weights[v])
    vert_idx = {v: i for i, v in enumerate(verts)}
    
    best_weight = [0.0]
    best_set = [set()]
    node_count = [0]
    start_time = time.time()
    
    # Precompute: for each vertex in verts order, which subsequent vertices it conflicts with
    # conflict_mask[i] = set of indices j > i where verts[i] conflicts with verts[j]
    m = len(verts)
    conflict_after = [set() for _ in range(m)]
    for i in range(m):
        vi = verts[i]
        for j in range(i + 1, m):
            vj = verts[j]
            if vj in conflict[vi]:
                conflict_after[i].add(j)
    
    # Also: for each vertex, which earlier vertices it conflicts with (for pruning)
    # We'll track selected vertices and their indices
    
    def bb(idx, selected_indices, current_weight):
        """
        idx: current position in verts
        selected_indices: set of indices of selected vertices
        current_weight: sum of weights of selected vertices
        """
        node_count[0] += 1
        if node_count[0] % 5000 == 0:
            if time.time() > deadline:
                return
        
        if idx >= m:
            if current_weight > best_weight[0]:
                best_weight[0] = current_weight
                best_set[0] = {verts[i] for i in selected_indices}
            return
        
        # Upper bound: current_weight + sum of weights from idx to m-1
        # (all remaining vertices, even if some conflict)
        bound = current_weight
        for j in range(idx, m):
            bound += weights[verts[j]]
        
        if bound <= best_weight[0] + 1e-12:
            return
        
        v = verts[idx]
        
        # Check if v can be added: no conflict with any selected vertex
        can_add = True
        for si in selected_indices:
            if v in conflict[verts[si]]:
                can_add = False
                break
        
        if can_add:
            # Branch 1: Include v
            new_selected = selected_indices | {idx}
            bb(idx + 1, new_selected, current_weight + weights[v])
            
            if time.time() > deadline:
                return
        
        # Branch 2: Exclude v
        bb(idx + 1, selected_indices, current_weight)
    
    bb(0, set(), 0.0)
    
    return best_set[0], best_weight[0]

def _branch_and_bound_optimized(conflict, weights, deadline):
    """
    Optimized B&B with better ordering and pruning.
    Uses a greedy lower bound to start.
    """
    N = _N
    verts = list(range(1, N + 1))
    verts.sort(key=lambda v: -weights[v])
    m = len(verts)
    
    # Get a good initial solution via greedy
    greedy_set = set()
    for v in verts:
        ok = True
        for u in greedy_set:
            if u in conflict[v]:
                ok = False
                break
        if ok:
            greedy_set.add(v)
    
    greedy_weight = sum(weights[v] for v in greedy_set)
    best_weight = [greedy_weight]
    best_set = [set(greedy_set)]
    node_count = [0]
    start_time = time.time()
    
    # For faster conflict checking, precompute conflict as a set of sets
    # conflict[v] is already a frozenset
    
    # Use iterative deepening or just straight B&B
    # Key insight: process in order of decreasing weight, and at each step
    # decide include/exclude
    
    # To speed up, we can use a "conflict mask" approach
    # For each vertex, store which other vertices it conflicts with
    # When we include vertex v, all its neighbors are excluded from consideration
    
    def bb(idx, forbidden_mask, current_weight, current_set):
        """
        idx: next vertex to consider (in verts order)
        forbidden_mask: set of vertex values that are forbidden (conflict with selected)
        current_weight: weight of current set
        current_set: set of selected vertex values
        """
        node_count[0] += 1
        if node_count[0] % 10000 == 0:
            if time.time() > deadline:
                return
        
        # Find next available vertex
        while idx < m and verts[idx] in forbidden_mask:
            idx += 1
        
        if idx >= m:
            if current_weight > best_weight[0]:
                best_weight[0] = current_weight
                best_set[0] = set(current_set)
            return
        
        v = verts[idx]
        
        # Upper bound
        bound = current_weight
        for j in range(idx, m):
            if verts[j] not in forbidden_mask:
                bound += weights[verts[j]]
        
        if bound <= best_weight[0] + 1e-12:
            return
        
        # Branch 1: Include v
        new_forbidden = forbidden_mask | conflict[v]
        bb(idx + 1, new_forbidden, current_weight + weights[v], current_set | {v})
        
        if time.time() > deadline:
            return
        
        # Branch 2: Exclude v
        bb(idx + 1, forbidden_mask, current_weight, current_set)
    
    bb(0, set(), 0.0, set())
    
    return best_set[0], best_weight[0]

def _hill_climb(S0, conflict, inv, deadline):
    """Local search to improve a solution."""
    S = set(S0)
    weight = sum(inv[x] for x in S)
    best_S = set(S)
    best_w = weight
    eps = 1e-12
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_gain = -1.0
        best_typ = None
        best_x = None
        best_y = None
        
        # Try adding each non-conflicting element
        for x in range(1, _N + 1):
            if x not in S:
                # Check if x conflicts with any in S
                conflict_free = True
                for y in S:
                    if y in conflict[x]:
                        conflict_free = False
                        break
                if conflict_free:
                    g = inv[x]
                    if g > best_gain:
                        best_gain = g
                        best_typ = 'add'
                        best_x = x
                        best_y = None
        
        # Try swaps
        for x in list(S):
            S.discard(x)
            for y in range(1, _N + 1):
                if y not in S:
                    # Check if y conflicts with any in S (x already removed)
                    conflict_free = True
                    for z in S:
                        if z in conflict[y]:
                            conflict_free = False
                            break
                    if conflict_free:
                        g = inv[y] - inv[x]
                        if g > best_gain:
                            best_gain = g
                            best_typ = 'swap'
                            best_x = x
                            best_y = y
                        break
            S.add(x)
        
        if best_gain > eps:
            if best_typ == 'add':
                S.add(best_x)
                weight += inv[best_x]
            else:
                x = best_x
                y = best_y
                S.discard(x)
                weight -= inv[x]
                S.add(y)
                weight += inv[y]
            if weight > best_w:
                best_w = weight
                best_S = set(S)
            improved = True
    
    return best_S

def construct():
    start = time.time()
    deadline = start + _KILL - _MARGIN
    rng = random.Random(20240517)
    inv = [0.0] + [1.0 / x for x in range(1, _N + 1)]
    
    # Build conflict structure
    conflict = _build_conflict()
    
    def w(S):
        return sum(inv[x] for x in S)
    
    # Try B&B with time limit
    best_S = None
    best_w = -1.0
    
    # First, get a good initial solution via multiple greedy inits
    inits = []
    inits.append(_greedy_init_v2(rng, 'asc'))
    inits.append(_greedy_init_v2(rng, 'desc'))
    for _ in range(3):
        inits.append(_greedy_init_v2(rng, 'rand'))
    
    # Improve each with hill climbing
    for S in inits:
        if time.time() > deadline - 30.0:
            break
        opt = _hill_climb(S, conflict, inv, deadline)
        cw = w(opt)
        if cw > best_w:
            best_w = cw
            best_S = opt
    
    # Now try B&B for exact/near-exact solution
    if time.time() < deadline - 5.0:
        bb_set, bb_w = _branch_and_bound_optimized(conflict, inv, deadline)
        if bb_w > best_w:
            best_w = bb_w
            best_S = bb_set
    
    # Final hill climb on best
    if best_S is not None and time.time() < deadline - 2.0:
        opt = _hill_climb(best_S, conflict, inv, deadline)
        cw = w(opt)
        if cw > best_w:
            best_w = cw
            best_S = opt
    
    if best_S is None or len(best_S) == 0:
        best_S = {1}
    
    res = sorted(best_S)
    
    # Defensive verification
    Sset = set(res)
    ok = True
    for a in res:
        for b in res:
            if a >= b:
                continue
            c = 2 * b - a
            if c in Sset and c > b:
                ok = False
    
    if not ok:
        S = _greedy_init_v2(rng, 'asc')
        res = sorted(S)
    
    return res
# EVOLVE-BLOCK-END
