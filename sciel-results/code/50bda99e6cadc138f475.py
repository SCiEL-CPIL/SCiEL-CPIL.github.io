"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _girth_check_fast(adj_np, n, g):
    """Fast girth check using vectorized non-backtracking walk enumeration.
    adj_np: numpy array shape (n, 4) of int32, each row has 4 neighbors.
    Returns the girth (shortest cycle length) or g if girth >= g.
    Uses the fact that in a bipartite graph, cycles are even, so we only need to check
    lengths 4, 6, 8 (for g=10).
    A cycle of length 2L exists iff there's a non-backtracking closed walk of length 2L
    from some vertex to itself (in a simple graph, any closed NB walk contains a cycle
    of length <= the walk length; for bipartite, the cycle length is even and <= 2L).
    Actually more precisely: a closed NB walk of length L from s to s implies a cycle
    of length <= L. For girth >= g, we need NO closed NB walk of length < g.
    For bipartite: no closed NB walk of length 4, 6, or 8.
    
    We check: for each start s, do DP over (prev, cur) states, checking if cur == s.
    Vectorize over starts using numpy where possible.
    """
    # For bipartite graphs, shortest cycle is even. Check 4, 6, 8.
    # Use bitset approach: for each start s, track set of (prev, cur) pairs.
    # Since n is at most ~320 and degree is 4, states per start per level <= 4*3=12 (NB).
    # Actually max states = n*(k-1) = n*3 for the frontier.
    
    # We'll use a set-based approach but with early termination.
    # Key optimization: for girth >= 10 in bipartite, we need no 4-cycles, 6-cycles, or 8-cycles.
    # 4-cycle: exists iff two vertices share 2+ common neighbors.
    # 6-cycle: exists iff ... (more complex)
    
    # Fast 4-cycle check: for each pair of vertices, count common neighbors.
    # In bipartite: 4-cycle iff two left vertices share 2 right neighbors (or two right share 2 left).
    # Use the adjacency matrix approach.
    
    # Build adjacency matrix as numpy
    adj_mat = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(4):
            adj_mat[i, adj_np[i, j]] = 1
    
    # 4-cycle check: (A^2)[i,j] >= 2 for some i != j means i and j share >= 2 neighbors -> 4-cycle
    A2 = adj_mat @ adj_mat
    for i in range(n):
        for j in range(i+1, n):
            if A2[i, j] >= 2:
                return 4
    
    # 6-cycle check: need a path of length 3 from i to j with i != j and the path not using
    # backtracking, plus edge j-i. More precisely: 6-cycle exists iff there exist i,j with
    # (A^3)[i,j] >= 1 and i != j and the walk i->a->b->j->i is non-backtracking and i,j not adjacent
    # in a way that creates a shorter cycle. Actually simpler: 6-cycle iff there exist i != j
    # such that (A^3)[i,j] >= 1 and not adjacent(i,j) and the specific walk doesn't backtrack.
    # This is getting complex. Let's use the direct NB walk approach for 6 and 8.
    
    # Direct NB walk check for length 6 and 8 from each start.
    # State: set of (prev, cur) for walks of current length starting at s.
    # For speed, we process all starts simultaneously using a list of sets.
    
    # Actually, let's just do it per-start but with numpy-accelerated neighbor lookup.
    # adj_np[i] gives the 4 neighbors of vertex i as a numpy array.
    
    for s in range(n):
        # Length 2: (s, v) for v in adj[s]
        cur_states = set()
        for v in adj_np[s]:
            cur_states.add((s, int(v)))
        
        # Length 3: (v, w) where w in adj[v], w != s
        nxt = set()
        for (u, v) in cur_states:
            for w in adj_np[v]:
                w = int(w)
                if w == u:
                    continue
                if w == s:
                    return 4  # shouldn't happen (would be a 4-cycle already caught)
                nxt.add((v, w))
        cur_states = nxt
        if not cur_states:
            break
        
        # Length 4: check if any (v, w) has w == s -> 4-cycle
        # (already caught by A2 check, but verify)
        nxt = set()
        for (u, v) in cur_states:
            for w in adj_np[v]:
                w = int(w)
                if w == u:
                    continue
                if w == s:
                    return 4
                nxt.add((v, w))
        cur_states = nxt
        if not cur_states:
            break
        
        # Length 5
        nxt = set()
        for (u, v) in cur_states:
            for w in adj_np[v]:
                w = int(w)
                if w == u:
                    continue
                nxt.add((v, w))
        cur_states = nxt
        if not cur_states:
            break
        
        # Length 6: check if any (v, w) has w == s -> 6-cycle
        nxt = set()
        for (u, v) in cur_states:
            for w in adj_np[v]:
                w = int(w)
                if w == u:
                    continue
                if w == s:
                    return 6
                nxt.add((v, w))
        cur_states = nxt
        if not cur_states:
            break
        
        # Length 7
        nxt = set()
        for (u, v) in cur_states:
            for w in adj_np[v]:
                w = int(w)
                if w == u:
                    continue
                nxt.add((v, w))
        cur_states = nxt
        if not cur_states:
            break
        
        # Length 8: check if any (v, w) has w == s -> 8-cycle
        for (u, v) in cur_states:
            for w in adj_np[v]:
                w = int(w)
                if w == u:
                    continue
                if w == s:
                    return 8
    
    return g


def _build_girth_aware_bipartite(n_half, rng, deadline):
    """Build a 4-regular bipartite graph with girth-aware edge sampling.
    Uses a configuration model but biases against creating 4-cycles and 6-cycles.
    Returns adj as numpy array (n, 4) or None if it can't be built in time."""
    n = 2 * n_half
    # Build a random 4-regular bipartite graph using the configuration model
    # with rejection of edges that create short cycles.
    
    # Start with a random perfect matching-based construction
    # Left vertices: 0..n_half-1, Right vertices: n_half..n-1
    
    # Generate 4 perfect matchings between left and right
    adj = np.zeros((n, 4), dtype=np.int32)
    right_of_left = [set() for _ in range(n_half)]
    left_of_right = [set() for _ in range(n_half)]
    
    for match_idx in range(4):
        # Generate a random permutation
        perm = list(range(n_half))
        rng.shuffle(perm)
        
        # Try to assign, rejecting if it would create a 4-cycle
        # A 4-cycle is created if left l is connected to right r, and there already exists
        # another left l2 connected to both l's other rights and r's other lefts.
        # Simpler: just check if the edge (l, r) would create a 4-cycle given current edges.
        
        for i, l in enumerate(range(n_half)):
            r = perm[i]
            # Check if adding edge (l, r) creates a 4-cycle
            # 4-cycle: exists l2 != l and r2 != r such that (l,r2) and (l2,r) and (l2,r2) all exist
            # i.e., l and r share a common neighbor on the other side via two different edges
            # More precisely: 4-cycle through (l,r) exists if there's l2 with
            # (l, r2) in edges and (l2, r2) in edges and (l2, r) in edges for some r2
            # Equivalent: (l, r) creates 4-cycle iff |N(l) ∩ N(r)| >= 1 where N is on the other side
            # Wait, in bipartite: 4-cycle through edge (l,r) iff there exist l2, r2 with
            # edges (l,r2), (l2,r2), (l2,r) all present.
            # This means: there's a right r2 in N(l) (from previous matchings) and a left l2 in N(r) (from previous)
            # such that (l2, r2) is an edge.
            
            creates_4cycle = False
            for r2 in right_of_left[l]:
                for l2 in left_of_right[r]:
                    if r2 in right_of_left[l2]:
                        creates_4cycle = True
                        break
                if creates_4cycle:
                    break
            
            if creates_4cycle and match_idx < 3:
                # Try to find an alternative assignment
                # Swap r with some other r' that doesn't create a 4-cycle
                found = False
                for i2 in range(i+1, n_half):
                    r_alt = perm[i2]
                    # Check if (l, r_alt) creates 4-cycle
                    creates = False
                    for r2 in right_of_left[l]:
                        for l2 in left_of_right[r_alt]:
                            if r2 in right_of_left[l2]:
                                creates = True
                                break
                        if creates:
                            break
                    if not creates:
                        # Also check (l2, r) for the swapped vertex
                        creates2 = False
                        for r2 in right_of_left[i2]:
                            for l2 in left_of_right[r]:
                                if r2 in right_of_left[l2]:
                                    creates2 = True
                                    break
                            if creates2:
                                break
                        if not creates2:
                            # Do the swap
                            perm[i], perm[i2] = perm[i2], perm[i]
                            r = perm[i]
                            found = True
                            break
                if not found:
                    # Just accept it (we'll fix later)
                    pass
            
            # Assign edge
            adj[l, match_idx] = n_half + r
            adj[n_half + r, match_idx] = l
            right_of_left[l].add(r)
            left_of_right[r].add(l)
    
    # Now do local search to eliminate 4-cycles and improve girth
    # Convert to list-of-lists for easier manipulation
    adj_list = [list(adj[i]) for i in range(n)]
    
    # Iteratively fix 4-cycles by swapping edges
    for iteration in range(500):
        if time.time() > deadline:
            break
        # Find a 4-cycle
        found_cycle = False
        for l1 in range(n_half):
            if found_cycle:
                break
            for l2 in range(l1+1, n_half):
                if found_cycle:
                    break
                # Check if l1 and l2 share 2+ right neighbors
                common = right_of_left[l1] & right_of_left[l2]
                if len(common) >= 2:
                    # 4-cycle exists: l1-r1-l2-r2-l1
                    r1, r2 = list(common)[:2]
                    # Swap: l1's r2 -> l3's r3, l2's r1 -> l3's r1 (or similar)
                    # Find l3 != l1,l2 such that (l3, r1) not in edges and (l3, r2) not in edges
                    # and swapping doesn't create new 4-cycles
                    fixed = False
                    for l3 in range(n_half):
                        if l3 == l1 or l3 == l2:
                            continue
                        if r1 in right_of_left[l3] or r2 in right_of_left[l3]:
                            continue
                        # Swap l1-r2 with l3-r1: l1 gets r1 (keep), l2 gets... 
                        # Actually: remove (l1,r2) and (l2,r1), add (l1,r1) and (l2,r2)? No that's the same.
                        # We need to break the 4-cycle. Remove one of the common edges.
                        # Swap: l1's edge to r2 becomes l3's edge to r2, and l3's edge to r1 becomes l1's edge to r1.
                        # Wait, this is getting complicated. Let's just do a simple 2-opt swap.
                        
                        # Find which match_idx l1 has r2
                        idx1 = None
                        for mi in range(4):
                            if adj_list[l1][mi] == n_half + r2:
                                idx1 = mi
                                break
                        idx3 = None
                        for mi in range(4):
                            if adj_list[l3][mi] == n_half + r1:
                                idx3 = mi
                                break
                        if idx1 is None or idx3 is None:
                            continue
                        
                        # Perform swap: l1's r2 <-> l3's r1
                        adj_list[l1][idx1] = n_half + r1
                        adj_list[l3][idx3] = n_half + r2
                        adj_list[n_half + r2][idx1] = l3
                        adj_list[n_half + r1][idx3] = l1
                        right_of_left[l1].discard(r2)
                        right_of_left[l1].add(r1)
                        right_of_left[l3].discard(r1)
                        right_of_left[l3].add(r2)
                        left_of_right[r2].discard(l1)
                        left_of_right[r2].add(l3)
                        left_of_right[r1].discard(l3)
                        left_of_right[r1].add(l1)
                        fixed = True
                        found_cycle = True
                        break
                    if fixed:
                        break
        if not found_cycle:
            break
    
    # Rebuild numpy array
    adj_np = np.array(adj_list, dtype=np.int32)
    return adj_np


def construct(k, g):
    """Construct a k-regular graph of girth at least g on as few vertices as possible.
    Returns (n, edges)."""
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    rng = random.Random(1234567)
    
    if k != 4 or g != 10:
        return _luw_fallback(k, g)
    
    # Search for n < 320 with girth >= 10
    best = None  # (n, edges)
    
    # Try n from 242 upward
    for n in range(242, 320, 2):
        if time.time() > deadline:
            break
        n_half = n // 2
        
        # Try girth-aware construction
        for attempt in range(50):
            if time.time() > deadline:
                break
            adj_np = _build_girth_aware_bipartite(n_half, rng, deadline)
            if adj_np is None:
                break
            
            # Verify degree and simplicity
            ok = True
            for v in range(n):
                if len(set(adj_np[v])) != 4:
                    ok = False
                    break
            if not ok:
                continue
            
            # Check girth
            cur_girth = _girth_check_fast(adj_np, n, g)
            if cur_girth >= g:
                edges = []
                for v in range(n_half):
                    for w in adj_np[v]:
                        edges.append((v, int(w)))
                best = (n, edges)
                break
            # If girth is close, do local search
            elif cur_girth >= 8:
                # Try to improve via random swaps
                adj_list = [list(adj_np[v]) for v in range(n)]
                for _ in range(100):
                    if time.time() > deadline:
                        break
                    l1 = rng.randrange(n_half)
                    l2 = rng.randrange(n_half)
                    if l1 == l2:
                        continue
                    r1 = rng.choice(adj_list[l1])
                    r2 = rng.choice(adj_list[l2])
                    r1, r2 = int(r1), int(r2)
                    if r1 == r2:
                        continue
                    # Check no duplicate would be created
                    if r2 in adj_list[l1] or r1 in adj_list[l2]:
                        continue
                    # Swap
                    adj_list[l1][adj_list[l1].index(r1)] = r2
                    adj_list[l2][adj_list[l2].index(r2)] = r1
                    adj_list[r1].remove(l1); adj_list[r1].append(l2)
                    adj_list[r2].remove(l2); adj_list[r2].append(l1)
                    adj_np2 = np.array(adj_list, dtype=np.int32)
                    new_girth = _girth_check_fast(adj_np2, n, g)
                    if new_girth > cur_girth:
                        cur_girth = new_girth
                        adj_np = adj_np2
                        if cur_girth >= g:
                            edges = []
                            for v in range(n_half):
                                for w in adj_np[v]:
                                    edges.append((v, int(w)))
                            best = (n, edges)
                            break
                    else:
                        # Revert
                        adj_list[l1][adj_list[l1].index(r2)] = r1
                        adj_list[l2][adj_list[l2].index(r1)] = r2
                        adj_list[r1].remove(l2); adj_list[r1].append(l1)
                        adj_list[r2].remove(l1); adj_list[r2].append(l2)
                if best is not None:
                    break
        
        if best is not None:
            break
    
    if best is None:
        return _luw_fallback(k, g)
    
    return best


def _luw_fallback(k, g):
    """Original Lazebnik-Ustimenko-Woldar construction as fallback.
    For k=4, g=10: produces a 4-regular graph of girth 10 on 320 vertices."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

# EVOLVE-BLOCK-END
