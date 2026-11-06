"""Rewritten program: maximum independent set via branch-and-bound on conflict graph."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _build_conflict_graph(n=128):
    """Vertices 0..n-1 represent numbers 1..n. Edge i-j exists iff there is some
    k distinct from i,j such that {num_i,num_j,num_k} is a forbidden triple
    (all three pairwise gcds equal)."""
    nums = list(range(1, n + 1))
    N = n
    # conflict[i] is a set of j such that i,j conflict
    conflict = [set() for _ in range(N)]
    # Precompute gcd matrix
    g = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            g[i][j] = gcd(nums[i], nums[j])
    # For each pair (i,j), check if there exists k with equal gcds
    # Precompute for each pair the gcd value and whether any k completes it
    for i in range(N):
        for j in range(i + 1, N):
            gij = g[i][j]
            # need gcd(i,k)==gij and gcd(j,k)==gij for some k != i,j
            found = False
            for k in range(N):
                if k == i or k == j:
                    continue
                if g[i][k] == gij and g[j][k] == gij:
                    found = True
                    break
            if found:
                conflict[i].add(j)
                conflict[j].add(i)
    # Convert to bitmasks for speed
    mask = [0] * N
    for i in range(N):
        m = 0
        for j in conflict[i]:
            m |= (1 << j)
        mask[i] = m
    return mask, nums

def _greedy_color_upper_bound(candidates, mask):
    """Upper bound on independent set size in the induced subgraph on candidates.
    Uses greedy sequential coloring: each color class is an independent set, so
    number of colors is an upper bound on the independence number... actually
    the number of colors is an upper bound on the size of a clique, but for
    independence we use: independence number <= number of colors in a proper
    coloring of the graph. Yes: any independent set uses at most one vertex per
    color, so |IS| <= #colors."""
    # Greedy coloring of the induced subgraph
    # Order candidates by descending degree within candidates for tighter bound
    cand = list(candidates)
    # degree within candidates
    deg = {}
    for v in cand:
        deg[v] = 0
    for v in cand:
        m = mask[v] & candidates
        # count bits
        deg[v] = bin(m).count('1')
    cand.sort(key=lambda v: -deg[v])
    color = {}
    num_colors = 0
    for v in cand:
        # find smallest color not used by neighbors
        used = set()
        m = mask[v] & candidates
        # iterate neighbors
        mm = m
        while mm:
            low = mm & (-mm)
            u = low.bit_length() - 1
            if u in color:
                used.add(color[u])
            mm ^= low
        c = 0
        while c in used:
            c += 1
        color[v] = c
        if c >= num_colors:
            num_colors = c + 1
    return num_colors

def _branch_and_bound(mask, nums, deadline):
    N = len(nums)
    # Start with all candidates
    candidates = (1 << N) - 1
    best_set = set()
    best_size = 0
    current = set()

    # Ordering: sort vertices by degree descending to get good initial bound
    # We'll do recursive search. To keep it fast, use a simple but effective
    # recursive B&B with the greedy coloring upper bound.

    def popcount(x):
        return bin(x).count('1')

    # Precompute a static vertex ordering by degree for branching
    degs = [bin(mask[i]).count('1') for i in range(N)]
    order = list(range(N))
    order.sort(key=lambda v: -degs[v])

    def upper_bound(cand_mask):
        # greedy coloring bound
        # get list of vertices
        verts = []
        mm = cand_mask
        while mm:
            low = mm & (-mm)
            v = low.bit_length() - 1
            verts.append(v)
            mm ^= low
        if not verts:
            return 0
        # degree within cand
        deg = {}
        for v in verts:
            deg[v] = bin(mask[v] & cand_mask).count('1')
        verts.sort(key=lambda v: -deg[v])
        color = {}
        nc = 0
        for v in verts:
            used = set()
            mm2 = mask[v] & cand_mask
            while mm2:
                low = mm2 & (-mm2)
                u = low.bit_length() - 1
                if u in color:
                    used.add(color[u])
                mm2 ^= low
            c = 0
            while c in used:
                c += 1
            color[v] = c
            if c >= nc:
                nc = c + 1
        return nc

    # Global deadline
    t_end = deadline

    # We'll maintain a stack-based or recursive search. Use recursion with
    # branch on a vertex: include or exclude.
    # To speed up, we use the standard technique: pick a vertex v, branch on
    # including v (remove its neighbors) or excluding v.

    def search(cand_mask, current_size):
        nonlocal best_set, best_size
        if time.time() > t_end:
            return
        if cand_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current)
            return
        # Upper bound
        ub = upper_bound(cand_mask)
        if current_size + ub <= best_size:
            return
        # Pick a vertex to branch on: the one with max degree in cand
        # (already we can pick first in order)
        # Find a vertex in cand
        mm = cand_mask
        # choose vertex with max degree in cand
        best_v = -1
        best_d = -1
        mm2 = cand_mask
        while mm2:
            low = mm2 & (-mm2)
            v = low.bit_length() - 1
            d = bin(mask[v] & cand_mask).count('1')
            if d > best_d:
                best_d = d
                best_v = v
            mm2 ^= low
        v = best_v
        # Branch 1: include v
        current.add(v)
        new_cand = cand_mask & ~(mask[v] | (1 << v))
        search(new_cand, current_size + 1)
        current.discard(v)
        # Branch 2: exclude v
        search(cand_mask & ~(1 << v), current_size)

    search(candidates, 0)

    if best_size > 0:
        return sorted(nums[i] for i in best_set)
    return None

def _greedy_initial(mask, nums, seed=0):
    """Greedy construction to get a decent initial set."""
    rng = random.Random(seed)
    N = len(nums)
    # order by degree ascending (include low-degree first)
    degs = [bin(mask[i]).count('1') for i in range(N)]
    order = list(range(N))
    rng.shuffle(order)
    order.sort(key=lambda v: degs[v])
    selected = set()
    forbidden = 0  # bitmask of vertices that conflict with selected
    for v in order:
        if v in forbidden:
            continue
        # check v not already selected
        if (1 << v) in [ (1<<s) for s in selected ]:
            continue
        # add v
        selected.add(v)
        forbidden |= mask[v] | (1 << v)
    return sorted(nums[i] for i in selected)

def _is_valid(S, n=128):
    """Check that S has no forbidden triple."""
    Sset = set(S)
    nums = S
    N = len(nums)
    # map value->index
    idx = {v: i for i, v in enumerate(nums)}
    for a_i in range(N):
        for b_i in range(a_i + 1, N):
            a, b = nums[a_i], nums[b_i]
            g = gcd(a, b)
            for c_i in range(b_i + 1, N):
                c = nums[c_i]
                if gcd(a, c) == g and gcd(b, c) == g:
                    return False
    return True

def construct():
    t0 = time.time()
    deadline = t0 + 78.0  # safety margin under 90s
    mask, nums = _build_conflict_graph(128)

    # Get an initial solution via greedy
    best = _greedy_initial(mask, nums, seed=1)
    best_size = len(best)

    # Try branch and bound
    try:
        result = _branch_and_bound(mask, nums, deadline)
        if result is not None and len(result) > best_size:
            best = result
            best_size = len(result)
    except Exception:
        pass

    # If we have spare time, try a few more greedy seeds and local improvement
    if time.time() < deadline - 2.0:
        for seed in range(2, 12):
            if time.time() > deadline - 2.0:
                break
            cand = _greedy_initial(mask, nums, seed=seed)
            if len(cand) > best_size:
                best = cand
                best_size = len(cand)

    # Final safety: ensure nonempty and valid
    if not best:
        best = [1]
    # Validate; if invalid, fall back to a known-safe small set
    try:
        if not _is_valid(best):
            # fall back to all numbers that are pairwise... just return {1}
            best = [1]
    except Exception:
        best = [1]

    return best
# EVOLVE-BLOCK-END
