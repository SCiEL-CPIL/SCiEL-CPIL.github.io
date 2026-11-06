"""
4-colouring of K_n with no monochromatic triangle (R(3,3,3,3) > n).

Strategy:
- Start from the known n=32 Greenwood-Gleason base (two copies of K16 with
  cubic-residue 3-colouring, colour 3 on all cross edges).
- Greedily add vertices one at a time, choosing for each new vertex the
  attachment pattern (a 4-tuple of colour-subsets to the existing vertices)
  that minimises the number of new monochromatic triangles created.
- After the greedy phase, run simulated annealing on the full edge-colouring
  to repair any monochromatic triangles that the greedy phase left behind,
  trying to reach n=50 or as high as possible within the time budget.

Deadline guard: track wall-clock and return best valid result before 150 s.
"""
import time
import random
import math

_DEADLINE = 145.0  # seconds from start of construct()
_START_TIME = None

def _elapsed():
    return time.time() - _START_TIME

def _edge_idx(i, j, n):
    """Lexicographic index of edge (i,j) in the flat colour array."""
    if i > j:
        i, j = j, i
    # number of edges before row i
    return i * n - i * (i + 1) // 2 - i + (j - i - 1)

def _edge_idx_simple(i, j):
    """Index in lexicographic order for vertices 0..n-1, i<j."""
    # edges: (0,1),(0,2),...,(0,n-1),(1,2),...
    # index = sum_{k=0}^{i-1} (n-1-k) + (j - i - 1)
    return i * (2 * n - i - 1) // 2 + (j - i - 1)

def _count_mono_triangles(colors, n):
    """Count monochromatic triangles in the colouring."""
    # Build adjacency per colour
    # For speed, use bitmasks
    adj = [[0] * n for _ in range(4)]
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[_edge_idx_simple(i, j)]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
    count = 0
    for c in range(4):
        for i in range(n):
            # for each pair j,k in adj[c][i] with j<k
            mask = adj[c][i]
            # iterate over set bits
            m = mask
            while m:
                lb = m & (-m)
                j = lb.bit_length() - 1
                m ^= lb
                # k must be > j and in adj[c][i] and adj[c][j]
                common = adj[c][i] & adj[c][j] & (mask >> (j + 1) << (j + 1))
                # simpler: common = adj[c][i] & adj[c][j], count bits with index > j
                common = adj[c][i] & adj[c][j]
                # mask out bits <= j
                common &= ~((1 << (j + 1)) - 1)
                count += common.bit_count()
    return count

def _greedy_add_vertex(colors, n, new_vertex):
    """
    Add a new vertex at index n to the existing (n)-vertex colouring.
    colors is a list of length n*(n-1)//2 for the old vertices.
    Returns new colors list of length (n+1)*n//2, and the new vertex's
    colour assignments to old vertices.
    """
    best_pattern = None
    best_score = float('inf')
    
    # For each of the 4^? patterns... too many. Instead, for each edge to old
    # vertices, choose a colour that minimises new mono-triangles.
    # Greedy: for each old vertex i, pick colour c for edge (i, new) that
    # minimises the number of vertices j < i such that edge (i,j) has colour c
    # and edge (j, new) would also need to be c (but we haven't assigned that yet).
    
    # Actually, a simpler greedy: assign colours one by one to old vertices,
    # for vertex i pick the colour c that minimises the count of j < i where
    # color(i,j) == c (since if we later assign color(j,new)=c, we'd form a triangle).
    
    # Better: for each old vertex i, the number of new mono-triangles involving
    # edge (i, new) with colour c equals the number of j < i such that
    # color(i,j) == c AND color(j, new) == c. But color(j,new) is determined
    # when we assign it. So this is coupled.
    
    # Simplest effective greedy: for each i from 0 to n-1, pick colour c for
    # (i, new) that minimises the number of j in [0, i) with color(i,j) == c
    # (these are potential triangle partners if their edge to new is also c).
    # This is an upper bound on new triangles.
    
    assignments = [0] * n  # colour for edge (i, new)
    for i in range(n):
        best_c = 0
        best_cnt = float('inf')
        for c in range(4):
            cnt = 0
            for j in range(i):
                if colors[_edge_idx_simple(i, j)] == c:
                    cnt += 1
            if cnt < best_cnt:
                best_cnt = cnt
                best_c = c
        assignments[i] = best_c
    
    # Build new colors
    new_colors = colors[:]
    # Append edges (i, n) for i in 0..n-1
    for i in range(n):
        new_colors.append(assignments[i])
    
    return new_colors, assignments

def _sa_repair(colors, n, time_budget):
    """
    Simulated annealing to reduce mono-triangle count to 0.
    colors: list of length n*(n-1)//2
    Returns (colors, mono_count)
    """
    random.seed(42)
    mono = _count_mono_triangles(colors, n)
    if mono == 0:
        return colors, 0
    
    T = 2.0
    cool = 0.9995
    best_colors = colors[:]
    best_mono = mono
    
    while mono > 0 and _elapsed() < time_budget:
        # Pick a random edge
        idx = random.randrange(len(colors))
        # Decode (i, j) from idx
        # idx = i*(2n-i-1)//2 + (j-i-1)
        # Solve for i: i*(2n-i-1)//2 <= idx
        # Approximate i
        i = int((2 * n - 1 - math.sqrt((2 * n - 1) ** 2 - 8 * idx)) / 2)
        # Verify and adjust
        while i < n - 1:
            base = i * (2 * n - i - 1) // 2
            if base + (n - 1 - i) > idx:
                break
            i += 1
        if i >= n - 1:
            i = n - 2
        base = i * (2 * n - i - 1) // 2
        j = i + 1 + (idx - base)
        
        old_c = colors[idx]
        new_c = random.randrange(4)
        if new_c == old_c:
            continue
        
        # Compute delta in mono-triangle count
        # A triangle (i, j, k) is mono if all three edges have same colour.
        # Changing edge (i,j) from old_c to new_c affects triangles (i,j,k) for all k != i,j.
        # For each k:
        #   If color(i,k) == color(j,k) == old_c, then (i,j,k) was mono (old_c), now not.
        #   If color(i,k) == color(j,k) == new_c, then (i,j,k) was not mono (or was mono other), now mono (new_c).
        delta = 0
        for k in range(n):
            if k == i or k == j:
                continue
            c_ik = colors[_edge_idx_simple(i, k)] if i < k else colors[_edge_idx_simple(k, i)]
            c_jk = colors[_edge_idx_simple(j, k)] if j < k else colors[_edge_idx_simple(k, j)]
            if c_ik == c_jk == old_c:
                delta -= 1
            if c_ik == c_jk == new_c:
                delta += 1
        
        new_mono = mono + delta
        if new_mono < mono or random.random() < math.exp(-delta / T):
            colors[idx] = new_c
            mono = new_mono
            if mono < best_mono:
                best_mono = mono
                best_colors = colors[:]
        else:
            pass  # reject
        
        T *= cool
        if T < 0.01:
            T = 0.01
    
    return best_colors, best_mono

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Build the n=32 Greenwood-Gleason base
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    
    colors32 = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors32.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    
    n = 32
    colors = colors32
    
    # Verify base is valid
    assert _count_mono_triangles(colors, n) == 0, "Base n=32 colouring has mono triangles"
    
    # Greedily add vertices up to n=50 or until time runs out
    target_n = 50
    while n < target_n and _elapsed() < _DEADLINE - 30:
        colors, _ = _greedy_add_vertex(colors, n, n)
        n += 1
    
    # Check current mono count
    mono = _count_mono_triangles(colors, n)
    if mono == 0:
        return {'n': n, 'colors': colors}
    
    # Simulated annealing to repair
    sa_budget = min(_DEADLINE - 5, _elapsed() + 100)
    colors, mono = _sa_repair(colors, n, sa_budget)
    
    if mono == 0:
        return {'n': n, 'colors': colors}
    
    # If SA didn't fully repair, try to find a valid sub-colouring
    # by removing vertices that participate in mono-triangles
    # This is a fallback; ideally we return the best valid n found.
    
    # Try: keep the n=32 base which is known valid
    return {'n': 32, 'colors': colors32}
