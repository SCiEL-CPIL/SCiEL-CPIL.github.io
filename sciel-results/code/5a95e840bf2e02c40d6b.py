import time
import random

N = 16
TOTAL_POINTS = N * N

def get_coords(idx):
    return (idx // N, idx % N)

def dist2(i, j):
    xi, yi = get_coords(i)
    xj, yj = get_coords(j)
    return (xi - xj) ** 2 + (yi - yj) ** 2

# Precompute all pairwise distances squared
D = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]
for i in range(TOTAL_POINTS):
    for j in range(TOTAL_POINTS):
        if i != j:
            D[i][j] = dist2(i, j)

# Precompute IsoMask[i][j]: bitmask of k such that {i,j,k} forms an isosceles triangle
# k != i, k != j
IsoMask = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]

for i in range(TOTAL_POINTS):
    for j in range(TOTAL_POINTS):
        if i == j:
            continue
        mask = 0
        d_ij = D[i][j]
        for k in range(TOTAL_POINTS):
            if k == i or k == j:
                continue
            d_ik = D[i][k]
            d_jk = D[j][k]
            if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                mask |= (1 << k)
        IsoMask[i][j] = mask

def compute_blocked_by_set(S_list):
    """Compute the union of all IsoMask[a][b] for a,b in S_list, a != b."""
    blocked = 0
    n = len(S_list)
    for idx_a in range(n):
        a = S_list[idx_a]
        for idx_b in range(n):
            if idx_a == idx_b:
                continue
            blocked |= IsoMask[a][S_list[idx_b]]
    return blocked

def is_no_isosceles(S_list):
    """Check that no three points in S_list form an isosceles triangle."""
    n = len(S_list)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = S_list[i], S_list[j], S_list[k]
                d_ab = D[a][b]
                d_ac = D[a][c]
                d_bc = D[b][c]
                if d_ab == d_ac or d_ab == d_bc or d_ac == d_bc:
                    return False
    return True

def is_maximal(S_list):
    """Check that no point can be added to S_list without creating an isosceles triangle."""
    S_set = set(S_list)
    for p in range(TOTAL_POINTS):
        if p in S_set:
            continue
        # Check if adding p would create an isosceles triangle
        for a in S_list:
            for b in S_list:
                if a == b:
                    continue
                d_pa = D[p][a]
                d_pb = D[p][b]
                d_ab = D[a][b]
                if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                    break
            else:
                # No isosceles triangle with p and any pair in S
                return False
    return True

def greedy_construct(order=None):
    """Greedy construction using a given order of points."""
    if order is None:
        order = list(range(TOTAL_POINTS))
    
    S = []
    blocked = 0  # Points that are blocked (cannot be added)
    
    for p in order:
        if blocked & (1 << p):
            continue
        # Check if adding p creates an isosceles triangle with existing S
        # p is not blocked means it doesn't form isosceles with any pair in S
        # But we need to verify: blocked was computed from previous additions
        # Actually, we need to check against current S
        # Let's recompute: for each a in S, does (p, a, b) form isosceles for some b in S?
        # That's what blocked tracks. If p is not in blocked, it's safe to add.
        
        S.append(p)
        # Update blocked: add points that form isosceles with p and any a in S (before adding p)
        for a in S[:-1]:
            blocked |= IsoMask[p][a]
        blocked |= (1 << p)  # p itself is now in S
    
    return S

def prune_essential(S_list, deadline):
    """Iteratively remove non-essential points from S_list.
    A point is essential if removing it makes the set non-maximal.
    Returns a maximal no-isosceles subset of S_list."""
    S = list(S_list)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        n = len(S)
        for i in range(n):
            if time.time() > deadline:
                break
            # Try removing S[i]
            candidate = S[:i] + S[i+1:]
            if not is_no_isosceles(candidate):
                continue
            if not is_maximal(candidate):
                continue
            # S[i] is non-essential
            S = candidate
            improved = True
            break  # Restart the scan
    
    return S

def local_search(S_list, deadline, max_iters=50):
    """Try to improve by swapping points.
    Try removing one point and adding another to reduce size or maintain maximality with smaller size."""
    S = list(S_list)
    best = S
    best_size = len(S)
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try single removals that maintain maximality
        improved = False
        n = len(S)
        for i in range(n):
            if time.time() > deadline:
                break
            candidate = S[:i] + S[i+1:]
            if is_no_isosceles(candidate) and is_maximal(candidate):
                S = candidate
                best = S
                best_size = len(S)
                improved = True
                break
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    best_S = None
    best_size = float('inf')
    
    # Phase 1: Multiple greedy constructions with different orderings
    # Use several deterministic orderings
    orderings = [
        list(range(TOTAL_POINTS)),
        list(range(TOTAL_POINTS - 1, -1, -1)),
    ]
    
    # Add some randomized orderings
    rng = random.Random(42)
    for _ in range(5):
        order = list(range(TOTAL_POINTS))
        rng.shuffle(order)
        orderings.append(order)
    
    for order in orderings:
        if time.time() > deadline - 10:
            break
        S = greedy_construct(order)
        if len(S) < best_size:
            best_size = len(S)
            best_S = S
    
    if best_S is None:
        # Fallback: simple greedy
        best_S = greedy_construct()
        best_size = len(best_S)
    
    # Phase 2: Prune non-essential points
    if time.time() < deadline - 5:
        pruned = prune_essential(best_S, deadline - 5)
        if len(pruned) < best_size:
            best_size = len(pruned)
            best_S = pruned
    
    # Phase 3: Local search for further improvements
    if time.time() < deadline - 2:
        improved = local_search(best_S, deadline - 2)
        if len(improved) < best_size:
            best_size = len(improved)
            best_S = improved
    
    # Convert to (x, y) tuples
    result = [get_coords(i) for i in best_S]
    return result
