import time
import random
import sys

# --- Problem Constants ---
N = 16
POINTS = [(x, y) for x in range(N) for y in range(N) for _ in [0]]
# Ensure unique points
POINTS = list(set(POINTS))
# Sort for determinism in precomputation, though randomization is used later
POINTS.sort()
P = len(POINTS)
POINT_INDEX = {p: i for i, p in enumerate(POINTS)}

# Precomputation
# We model the problem as finding a Minimum Maximal Independent Set (MMIS) in a hypergraph.
# Vertices: Grid points (P=256)
# Hyperedges: Triples {u, v, w} that form an isosceles triangle.
# A set S is independent if it contains no hyperedge.
# A set S is maximal independent if for every v not in S, S U {v} is NOT independent.
# This is equivalent to: for every v not in S, there exists a hyperedge e such that e is a subset of S U {v} and v is in e.
# In other words, v is "blocked" by S.
# We want to minimize |S|.

# For fast checks, we precompute for each pair (u, v) the set of w such that {u, v, w} is isosceles.
# Let iso_partners[u][v] be a bitmask of w's.
# Note: The triangle {u, v, w} is isosceles if any two sides are equal.
# d(u,v) == d(u,w) OR d(u,v) == d(v,w) OR d(v,w) == d(u,w).

# To check if a candidate point c can be added to a set S (maintaining independence):
# For all u, v in S, {c, u, v} must NOT be isosceles.
# This is equivalent to: For all u in S, c is NOT in iso_partners[u][v] for any v in S?
# Actually, let's refine the "blocked" check.
# c is blocked by S if there exist u, v in S such that {c, u, v} is isosceles.
# This can be checked by:
# 1. For each u in S, check if c is in iso_partners[u][v] for any v in S.
#    Wait, iso_partners[u][v] contains w such that {u, v, w} is isosceles.
#    So if c is in iso_partners[u][v], then {c, u, v} is isosceles.
#    So we need to check if there exist u, v in S such that c in iso_partners[u][v].
#    This is equivalent to: (S_mask & (1 << u)) and (S_mask & (1 << v)) and (c in iso_partners[u][v]).
#    We can precompute for each c, a list of pairs (u, v) that block c.
#    Or better: For each c, precompute a bitmask `blockers_of_c`? No, blockers depend on S.
#    
#    Let's use the property: c is blocked by S iff there exists u in S such that (S & iso_partners[u][c]) is non-empty?
#    Let's check:
#    If there is u in S and v in S such that {c, u, v} is isosceles.
#    Then v is in iso_partners[u][c] (since {u, c, v} is isosceles).
#    So if u is in S, and (S & iso_partners[u][c]) is non-empty, then there is a v in S that forms an isosceles triangle with u and c.
#    So yes, c is blocked by S iff exists u in S such that (S & iso_partners[u][c]) != 0.

# Precompute iso_partners[u][c] as a bitmask of v's.
# u, c, v indices.
iso_partners = [[0] * P for _ in range(P)]

# Optimization: Symmetry? iso_partners[u][c] is not necessarily symmetric in u and c?
# {u, c, v} is isosceles is symmetric in u and c. So iso_partners[u][c] == iso_partners[c][u].
# But we need to be careful with the definition.
# d(u,c) == d(u,v) OR d(u,c) == d(c,v) OR d(c,v) == d(u,v).
# This condition is symmetric in u and c.
# So we can compute for u < c and mirror, or just compute all. P=256, P^2=65k, inner loop P=256. 16M ops. Fast enough.

# Let's compute distances squared first for speed.
coords = [(p[0], p[1]) for p in POINTS]
dist2 = [[0]*P for _ in range(P)]
for i in range(P):
    xi, yi = coords[i]
    for j in range(P):
        xj, yj = coords[j]
        dist2[i][j] = (xi-xj)**2 + (yi-yj)**2

for u in range(P):
    for c in range(P):
        if u == c:
            continue
        mask = 0
        du_c = dist2[u][c]
        for v in range(P):
            if v == u or v == c:
                continue
            # Check if {u, c, v} is isosceles
            # Sides: du_c, du_v, dc_v
            du_v = dist2[u][v]
            dc_v = dist2[c][v]
            if du_c == du_v or du_c == dc_v or du_v == dc_v:
                mask |= (1 << v)
        iso_partners[u][c] = mask

def is_blocked(c, S_mask):
    """
    Check if point c is blocked by set S_mask.
    c is blocked if there exists u in S such that (S & iso_partners[u][c]) != 0.
    """
    # Iterate through u in S
    # We can iterate through set bits of S_mask
    s = S_mask
    while s:
        # Get lowest set bit
        lsb = s & -s
        u = lsb.bit_length() - 1
        if S_mask & iso_partners[u][c]:
            return True
        s ^= lsb
    return False

def is_maximal(S_mask):
    """
    Check if S_mask is a maximal independent set.
    All points not in S must be blocked.
    """
    # Points not in S
    not_in_S = ((1 << P) - 1) ^ S_mask
    s = not_in_S
    while s:
        lsb = s & -s
        c = lsb.bit_length() - 1
        if not is_blocked(c, S_mask):
            return False
        s ^= lsb
    return True

def is_independent(S_mask):
    """
    Check if S_mask is an independent set (no isosceles triangles).
    For all u, v in S, S & iso_partners[u][v] must be 0?
    Wait. If u, v are in S, then any w in S such that {u, v, w} is isosceles violates independence.
    So for all u, v in S, (S & iso_partners[u][v]) must be 0.
    Note: iso_partners[u][v] contains w. It does not contain u or v (we skipped them in precomp).
    """
    s = S_mask
    while s:
        lsb_u = s & -s
        u = lsb_u.bit_length() - 1
        s_rest = s ^ lsb_u
        # For each v in s_rest
        t = s_rest
        while t:
            lsb_v = t & -t
            v = lsb_v.bit_length() - 1
            if S_mask & iso_partners[u][v]:
                return False
            t ^= lsb_v
        s = s_rest
    return True

def construct():
    start_time = time.time()
    time_limit = 300.0  # 30s margin for safety
    
    best_size = P
    best_set = 0
    
    # Strategy:
    # 1. Generate random independent sets (greedy or random).
    # 2. Ensure maximality by adding points? No, we want minimal maximal.
    #    Starting from a maximal independent set, we try to remove points while maintaining maximality.
    #    This is the "Minimality Reduction" phase.
    #    How to get a starting maximal independent set?
    #    Random Greedy: Pick random points, add if independent. Continue until no more can be added.
    #    This yields a maximal independent set.
    
    # Improvement:
    # Instead of simple removal, we can use a "repair" strategy.
    # If removing v makes S non-maximal, there is some u not in S that is no longer blocked.
    # We must add u back? Or add some other point to block u?
    # Actually, if S \ {v} is not maximal, it means there exists u not in S \ {v} such that u is not blocked by S \ {v}.
    # Since S was maximal, u was blocked by S.
    # u was blocked by S because there existed x, y in S such that {u, x, y} is isosceles.
    # If v is removed, maybe v was x or y?
    # If v was part of the blocking triangle for u, then removing v might unblock u.
    # To keep S \ {v} maximal, we must ensure u is blocked again.
    # We can try to add a different point w to block u?
    # But adding w increases size. We want to decrease size.
    # So if removal of v is not directly possible, we might try a swap: remove v, add w.
    # But swap keeps size same. We need net removal.
    
    # Let's stick to the proven "Removal with Maximality Check" but optimize the check.
    # is_maximal is O(P * |S| * bit_ops).
    # With P=256, |S|~20, this is fast.
    
    # To find smaller sets, we can also try to "shake" the set.
    # 1. Remove a point v.
    # 2. If not maximal, find the unblocked point u.
    # 3. Try to add a point w (not v) that blocks u and is independent with S \ {v}.
    # 4. If such w exists, swap v for w? No, that's size constant.
    # 5. Maybe w allows further removals?
    
    # Let's try a more aggressive local search:
    # Random Restart + Greedy Construction + Local Search (Removal + Swap).
    
    for restart in range(10000):
        if time.time() - start_time > time_limit:
            break
        
        # 1. Greedy Construction
        # Random order
        order = list(range(P))
        random.shuffle(order)
        
        S_mask = 0
        for p in order:
            # Check if p can be added
            # p can be added if it is independent with S.
            # Independence: For all u in S, (S & iso_partners[u][p]) == 0?
            # Wait, iso_partners[u][p] contains v such that {u, p, v} is isosceles.
            # So if any v in S is in iso_partners[u][p], conflict.
            # So check: For all u in S, (S & iso_partners[u][p]) == 0.
            
            can_add = True
            s = S_mask
            while s:
                lsb = s & -s
                u = lsb.bit_length() - 1
                if S_mask & iso_partners[u][p]:
                    can_add = False
                    break
                s ^= lsb
            
            if can_add:
                S_mask |= (1 << p)
        
        current_size = bin(S_mask).count('1')
        
        # 2. Local Search: Minimality Reduction
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Try removing points in random order
            removal_order = list(range(P))
            random.shuffle(removal_order)
            
            for v in removal_order:
                if not (S_mask & (1 << v)):
                    continue
                
                new_S_mask = S_mask & ~(1 << v)
                
                # Check if new_S_mask is maximal
                if is_maximal(new_S_mask):
                    S_mask = new_S_mask
                    current_size -= 1
                    improved = True
                    if current_size < best_size:
                        best_size = current_size
                        best_set = S_mask
                else:
                    # Optional: Try to repair by swapping?
                    # If not maximal, find unblocked point u.
                    # Then try to find w to add such that S\{v} U {w} is independent and maximal?
                    # This is complex. Let's stick to removal for now, as it's the primary driver for size reduction.
                    # Swaps are useful to escape local minima in size, but removal is the goal.
                    pass
        
        if current_size < best_size:
            best_size = current_size
            best_set = S_mask

    # Convert best_set to list of points
    result = []
    for i in range(P):
        if best_set & (1 << i):
            result.append(POINTS[i])
    
    return result

# To ensure we don't hit the time limit during precomputation or other overheads,
# we rely on the time check inside the loop.
# Precomputation is ~16M ops, should be < 1s in Python with bit ops?
# Let's estimate: 256*256*256 = 16,777,216 iterations.
# Each iteration has a few distance lookups and bit ops.
# In Python, 16M iterations might take ~1-2 seconds. Acceptable.

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Found set of size {len(res)}")
    # Verify
    if not is_independent(0): pass # Dummy
    # Construct mask from res for verification
    mask = 0
    for p in res:
        idx = POINT_INDEX[p]
        mask |= (1 << idx)
    if not is_independent(mask):
        print("ERROR: Not independent")
    if not is_maximal(mask):
        print("ERROR: Not maximal")
