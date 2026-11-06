"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def _precompute_dist2():
    """Precompute squared distances between all pairs of grid points."""
    points = list(itertools.product(range(N), repeat=2))
    V = len(points)
    dist2 = np.zeros((V, V), dtype=np.int64)
    for i in range(V):
        xi, yi = points[i]
        for j in range(i + 1, V):
            xj, yj = points[j]
            d = (xi - xj) ** 2 + (yi - yj) ** 2
            dist2[i, j] = d
            dist2[j, i] = d
    return points, dist2

# Precompute at module load time
POINTS, DIST2 = _precompute_dist2()
V = len(POINTS)

def _is_blocked(u, S_set, S_list):
    """
    Check if point u is blocked by the set S.
    u is blocked if there exist a, b in S such that triangle (u, a, b) is isosceles.
    S_set is a set of indices for O(1) membership check.
    S_list is a list of indices for iteration.
    """
    n = len(S_list)
    if n < 2:
        return False
    
    # Check 1: d(u, a) == d(u, b) for some a, b in S
    # This is equivalent to checking if any distance from u to S is repeated.
    # Since we are checking a specific u against a fixed S, we can just look at distances.
    # However, we also need to check d(u, a) == d(a, b) and d(u, b) == d(a, b).
    
    # Optimization: 
    # Precompute distances from u to all points in S.
    # d_u = [DIST2[u, a] for a in S_list]
    
    # Condition 1: Any duplicate in d_u?
    # If yes, u is equidistant to two points in S -> Isosceles.
    # Condition 2: For any a in S, is d(u, a) equal to d(a, b) for some b in S?
    
    # Let's do the direct check but optimized.
    # 1. Check if u has duplicate distances to S.
    dists_u = set()
    for a in S_list:
        d = DIST2[u, a]
        if d in dists_u:
            return True
        dists_u.add(d)
        
    # 2. Check if any d(u, a) matches d(a, b) for any b in S.
    # We can precompute a set of all distances between pairs in S: D_S = {d(a,b) | a,b in S}
    # But D_S changes with S.
    # Alternatively, for each a in S, check if d(u,a) is in {d(a,b) for b in S}.
    
    # To make this faster, we can precompute for the current S the set of all internal distances.
    # However, _is_blocked is called very frequently.
    # Let's assume the caller provides S_list and maybe a precomputed set of internal distances?
    # No, let's keep it self-contained but efficient.
    
    for a in S_list:
        d_ua = DIST2[u, a]
        # Check if d_ua matches distance from a to any other b in S
        # We can iterate b in S_list
        for b in S_list:
            if a == b:
                continue
            if d_ua == DIST2[a, b]:
                return True
    return False

def _is_valid_set(S_set):
    """
    Check if S_set is a valid no-isosceles set.
    i.e., no three points in S form an isosceles triangle.
    """
    S_list = list(S_set)
    n = len(S_list)
    if n < 3:
        return True
        
    # Check all triples? O(n^3) is too slow if n is large.
    # But n is small (maybe < 30).
    # Better: For each pair (a, b), check if there is a c such that d(a,c)==d(a,b) or d(b,c)==d(a,b) or d(a,c)==d(b,c).
    # Actually, simpler:
    # For each point a in S, the distances to all other points in S must be distinct.
    # AND for any pair (a, b), d(a, b) must not be equal to d(a, c) or d(b, c) for any c.
    
    # Let's check condition: No isosceles triangle.
    # An isosceles triangle (a, b, c) exists if:
    # d(a,b) == d(a,c)  (a is apex)
    # d(a,b) == d(b,c)  (b is apex)
    # d(a,b) == d(c,b)  (b is apex, same as above)
    # d(a,c) == d(b,c)  (c is apex)
    
    # So, for every pair (a, b):
    # 1. No c in S \ {a,b} such that d(a,c) == d(a,b)
    # 2. No c in S \ {a,b} such that d(b,c) == d(a,b)
    # 3. No c in S \ {a,b} such that d(a,c) == d(b,c)
    
    # Optimization:
    # Precompute for each point a, the set of distances to other points in S.
    # dists[a] = set()
    for a in S_list:
        dists_a = set()
        for b in S_list:
            if a == b:
                continue
            d = DIST2[a, b]
            if d in dists_a:
                return False
            dists_a.add(d)
            
    # Now check cross conditions.
    # For each pair (a, b):
    # d_ab = DIST2[a, b]
    # We need to ensure no c exists such that d(a,c) == d_ab (checked above for a)
    # and no c exists such that d(b,c) == d_ab (checked above for b)
    # and no c exists such that d(a,c) == d(b,c).
    
    # The last condition: d(a,c) == d(b,c) means c is on the perpendicular bisector of ab.
    # This is harder to check with just distance sets.
    # Let's just iterate pairs and check the third condition.
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            d_ab = DIST2[a, b]
            # Check if there is c such that d(a,c) == d(b,c)
            # We can just check all c != a, b
            # But we already know d(a,c) != d(a,b) and d(b,c) != d(a,b) from the first loop.
            # We need to check if d(a,c) == d(b,c).
            # This is O(n^3) in worst case if we do it naively.
            # But n is small.
            for k in range(n):
                c = S_list[k]
                if c == a or c == b:
                    continue
                if DIST2[a, c] == DIST2[b, c]:
                    return False
    return True

def _is_maximal(S_set):
    """
    Check if S_set is maximal.
    i.e., every point u not in S_set is blocked by S_set.
    """
    S_list = list(S_set)
    if not S_list:
        return V == 0 # Empty set is maximal only if grid is empty, which is not the case.
        
    # If S is empty, no point is blocked, so it's not maximal (unless V=0).
    # If S has 1 point, no point is blocked (need 2 points to form a triangle). So not maximal.
    if len(S_list) < 2:
        return False

    for u in range(V):
        if u in S_set:
            continue
        if not _is_blocked(u, S_set, S_list):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 290.0 # 10s buffer for safety
    
    best_size = V
    best_set = []
    
    # Strategy: Simulated Annealing
    # State: A set S of indices.
    # We want to minimize |S| subject to S being valid (no isosceles) and maximal.
    # Actually, the problem asks for a "maximal no-isosceles set".
    # "Maximal" means you can't add any more points.
    # "No-isosceles" means the set itself contains no isosceles triangles.
    # So S must be valid (no isosceles) AND maximal (every other point is blocked).
    
    # Initial Solution:
    # Generate a random valid set, then extend it to be maximal by adding random points that don't break validity.
    
    def _generate_initial():
        S = set()
        # Randomly pick points until we can't add more without breaking validity?
        # No, we want a maximal set.
        # Greedy random construction:
        # 1. Shuffle all points.
        # 2. Iterate through them. If adding the point keeps the set valid (no isosceles), add it.
        # This produces a maximal valid set (since we tried every point).
        # Is it guaranteed to be "maximal no-isosceles"? Yes, because we considered all points.
        # If a point wasn't added, it's because adding it would create an isosceles triangle with existing points.
        # So it is "blocked" in the sense that it can't be added to the set while maintaining validity.
        # Wait, the definition of "maximal no-isosceles set" usually means:
        # 1. The set S has no isosceles triangles.
        # 2. For any point p not in S, S U {p} HAS an isosceles triangle.
        # This is exactly what the greedy algorithm produces.
        
        order = list(range(V))
        random.shuffle(order)
        for p in order:
            # Check if adding p to S creates an isosceles triangle.
            if _can_add(p, S):
                S.add(p)
        return S

    def _can_add(p, S):
        """Check if p can be added to S without creating an isosceles triangle."""
        if len(S) < 2:
            return True
            
        S_list = list(S)
        # Check 1: d(p, a) must be distinct for all a in S.
        dists_p = set()
        for a in S_list:
            d = DIST2[p, a]
            if d in dists_p:
                return False
            dists_p.add(d)
            
        # Check 2: d(p, a) != d(a, b) for any a, b in S.
        # And d(p, a) != d(b, c)? No, just triangle (p, a, b).
        # Triangle (p, a, b) is isosceles if:
        # d(p,a) == d(p,b) (Checked in 1)
        # d(p,a) == d(a,b)
        # d(p,b) == d(a,b)
        
        for a in S_list:
            d_pa = DIST2[p, a]
            for b in S_list:
                if a == b:
                    continue
                d_ab = DIST2[a, b]
                if d_pa == d_ab:
                    return False
                # Note: d_pb == d_ab is symmetric to d_pa == d_ab when swapping a,b.
                # So checking d_pa == d_ab for all a,b covers both cases?
                # If d_pb == d_ab, then for pair (b, a), d_pb == d_ba. Yes.
        return True

    # Start SA
    S = _generate_initial()
    current_size = len(S)
    if current_size < best_size:
        best_size = current_size
        best_set = list(S)
        
    T = 10.0 # Initial temperature
    T_min = 0.01
    alpha = 0.999 # Cooling rate
    
    # We will perform moves:
    # 1. Remove a random point from S.
    # 2. Add a random point not in S.
    # 3. Swap: Remove one, add one.
    
    # To ensure the set remains valid (no isosceles), we only accept moves that result in a valid set.
    # But we also need it to be maximal.
    # If we remove a point, the set might not be maximal anymore (some points might become unblocked).
    # If we add a point, the set might not be valid anymore.
    
    # Strategy:
    # Maintain a valid set S.
    # Move 1: Remove point p from S.
    #   - Check if S \ {p} is still valid. (It always is, since removing points doesn't create new triangles).
    #   - Check if S \ {p} is maximal.
    #   - If maximal, accept with probability 1 (since size decreases).
    #   - If not maximal, we could try to "fix" it by adding points, but that complicates things.
    #   - Alternatively, define the state space as "Maximal Valid Sets".
    #   - Transition: Remove p. If not maximal, find a point q that can be added to make it maximal?
    #     This is complex.
    
    # Simpler Strategy: "Deletion with Repair"
    # 1. Start with a maximal valid set.
    # 2. Pick a random point p in S.
    # 3. Try to remove p.
    # 4. If S \ {p} is maximal, accept (size - 1).
    # 5. If S \ {p} is not maximal, find a point q not in S \ {p} that can be added to S \ {p} to make it maximal?
    #    Actually, if S \ {p} is not maximal, it means there exists some u not in S \ {p} that is NOT blocked.
    #    We can try to add u. If u can be added (validity), then S' = (S \ {p}) U {u} is a new maximal valid set?
    #    Not necessarily maximal, but it is valid. We can then extend it to maximal.
    #    This effectively swaps p with u (or adds u if p wasn't needed).
    
    # Let's implement a local search that allows:
    # - Removing a point if the set remains maximal.
    # - Swapping a point p with a point q (not in S) if the new set is valid and maximal.
    
    # To speed up maximality check, we can cache or use incremental updates?
    # For now, full check.
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        
        # Pick a move
        move_type = random.random()
        S_list = list(S)
        
        if not S_list:
            break
            
        if move_type < 0.7:
            # Try removal
            p = random.choice(S_list)
            S_new = S - {p}
            if _is_maximal(S_new):
                S = S_new
                current_size -= 1
                if current_size < best_size:
                    best_size = current_size
                    best_set = list(S)
                # Reset temperature on improvement? Or just continue cooling.
            else:
                # Reject removal
                pass
        else:
            # Try swap
            # Pick p in S to remove
            p = random.choice(S_list)
            # Pick q not in S to add
            # To be efficient, pick q randomly from the complement.
            # Complement size is V - current_size.
            # If complement is large, picking randomly might be slow to find a valid q.
            # But V=256, so it's fine.
            
            # Generate a random q not in S
            q = random.randint(0, V - 1)
            while q in S:
                q = random.randint(0, V - 1)
                
            S_new = (S - {p}) | {q}
            
            # Check validity of S_new
            if _is_valid_set(S_new):
                # Check maximality of S_new
                if _is_maximal(S_new):
                    # Accept swap (size remains same, but we might escape local minima)
                    # Since size is same, we accept with probability 1 to explore.
                    # Or accept with probability exp(-0/T) = 1.
                    S = S_new
                    # current_size unchanged
                else:
                    # Not maximal, reject
                    pass
            else:
                # Not valid, reject
                pass
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            
        # Occasional restart if stuck?
        # If we haven't improved in a long time, we could restart from a new random initial.
        # But SA should handle this.
        
        # Check if we found a very small set
        if best_size <= 8:
            break

    return [POINTS[i] for i in best_set]

# EVOLVE-BLOCK-END
