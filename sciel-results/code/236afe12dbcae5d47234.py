"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import itertools
import sys

def _precompute_conflicts(N):
    """
    Precompute conflict bitsets.
    conflict[u][v] is a bitset of points w such that {u, v, w} is isosceles.
    Also precompute conf_pairs[w] = list of (u, v) such that {u, v, w} is isosceles.
    """
    points = list(itertools.product(range(N), repeat=2))
    idx_map = {p: i for i, p in enumerate(points)}
    V = len(points)
    
    # conflict[u][v] : bitset of w
    conflict = [[0] * V for _ in range(V)]
    # conf_pairs[w] : list of (u, v)
    conf_pairs = [[] for _ in range(V)]
    
    # Generate all isosceles triples
    # To avoid duplicates, we can just iterate all triples? O(V^3) = 256^3 = 16M. OK.
    # Or iterate pairs (u, v) and find w?
    # For each pair (u, v), the set of w forming isosceles with them is finite.
    # d1 = dist(u, v)
    # w must be on circle centered u with radius d1, or circle centered v with radius d1, or midpoint perpendicular bisector?
    # Actually, isosceles means d(u,w)==d(v,w) or d(u,w)==d(u,v) or d(v,w)==d(u,v).
    
    for i in range(V):
        u = points[i]
        for j in range(i + 1, V):
            v = points[j]
            duv2 = (u[0] - v[0])**2 + (u[1] - v[1])**2
            
            # Case 1: w is equidistant from u and v (d(u,w) == d(v,w))
            # w lies on perpendicular bisector of uv
            # Case 2: d(u,w) == duv
            # Case 3: d(v,w) == duv
            
            # Iterate all w?
            for k in range(V):
                w = points[k]
                duw2 = (u[0] - w[0])**2 + (u[1] - w[1])**2
                dvw2 = (v[0] - w[0])**2 + (v[1] - w[1])**2
                
                if duw2 == dvw2 or duw2 == duv2 or dvw2 == duv2:
                    # {u, v, w} is isosceles
                    # Add w to conflict[u][v]
                    conflict[u_idx][v_idx] |= (1 << k) # Need indices
                    conflict[v_idx][u_idx] |= (1 << k)
                    
                    # Add pair (i, j) to conf_pairs[k]
                    conf_pairs[k].append((i, j))
                    
    # Fix indices in the loop above (pseudo-code)
    # Re-do with proper indices
    conflict = [[0] * V for _ in range(V)]
    conf_pairs = [[] for _ in range(V)]
    
    for i in range(V):
        u = points[i]
        for j in range(i + 1, V):
            v = points[j]
            duv2 = (u[0] - v[0])**2 + (u[1] - v[1])**2
            
            for k in range(V):
                if k == i or k == j: continue
                w = points[k]
                duw2 = (u[0] - w[0])**2 + (u[1] - w[1])**2
                dvw2 = (v[0] - w[0])**2 + (v[1] - w[1])**2
                
                if duw2 == dvw2 or duw2 == duv2 or dvw2 == duv2:
                    conflict[i][j] |= (1 << k)
                    conflict[j][i] |= (1 << k)
                    conf_pairs[k].append((i, j))
                    
    return points, conflict, conf_pairs

def construct():
    points, conflict, conf_pairs = _precompute_conflicts(N)
    V = len(points)
    full_mask = (1 << V) - 1
    
    # Greedy to get initial best
    def greedy_maximal():
        avail = full_mask
        S = []
        S_mask = 0
        blocked = 0
        while avail:
            # Pick a vertex
            # Heuristic: pick one that blocks most
            best_v = -1
            max_block = -1
            # Sample a few to save time? Or just first?
            # Let's pick random to vary
            # But for B&B, we need a good initial bound
            # Use deterministic greedy for bound
            for v in range(V):
                if (avail >> v) & 1:
                    # Check independence
                    indep = True
                    for u_idx, u in enumerate(S):
                        if conflict[u][v] & S_mask:
                            # v conflicts with some pair in S?
                            # conflict[u][v] is set of w.
                            # We need to check if v forms iso with any pair in S.
                            # This check is O(|S|^2) if done naively.
                            # Using conf_pairs[v]:
                            pass
                    # Simplified: just pick first available for now to get a bound
                    break
            
            # Proper Greedy:
            S = []
            S_mask = 0
            blocked = 0
            avail = full_mask
            for v in range(V):
                if (avail >> v) & 1:
                    # Check if v is independent with S
                    ok = True
                    for p, q in conf_pairs[v]:
                        if (S_mask >> p) & 1 and (S_mask >> q) & 1:
                            ok = False
                            break
                    if ok:
                        S.append(v)
                        S_mask |= (1 << v)
                        # Update blocked
                        new_blocked = 0
                        for u in S[:-1]: # pairs with previous
                            new_blocked |= conflict[u][v]
                        blocked |= new_blocked
                        avail &= ~blocked
                        avail &= ~(1 << v)
                        if avail == 0:
                            break
            return S

    best_S = greedy_maximal()
    best_size = len(best_S)
    
    # B&B
    # search(S, S_mask, blocked, avail, k, excluded_mask)
    # We search for size k.
    
    def is_independent(S_mask, v):
        for p, q in conf_pairs[v]:
            if (S_mask >> p) & 1 and (S_mask >> q) & 1:
                return False
        return True

    def search(S, S_mask, blocked, avail, k, excluded_mask):
        nonlocal best_S, best_size
        
        if len(S) > k:
            return False
            
        if len(S) == k:
            # Check maximality
            # All vertices not in S must be blocked
            # Vertices not in S: full_mask & ~S_mask
            # They must be in blocked OR excluded?
            # No, excluded vertices are not in S. They must be blocked.
            # So (full_mask & ~S_mask) must be subset of blocked.
            if (full_mask & ~S_mask) & ~blocked == 0:
                # Valid maximal set
                if len(S) < best_size:
                    best_size = len(S)
                    best_S = list(S)
                return True
            return False

        # Pruning: if we have enough vertices to reach k, but even if we add all, we can't block?
        # Simple prune: if len(S) >= best_size: return False
        
        if avail == 0:
            # No more vertices to add
            # Check if current S is maximal
            if (full_mask & ~S_mask) & ~blocked == 0:
                if len(S) < best_size:
                    best_size = len(S)
                    best_S = list(S)
                return True
            return False

        # Pick a vertex v from avail
        # Heuristic: pick v that blocks most
        # For speed, pick first
        v_bit = avail & -avail
        v = v_bit.bit_length() - 1
        
        # Branch 1: Include v
        if is_independent(S_mask, v):
            new_S = S + [v]
            new_S_mask = S_mask | (1 << v)
            new_blocked = blocked
            for u in S:
                new_blocked |= conflict[u][v]
            new_avail = avail & ~new_blocked & ~(1 << v)
            if search(new_S, new_S_mask, new_blocked, new_avail, k, excluded_mask):
                return True
        
        # Branch 2: Exclude v
        # v must be blocked later
        new_excluded = excluded_mask | (1 << v)
        new_avail = avail & ~(1 << v)
        if search(S, S_mask, blocked, new_avail, k, new_excluded):
            return True
            
        return False

    # Iterate k
    for k in range(1, best_size + 1):
        if search([], 0, 0, full_mask, k, 0):
            break
            
    return [points[i] for i in best_S]
# EVOLVE-BLOCK-END
