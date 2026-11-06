import random
import itertools
import numpy as np

N = 16
M = N * N

# Precompute conflict masks
# conflict_mask[i][j] is a bitset of points k such that i,j,k form an isosceles triangle
conflict_mask = [[0] * M for _ in range(M)]

def dist_sq(p, q):
    return (p[0] - q[0])**2 + (p[1] - q[1])**2

# Generate all points
points = [(x, y) for y in range(N) for x in range(N)]
point_to_idx = {p: i for i, p in enumerate(points)}

for i in range(M):
    for j in range(i + 1, M):
        p_i = points[i]
        p_j = points[j]
        d_ij = dist_sq(p_i, p_j)
        mask = 0
        for k in range(M):
            if k == i or k == j:
                continue
            p_k = points[k]
            d_ik = dist_sq(p_i, p_k)
            d_jk = dist_sq(p_j, p_k)
            if d_ik == d_jk or d_ik == d_ij or d_jk == d_ij:
                mask |= (1 << k)
        conflict_mask[i][j] = mask
        conflict_mask[j][i] = mask

def is_valid(S_mask, new_point_idx):
    """Check if adding new_point_idx to S_mask keeps it no-isosceles."""
    s = S_mask
    while s:
        lsb = s & -s
        i = lsb.bit_length() - 1
        s ^= lsb
        if (conflict_mask[i][new_point_idx] >> new_point_idx) & 1:
             # Wait, conflict_mask[i][j] contains k.
             # We need to check if new_point_idx is in conflict_mask[i][j] for any j in S.
             # Or if any j in S is in conflict_mask[i][new_point_idx].
             # Symmetric.
             # Check if S intersects conflict_mask[i][new_point_idx]
             # Note: conflict_mask[i][new_point_idx] might contain i or new_point_idx?
             # Our precomputation excluded i and j. So it contains only k != i, j.
             # So we check if S_mask & conflict_mask[i][new_point_idx] != 0.
             # But we are iterating i from S. So we check if S_mask (excluding i) intersects conflict_mask[i][new_point_idx].
             # Actually, just check S_mask & conflict_mask[i][new_point_idx].
             # If it's non-zero, there is a j in S such that {i, j, new} is isosceles.
             # We must exclude i itself from the check? No, conflict_mask[i][new] doesn't contain i.
             # But it might contain other points in S.
             if S_mask & conflict_mask[i][new_point_idx]:
                 return False
    return True

def is_maximal(S_mask):
    """Check if S_mask is a maximal no-isosceles set."""
    if S_mask == 0:
        return False # Empty set is not maximal unless M=0, but adding any point is valid? 
        # If S is empty, any point can be added. So not maximal.
    
    cover = 0
    s = S_mask
    while s:
        lsb = s & -s
        i = lsb.bit_length() - 1
        s ^= lsb
        s2 = S_mask
        while s2:
            lsb2 = s2 & -s2
            j = lsb2.bit_length() - 1
            s2 ^= lsb2
            if i == j:
                continue
            cover |= conflict_mask[i][j]
            
    not_in_S = (~S_mask) & ((1 << M) - 1)
    uncovered = not_in_S & ~cover
    return uncovered == 0

def greedy_construct(order):
    S_mask = 0
    for idx in order:
        if is_valid(S_mask, idx):
            S_mask |= (1 << idx)
    return S_mask

def bbs_minimize_maximal(S_mask, time_limit=5.0):
    """
    Use Branch and Bound to find a minimal maximal subset of S_mask.
    We search for a subset S' of S_mask such that:
    1. S' is no-isosceles.
    2. S' is maximal (covers all points in U \ S').
    3. |S'| is minimized.
    
    Since S_mask is already maximal, any subset S' that is maximal must cover all points in U \ S'.
    Note: Points in S_mask \ S' are excluded, so they must be covered by S'.
    Points in U \ S_mask are already covered by S_mask, so they are covered by S' if S' is maximal?
    No, if we remove points, we might lose coverage for points in U \ S_mask.
    So we must ensure S' covers (U \ S').
    """
    # Extract indices in S_mask
    s = S_mask
    indices = []
    while s:
        lsb = s & -s
        idx = lsb.bit_length() - 1
        s ^= lsb
        indices.append(idx)
    
    n = len(indices)
    best_size = n
    best_S = S_mask
    
    # Sort indices by some heuristic? 
    # Points that cover more other points in S_mask should be kept?
    # Or points that are "critical" for coverage.
    
    # B&B State:
    # included: list of indices in current S'
    # excluded: list of indices in S_mask not in current S'
    # We need to check if current S' can still be extended to a maximal set.
    
    # This is complex. Let's use a simpler heuristic B&B:
    # Try to remove points one by one, using backtracking if it fails.
    
    # Heuristic: Randomized Local Search with B&B on failures?
    # The current `minimize_maximal` is a local search.
    # I will implement a "Branch and Bound on Removals" where we decide to keep or remove each point.
    
    # To keep it simple and fast, I will use a "Greedy Removal with Backtracking" limited depth.
    
    # Actually, let's just improve the local search with multiple restarts and better ordering.
    # The directive says "Branch-and-bound".
    
    # I will implement a recursive function that tries to remove points.
    
    import time
    start_time = time.time()
    
    def dfs(current_S, remaining_indices, depth):
        nonlocal best_size, best_S
        
        if time.time() - start_time > time_limit:
            return
            
        # Pruning: If current_S is not maximal, we might still be able to add points?
        # No, we are only removing. So if current_S is not maximal, it's a valid candidate 
        # ONLY IF we stop removing and check if it's maximal.
        # Wait, the goal is to find a MAXIMAL subset.
        # So at any leaf (or any node), we check if current_S is maximal.
        # If it is, we update best.
        
        if is_maximal(current_S):
            size = bin(current_S).count('1')
            if size < best_size:
                best_size = size
                best_S = current_S
            # We can prune if size + lower_bound >= best_size
            # Lower bound: 0 (we can remove all? No, must be maximal)
        
        if not remaining_indices:
            return
            
        if time.time() - start_time > time_limit:
            return

        # Heuristic: Pick a point to try removing
        # Try removing the point that is "least useful" for coverage?
        # Or just iterate.
        
        idx = remaining_indices[0]
        rest = remaining_indices[1:]
        
        # Branch 1: Keep idx
        dfs(current_S, rest, depth + 1)
        
        # Branch 2: Remove idx
        new_S = current_S & ~(1 << idx)
        # Check if new_S is still valid (no-isosceles). It is, since subset of valid set.
        # Check if we can still reach a maximal set?
        # Pruning: If new_S cannot cover some point in (U \ new_S), then it's not maximal.
        # But we don't know the final S.
        # However, if new_S is not maximal, it might become maximal if we... wait, we are only removing.
        # So if new_S is not maximal, it will NEVER be maximal in any subset of it.
        # Because removing more points only reduces coverage.
        # So if new_S is not maximal, we can PRUNE this branch.
        
        if is_maximal(new_S):
            dfs(new_S, rest, depth + 1)
        # Else: branch is dead.
        
        # Note: is_maximal is expensive.
        # We can cache or optimize.
        
    dfs(S_mask, indices, 0)
    
    return best_S

def construct():
    best_S = None
    best_size = M + 1
    
    # Run multiple times with different random seeds/orders
    for _ in range(50):
        order = list(range(M))
        random.shuffle(order)
        S = greedy_construct(order)
        
        # Quick minimization
        S = minimize_maximal(S)
        
        # B&B minimization
        S = bbs_minimize_maximal(S, time_limit=2.0)
        
        size = bin(S).count('1')
        if size < best_size:
            best_size = size
            best_S = S
            
    # Convert best_S to list of points
    result = []
    s = best_S
    while s:
        lsb = s & -s
        idx = lsb.bit_length() - 1
        s ^= lsb
        result.append(points[idx])
        
    return result

def minimize_maximal(S_mask):
    """Try to remove points while maintaining maximality."""
    changed = True
    while changed:
        changed = False
        s = S_mask
        indices = []
        while s:
            lsb = s & -s
            idx = lsb.bit_length() - 1
            s ^= lsb
            indices.append(idx)
        
        random.shuffle(indices)
        
        for idx in indices:
            S_new = S_mask & ~(1 << idx)
            if is_maximal(S_new):
                S_mask = S_new
                changed = True
                break
    return S_mask
