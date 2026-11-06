"""
Strategy: Set Cover / Hitting Set Formulation with Greedy Construction and Local Search.

The key structural change from the previous approach:
1. Reformulate the problem: A set S is maximal if and only if for every point p not in S, 
   there exists a pair (a, b) in S such that (p, a, b) is isosceles.
   Equivalently, S must "hit" all potential extensions.
   
   Another view: Let U be the set of all valid non-isosceles pairs {a, b} in the grid.
   A set S is valid (no isosceles) if S contains no triangle.
   S is maximal if for every p not in S, there is no valid pair (a, b) in S such that 
   adding p doesn't create an isosceles triangle... wait.
   
   Let's stick to the definition:
   1. No isosceles triangles in S.
   2. For every p not in S, S ∪ {p} contains an isosceles triangle.
   
   This means for every p not in S, there exists a, b in S such that {a, b, p} is isosceles.
   
   We can view this as: We want to select a set S of points such that:
   - S is an independent set in the 3-uniform hypergraph of isosceles triangles.
   - S is a vertex cover of the 3-uniform hypergraph of isosceles triangles with respect to the 
     complement? No.
   
   Let's use the "Covering" perspective:
   For each point p, let C_p be the set of pairs {a, b} such that {a, b, p} is isosceles.
   S is maximal iff for every p not in S, S contains at least one pair from C_p.
   
   This is a complex constraint. However, we can use a greedy approach to build S:
   1. Start with S = empty.
   2. We want to minimize |S|.
   
   Alternative Strategy: 
   The inspirations suggest that a simple greedy or local search might work well.
   The current program uses a random greedy + local search.
   
   Let's try a different local search: 
   Instead of random swaps, use a "remove if possible" loop repeatedly.
   1. Start with a valid maximal set (e.g., greedy).
   2. Repeat until no change:
      - For each point p in S:
        - Check if S \ {p} is still maximal.
        - If yes, remove p.
   3. This finds a local minimum for the "remove" move.
   
   To escape local minima, we can use "swaps":
   - Try to replace p with q where q is not in S, such that S \ {p} ∪ {q} is valid and maximal, 
     and ideally this allows further removals.
   
   Actually, a simpler and often effective strategy for these types of combinatorial problems 
   is a "Minimization by Deletion" with a specific order, followed by a randomized restart 
   with different orders.
   
   Let's implement:
   1. Precompute isosceles relationships.
   2. Greedy construction to get a valid maximal set.
   3. Local search:
      - Iterate through points in S (random order).
      - Try to remove each point if maximality is preserved.
      - If removal fails, try a swap? 
      
   Let's refine the "Maximality Check" for removal:
   S is maximal iff for all p not in S, p is "covered" by S.
   p is covered by S iff there exists a pair {a, b} subset S such that {a, b, p} is isosceles.
   
   When we remove p from S:
   - New S' = S \ {p}.
   - We must check if S' is still maximal.
   - This requires checking for every q not in S', if q is covered by S'.
   - Note: q might be p itself (now outside S'). So we must check if p is covered by S'.
   - Also, other points q might have been covered ONLY by pairs involving p. 
     If so, they are now uncovered, and S' is not maximal.
   
   Efficient Maximality Check for S \ {p}:
   1. Check if p is covered by S \ {p}.
   2. For each q not in S, check if q is covered by S \ {p}.
      - q is covered by S \ {p} if there exists a pair {a, b} subset S \ {p} such that {a, b, q} is isosceles.
      - This is equivalent to: q is covered by S, AND the covering pair does not involve p, 
        OR q is covered by some other pair in S \ {p}.
      
   This check is O(N^2) per removal attempt in the worst case, which is 256 * 256 = 65536 ops.
   With 256 points, and potentially many removals, this is feasible.
"""

import time
import random
import itertools
import sys
import numpy as np

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
POINT_SET = set(POINTS)
N_POINTS = len(POINTS)
DEADLINE = 310.0  # Safety margin before 330s kill

# Precompute isosceles relationships
# iso_pairs_with[p] = set of frozenset({a,b}) such that {a,b,p} is isosceles
# iso_points_for_pair[frozenset({a,b})] = set of p such that {a,b,p} is isosceles

print("Precomputing isosceles relationships...", file=sys.stderr)
start_time = time.time()

pts_arr = np.array(POINTS, dtype=np.int32)
dist2_matrix = np.zeros((N_POINTS, N_POINTS), dtype=np.int64)
dx = pts_arr[:, 0:1] - pts_arr[:, 0:1].T
dy = pts_arr[:, 1:2] - pts_arr[:, 1:2].T
dist2_matrix = (dx**2 + dy**2).astype(np.int64)

iso_pairs_with = [set() for _ in range(N_POINTS)]
iso_points_for_pair = {}

for i in range(N_POINTS):
    for j in range(i+1, N_POINTS):
        d_ij = dist2_matrix[i, j]
        if d_ij == 0:
            continue
        di = dist2_matrix[i, :]
        dj = dist2_matrix[j, :]
        mask = (di == d_ij) | (dj == d_ij) | (di == dj)
        mask[i] = False
        mask[j] = False
        k_indices = np.where(mask)[0]
        if len(k_indices) > 0:
            pair_key = frozenset((i, j))
            iso_points_for_pair[pair_key] = set(k_indices.tolist())
            for k in k_indices:
                iso_pairs_with[k].add(pair_key)

print(f"Precomputation took {time.time() - start_time:.2f}s", file=sys.stderr)

# Convert to lists for faster iteration
iso_pairs_with_list = [list(s) for s in iso_pairs_with]
iso_points_for_pair_list = {k: list(v) for k, v in iso_points_for_pair.items()}

def _is_valid_set(S_indices):
    """Check if set of point indices has no isosceles triangles."""
    S_list = list(S_indices)
    for idx, a in enumerate(S_list):
        for b in S_list[idx+1:]:
            pair = frozenset((a, b))
            if pair in iso_points_for_pair_list:
                for c in iso_points_for_pair_list[pair]:
                    if c in S_indices:
                        return False
    return True

def _is_maximal(S_indices):
    """Check if set S is maximal."""
    S_set = set(S_indices)
    for p in range(N_POINTS):
        if p in S_set:
            continue
        covered = False
        for pair in iso_pairs_with_list[p]:
            a, b = list(pair)
            if a in S_set and b in S_set:
                covered = True
                break
        if not covered:
            return False
    return True

def _greedy_maximal():
    """Greedy construction of a valid maximal set."""
    S = set()
    pts = list(range(N_POINTS))
    random.shuffle(pts)
    
    for p in pts:
        # Check if adding p creates an isosceles triangle
        valid = True
        for pair in iso_pairs_with_list[p]:
            a, b = list(pair)
            if a in S and b in S:
                valid = False
                break
        if valid:
            S.add(p)
    
    # Make it maximal: for each uncovered point, try to add it
    S_list = list(S)
    for p in range(N_POINTS):
        if p in S:
            continue
        covered = False
        for pair in iso_pairs_with_list[p]:
            a, b = list(pair)
            if a in S and b in S:
                covered = True
                break
        if not covered:
            # p is uncovered. Try to add p if valid.
            valid = True
            for pair in iso_pairs_with_list[p]:
                a, b = list(pair)
                if a in S and b in S:
                    valid = False
                    break
            if valid:
                S.add(p)
                S_list.append(p)
            else:
                # Cannot add p directly. 
                # In a true maximal set, this shouldn't happen if we started with a valid set 
                # and added all valid points? 
                # Actually, if p is not valid to add, it means there is an isosceles triangle 
                # formed by p and two points in S. 
                # But if p is not covered, it means there is NO pair in S that forms an isosceles 
                # triangle with p. Contradiction?
                # Wait. "Covered" means there IS a pair in S that forms an isosceles triangle with p.
                # "Valid to add" means there is NO pair in S that forms an isosceles triangle with p.
                # So if p is not covered, it is valid to add.
                # Thus, the else case should never be triggered if the logic is correct.
                pass
    
    return S

def _try_remove(S, p):
    """
    Try to remove p from S. Return new S if valid and maximal, else None.
    """
    new_S = S - {p}
    
    # 1. Check if p is covered by new_S
    p_covered = False
    for pair in iso_pairs_with_list[p]:
        a, b = list(pair)
        if a in new_S and b in new_S:
            p_covered = True
            break
    if not p_covered:
        return None
    
    # 2. Check if any other point q (not in new_S) is still covered by new_S
    # We only need to check points q that were covered by pairs involving p, 
    # because if q was covered by a pair not involving p, it's still covered.
    # However, checking all q is safer and not too expensive.
    for q in range(N_POINTS):
        if q in new_S:
            continue
        covered = False
        for pair in iso_pairs_with_list[q]:
            a, b = list(pair)
            if a in new_S and b in new_S:
                covered = True
                break
        if not covered:
            return None
            
    return new_S

def construct():
    start_time = time.time()
    
    # Step 1: Get initial valid maximal set
    S = _greedy_maximal()
    
    best_S = set(S)
    best_size = len(best_S)
    
    # Step 2: Local search
    # We will repeatedly try to remove points.
    # To avoid getting stuck, we will also try random swaps if stuck.
    
    while time.time() < DEADLINE:
        improved = False
        # Randomize order of points to try
        points_to_try = list(S)
        random.shuffle(points_to_try)
        
        for p in points_to_try:
            if time.time() > DEADLINE:
                break
            if p not in S:
                continue
                
            new_S = _try_remove(S, p)
            if new_S is not None:
                S = new_S
                improved = True
                if len(S) < best_size:
                    best_S = set(S)
                    best_size = len(best_S)
        
        if not improved:
            # Stuck at a local minimum for removals.
            # Try a random swap to escape?
            # Or just restart with a new greedy?
            # Let's try a few random swaps.
            escaped = False
            for _ in range(50):
                if time.time() > DEADLINE:
                    break
                if len(S) < 2:
                    break
                p = random.choice(list(S))
                # Find a candidate q to swap with
                # q should not be in S
                candidates = [q for q in range(N_POINTS) if q not in S]
                if not candidates:
                    break
                q = random.choice(candidates)
                
                # Check if S' = S \ {p} ∪ {q} is valid
                S_temp = S - {p} | {q}
                valid = True
                for pair in iso_pairs_with_list[q]:
                    a, b = list(pair)
                    if a in S_temp and b in S_temp:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check if S' is maximal
                maximal = True
                for r in range(N_POINTS):
                    if r in S_temp:
                        continue
                    covered = False
                    for pair in iso_pairs_with_list[r]:
                        a, b = list(pair)
                        if a in S_temp and b in S_temp:
                            covered = True
                            break
                    if not covered:
                        maximal = False
                        break
                
                if maximal:
                    # Swap accepted
                    S = S_temp
                    escaped = True
                    # Try to remove again immediately
                    # (The loop will restart)
                    break
            
            if not escaped:
                # Try a new greedy restart
                S_new = _greedy_maximal()
                if len(S_new) < len(S):
                    S = S_new
                    if len(S) < best_size:
                        best_S = set(S)
                        best_size = len(best_S)
                else:
                    # If no improvement after several restarts, break?
                    # For now, just keep trying, but maybe add a counter.
                    pass
                    
    # Final verification
    if not _is_valid_set(best_S) or not _is_maximal(best_S):
        # Fallback to greedy
        best_S = _greedy_maximal()
    
    return [POINTS[p] for p in best_S]

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    print(f"Points: {res[:10]}...")
