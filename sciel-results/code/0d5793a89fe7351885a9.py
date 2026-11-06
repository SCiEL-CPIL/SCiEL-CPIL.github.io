import time
import random
import itertools
from collections import defaultdict

N = 16
GRID_SIZE = N * N

def _dist_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def _is_isosceles(a, b, c):
    d1 = _dist_sq(a, b)
    d2 = _dist_sq(a, c)
    d3 = _dist_sq(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _precompute_conflicts():
    """
    Precompute for each pair of points, which other points form isosceles with them.
    conflict[i][j] is a set of indices k such that {i, j, k} is isosceles.
    """
    all_points = [(x, y) for x in range(N) for y in range(N)]
    n = len(all_points)
    
    # conflict[i] is a dict: j -> set of k
    conflict = [defaultdict(set) for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            blocked = set()
            for k in range(n):
                if k == i or k == j:
                    continue
                if _is_isosceles(all_points[k], all_points[i], all_points[j]):
                    blocked.add(k)
            conflict[i][j] = blocked
            conflict[j][i] = blocked
            
    return all_points, conflict

def construct():
    all_points, conflict = _precompute_conflicts()
    n = len(all_points)
    
    start_time = time.time()
    deadline = start_time + 280
    
    best_set = None
    best_size = float('inf')
    
    # Helper to check if a set S (list of indices) is valid no-isosceles
    def is_valid(S):
        S_set = set(S)
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                # Check if any k in S is in conflict[S[i]][S[j]]
                # We just need to check if intersection is non-empty
                # But we can just check specific pairs
                # Actually, if S is built incrementally, we only check new point.
                # For general check:
                pass
        # Efficient check:
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                # If any point in S is in conflict[i][j], it's invalid
                # We can iterate S and check membership
                for k in S:
                    if k == i or k == j:
                        continue
                    if k in conflict[i][j]:
                        return False
        return True

    # Helper to compute C_p for p not in S
    # C[p] = count of pairs {a, b} in S such that p is in conflict[a][b]
    def compute_block_counts(S):
        S_set = set(S)
        C = [0] * n
        for p in range(n):
            if p in S_set:
                continue
            count = 0
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    if p in conflict[S[i]][S[j]]:
                        count += 1
            C[p] = count
        return C

    # Check if S is maximal using C
    # S is maximal if C[p] > 0 for all p not in S
    def is_maximal(S, C):
        S_set = set(S)
        for p in range(n):
            if p in S_set:
                continue
            if C[p] == 0:
                return False
        return True

    # Try to remove redundant points from S
    # Returns a new set S' which is minimal maximal
    def reduce_to_minimal_maximal(S):
        S = list(S)
        C = compute_block_counts(S)
        
        changed = True
        while changed:
            changed = False
            for i in range(len(S)):
                x = S[i]
                # Check if removing x keeps S maximal
                # 1. Check if x is blocked by S \ {x}
                # 2. Check if all p not in S are still blocked by S \ {x}
                
                # Precompute pairs involving x
                # For each p not in S, how many pairs {x, a} in S block p?
                # Let k_p = count of a in S\{x} such that p in conflict[x][a]
                
                # Also check if x is blocked by S\{x}
                # x is blocked if there exists pair {a, b} in S\{x} such that x in conflict[a][b]
                
                S_without_x = [s for s in S if s != x]
                
                # Check if x is blocked by S_without_x
                x_blocked = False
                for i_idx in range(len(S_without_x)):
                    for j_idx in range(i_idx + 1, len(S_without_x)):
                        if x in conflict[S_without_x[i_idx]][S_without_x[j_idx]]:
                            x_blocked = True
                            break
                    if x_blocked:
                        break
                
                if not x_blocked:
                    continue # Cannot remove x, it would not be maximal
                
                # Check if all p not in S are still blocked
                valid_removal = True
                for p in range(n):
                    if p in S:
                        continue
                    # Current C[p] is count of pairs in S blocking p
                    # Pairs in S that block p and involve x:
                    # {x, a} where a in S\{x} and p in conflict[x][a]
                    k_p = 0
                    for a in S_without_x:
                        if p in conflict[x][a]:
                            k_p += 1
                    new_C_p = C[p] - k_p
                    if new_C_p == 0:
                        valid_removal = False
                        break
                
                if valid_removal:
                    # Remove x
                    S = S_without_x
                    # Update C
                    C = compute_block_counts(S)
                    changed = True
                    break
        
        return S

    # Greedy construction
    def greedy_build():
        S = []
        remaining = set(range(n))
        
        while remaining:
            # Find valid candidates
            valid = []
            for pi in remaining:
                compatible = True
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        if pi in conflict[S[i]][S[j]]:
                            compatible = False
                            break
                    if not compatible:
                        break
                if compatible:
                    valid.append(pi)
            
            if not valid:
                break
            
            # Heuristic: pick point with max blocking degree
            # Blocking degree: how many other points in 'remaining' it blocks
            best_v = None
            best_deg = -1
            best_ties = []
            
            for v in valid:
                deg = 0
                for u in remaining:
                    if u == v:
                        continue
                    # Check if u is blocked by v given S
                    # u is blocked by v if there exists a in S such that u in conflict[v][a]
                    # OR if u in conflict[v][b] for some b? No, pair is {v, a}
                    # So we check if u is in conflict[v][a] for any a in S
                    is_blocked_by_v = False
                    for a in S:
                        if u in conflict[v][a]:
                            is_blocked_by_v = True
                            break
                    if is_blocked_by_v:
                        deg += 1
                
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                    best_ties = [v]
                elif deg == best_deg:
                    best_ties.append(v)
            
            chosen = random.choice(best_ties)
            S.append(chosen)
            remaining.remove(chosen)
            
            # Update remaining
            new_remaining = set()
            for q in remaining:
                compatible = True
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        if q in conflict[S[i]][S[j]]:
                            compatible = False
                            break
                    if not compatible:
                        break
                if compatible:
                    new_remaining.add(q)
            remaining = new_remaining
        
        return S

    # Local Search / ILS
    def local_search(S):
        S = list(S)
        C = compute_block_counts(S)
        
        start_time = time.time()
        limit = start_time + 10 # 10 seconds per restart
        
        while time.time() < limit:
            improved = False
            # Try swap moves
            # Pick random x in S
            if not S:
                break
            x = random.choice(S)
            
            # Pick random y not in S
            S_set = set(S)
            candidates = [p for p in range(n) if p not in S_set]
            if not candidates:
                break
            y = random.choice(candidates)
            
            # Check if S' = S \ {x} + {y} is valid
            S_without_x = [s for s in S if s != x]
            
            valid = True
            for i in range(len(S_without_x)):
                for j in range(i + 1, len(S_without_x)):
                    if y in conflict[S_without_x[i]][S_without_x[j]]:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # If valid, check if it's better or equal
            # We don't know if it's maximal yet, but we can try to reduce it
            # For now, just accept if size is smaller or same (with some probability)
            # Actually, we want to minimize size.
            # If we swap, size is same.
            # We need to check if it allows further reduction.
            
            # Let's just do the swap and then reduce
            S_new = S_without_x + [y]
            
            # Check if S_new is valid (already done)
            # Reduce S_new
            S_reduced = reduce_to_minimal_maximal(S_new)
            
            if len(S_reduced) < len(S):
                S = S_reduced
                C = compute_block_counts(S)
                improved = True
                if len(S) <= 10: # Early exit
                    break
            else:
                # If size is same, we might still accept to escape local minima
                # But we need to be careful not to cycle
                # Let's accept with probability 0.1
                if random.random() < 0.1:
                    S = S_reduced
                    C = compute_block_counts(S)
        
        return S

    # Run multiple restarts
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        S = greedy_build()
        S = reduce_to_minimal_maximal(S)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S.copy()
        
        # Local search
        remaining_time = deadline - time.time()
        if remaining_time > 15:
            S = local_search(S)
            S = reduce_to_minimal_maximal(S)
            
            if len(S) < best_size:
                best_size = len(S)
                best_set = S.copy()
        
        if best_size <= 10:
            break

    if best_set is None:
        best_set = list(range(n))
    
    return [all_points[i] for i in best_set]
