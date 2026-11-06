import time
import random

N = 16

def _dist_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def _is_isosceles(a, b, c):
    d1 = _dist_sq(a, b)
    d2 = _dist_sq(a, c)
    d3 = _dist_sq(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def construct():
    all_points = [(x, y) for x in range(N) for y in range(N)]
    num_points = len(all_points)
    
    # Precompute: for each pair (i, j), which points k form isosceles with i, j
    # iso_conflict[i][j] = set of k such that {i, j, k} is isosceles
    # Also: for each point i, and each existing point a, which points b would form isosceles with i and a
    # Actually, let's precompute: conflict[i][j] = set of k where {i,j,k} is isosceles
    
    # More useful: for a set S, a point q is blocked if exists a,b in S with q in conflict[a][b]
    # Let's precompute conflict as a 2D list of sets
    conflict = [[set() for _ in range(num_points)] for _ in range(num_points)]
    for i in range(num_points):
        for j in range(i + 1, num_points):
            ci = set()
            for k in range(num_points):
                if k == i or k == j:
                    continue
                if _is_isosceles(all_points[i], all_points[j], all_points[k]):
                    ci.add(k)
            conflict[i][j] = ci
            conflict[j][i] = ci
    
    # For fast lookup: given point index p and set S (as list of indices), 
    # check if p is compatible with all pairs in S
    def is_compatible(p_idx, S_indices):
        for i in range(len(S_indices)):
            for j in range(i + 1, len(S_indices)):
                if p_idx in conflict[S_indices[i]][S_indices[j]]:
                    return False
        return True
    
    # Get blocked points given S_indices
    def get_blocked(S_indices):
        blocked = set()
        for i in range(len(S_indices)):
            for j in range(i + 1, len(S_indices)):
                blocked.update(conflict[S_indices[i]][S_indices[j]])
        blocked.difference_update(set(S_indices))
        return blocked
    
    # Greedy build with heuristic
    def greedy_build(seed=None):
        if seed is not None:
            random.seed(seed)
        
        S_indices = []
        remaining = set(range(num_points))
        
        while remaining:
            # Find valid candidates
            valid = [p for p in remaining if is_compatible(p, S_indices)]
            
            if not valid:
                break
            
            # Heuristic: pick point that blocks the most remaining points
            # Blocking degree of v = |{u in remaining \ {v} : u is blocked by adding v to S}|
            # u is blocked by v if exists a in S such that u in conflict[v][a]
            
            best_v = None
            best_deg = -1
            best_ties = []
            
            for v in valid:
                deg = 0
                # Count how many remaining points would be blocked if we add v
                # A point u is blocked if u in conflict[v][a] for some a in S
                # OR if there exists b in S such that u in conflict[v][b]
                # Actually, u is blocked by v given S if exists a in S such that u in conflict[v][a]
                # But also, if S is empty, v doesn't block anyone (no pairs yet)
                
                if not S_indices:
                    # No pairs yet, so v doesn't block anyone directly
                    # But v will form pairs with future points
                    # Use a different heuristic: pick point with max potential
                    # Count how many points u such that {v, u, ?} could be isosceles
                    # Simpler: just pick randomly or by some static measure
                    deg = 0
                else:
                    for u in remaining:
                        if u == v:
                            continue
                        for a in S_indices:
                            if u in conflict[v][a]:
                                deg += 1
                                break
                
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                    best_ties = [v]
                elif deg == best_deg:
                    best_ties.append(v)
            
            chosen = random.choice(best_ties)
            S_indices.append(chosen)
            remaining.remove(chosen)
            
            # Update remaining: remove blocked points
            blocked = get_blocked(S_indices)
            remaining -= blocked
        
        return S_indices
    
    # Local search: try to remove a point from S and see if we can rebuild smaller
    def local_search(S_indices, max_time=5.0):
        S = list(S_indices)
        start = time.time()
        improved = True
        
        while improved and (time.time() - start) < max_time:
            improved = False
            current_size = len(S)
            
            # Try removing each point and rebuilding
            for i in range(len(S)):
                if time.time() - start > max_time:
                    break
                
                # Remove S[i]
                S_without_i = S[:i] + S[i+1:]
                
                # Try to greedily complete from S_without_i
                remaining = set(range(num_points)) - set(S_without_i)
                # Remove blocked
                blocked = get_blocked(S_without_i)
                remaining -= blocked
                
                new_S = list(S_without_i)
                
                # Greedily add points
                while remaining:
                    valid = [p for p in remaining if is_compatible(p, new_S)]
                    if not valid:
                        break
                    
                    # Pick point with max blocking degree
                    best_v = None
                    best_deg = -1
                    best_ties = []
                    
                    for v in valid:
                        deg = 0
                        if new_S:
                            for u in remaining:
                                if u == v:
                                    continue
                                for a in new_S:
                                    if u in conflict[v][a]:
                                        deg += 1
                                        break
                        
                        if deg > best_deg:
                            best_deg = deg
                            best_v = v
                            best_ties = [v]
                        elif deg == best_deg:
                            best_ties.append(v)
                    
                    chosen = random.choice(best_ties)
                    new_S.append(chosen)
                    remaining.remove(chosen)
                    blocked = get_blocked(new_S)
                    remaining -= blocked
                
                if len(new_S) < current_size:
                    S = new_S
                    improved = True
                    break
        
        return S
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s margin
    
    best_set = None
    best_size = float('inf')
    
    # Multiple restarts with different seeds
    seed_counter = 0
    while time.time() < deadline:
        # Try greedy build
        S_indices = greedy_build(seed=seed_counter)
        seed_counter += 1
        
        # Try local search to improve
        if time.time() < deadline - 10:  # Need time for local search
            S_improved = local_search(S_indices, max_time=min(5.0, max(1.0, deadline - time.time() - 5)))
            if len(S_improved) < len(S_indices):
                S_indices = S_improved
        
        if len(S_indices) < best_size:
            best_size = len(S_indices)
            best_set = [all_points[i] for i in S_indices]
    
    return best_set
