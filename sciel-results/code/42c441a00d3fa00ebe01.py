import time
import random
import numpy as np

N = 16
TOTAL = N * N
ALL_MASK = (1 << TOTAL) - 1

def construct():
    start = time.time()
    deadline = start + 280  # Safety margin for 300s limit

    # Precompute distances squared between all pairs
    coords = [(x, y) for x in range(N) for y in range(N)]
    
    # dist2[i][j] = squared distance between point i and point j
    dist2 = [[0] * TOTAL for _ in range(TOTAL)]
    for i in range(TOTAL):
        xi, yi = coords[i]
        for j in range(i + 1, TOTAL):
            xj, yj = coords[j]
            d = (xi - xj) ** 2 + (yi - yj) ** 2
            dist2[i][j] = d
            dist2[j][i] = d

    # For each point p, compute the set of points that are at the same distance from p
    # SameDist[p] is a list of sets: each set contains points q where dist2[p][q] = dist2[p][r] for all r in the set
    # Actually, we need: for a pair (a, b) in S, a point c is forbidden if dist(a,c)==dist(a,b) or dist(b,c)==dist(a,b) or dist(a,c)==dist(b,c)
    
    # Let's precompute for each point p, a dictionary mapping distance to bitmask of points at that distance
    # dist_map[p][d] = bitmask of points q (q != p) such that dist2[p][q] == d
    dist_map = [{} for _ in range(TOTAL)]
    for p in range(TOTAL):
        dm = {}
        for q in range(TOTAL):
            if q == p:
                continue
            d = dist2[p][q]
            if d not in dm:
                dm[d] = 0
            dm[d] |= (1 << q)
        dist_map[p] = dm

    # Precompute cover_mask[a][b]: bitmask of points c such that {a,b,c} is isosceles
    # c is forbidden if dist(a,b)==dist(a,c) or dist(a,b)==dist(b,c) or dist(a,c)==dist(b,c)
    # dist(a,b)==dist(a,c) => c is in dist_map[a][dist2[a][b]]
    # dist(a,b)==dist(b,c) => c is in dist_map[b][dist2[a][b]]
    # dist(a,c)==dist(b,c) => c is equidistant from a and b. This is harder to precompute simply.
    
    # Let's compute cover_mask fully.
    # cover_mask[a][b] = (dist_map[a][dist2[a][b]] | dist_map[b][dist2[a][b]]) & ~(1<<a) & ~(1<<b)
    # Plus points c where dist(a,c) == dist(b,c).
    
    # For the third condition, let's precompute equidistant masks.
    # Equi[a][b] = bitmask of points c such that dist2[a][c] == dist2[b][c]
    # This is 256*256 masks, each 256 bits. 256*256*256/8 = 2MB. Fine.
    
    Equi = [[0] * TOTAL for _ in range(TOTAL)]
    for a in range(TOTAL):
        for b in range(a + 1, TOTAL):
            mask = 0
            for c in range(TOTAL):
                if c == a or c == b:
                    continue
                if dist2[a][c] == dist2[b][c]:
                    mask |= (1 << c)
            Equi[a][b] = mask
            Equi[b][a] = mask

    cover_mask = [[0] * TOTAL for _ in range(TOTAL)]
    for a in range(TOTAL):
        for b in range(a + 1, TOTAL):
            d_ab = dist2[a][b]
            m1 = dist_map[a].get(d_ab, 0)
            m2 = dist_map[b].get(d_ab, 0)
            m3 = Equi[a][b]
            mask = (m1 | m2 | m3) & ~(1 << a) & ~(1 << b)
            cover_mask[a][b] = mask
            cover_mask[b][a] = mask

    def _is_maximal(S_mask):
        """Check if S_mask is a maximal no-isosceles set."""
        if S_mask == 0:
            return False
        
        # Check independence: for all pairs a,b in S, no c in S is in cover_mask[a][b]
        # We can check this by iterating over points in S
        S_points = []
        for i in range(TOTAL):
            if (S_mask >> i) & 1:
                S_points.append(i)
        
        for i in range(len(S_points)):
            for j in range(i + 1, len(S_points)):
                a = S_points[i]
                b = S_points[j]
                if cover_mask[a][b] & S_mask:
                    return False
        
        # Check maximality: for every point p not in S, p is covered by some pair in S
        not_in_S = ALL_MASK & ~S_mask
        if not_in_S == 0:
            return True  # S is the whole grid, which is maximal (and independent if grid has no isosceles triples, which it doesn't)
            
        # Compute union of covers of all pairs in S
        total_cover = 0
        for i in range(len(S_points)):
            for j in range(i + 1, len(S_points)):
                a = S_points[i]
                b = S_points[j]
                total_cover |= cover_mask[a][b]
        
        return (total_cover & not_in_S) == not_in_S

    def _greedy_random():
        """Greedy construction with randomization."""
        S = []
        S_mask = 0
        available = list(range(TOTAL))
        random.shuffle(available)
        
        for p in available:
            # Check if p can be added
            compatible = True
            for a in S:
                if cover_mask[a][p] & S_mask:
                    compatible = False
                    break
            if compatible:
                S.append(p)
                S_mask |= (1 << p)
        
        return S_mask

    def _local_search(S_mask):
        """Try to remove points while maintaining maximality."""
        S_mask = S_mask
        improved = True
        while improved:
            improved = False
            S_points = [i for i in range(TOTAL) if (S_mask >> i) & 1]
            random.shuffle(S_points)
            for p in S_points:
                if not ((S_mask >> p) & 1):
                    continue
                S_trial = S_mask & ~(1 << p)
                if _is_maximal(S_trial):
                    S_mask = S_trial
                    improved = True
                    break
        return S_mask

    def _simulated_annealing(S_mask, time_limit=5.0):
        """Simulated annealing to minimize size while maintaining maximality."""
        current = S_mask
        current_size = bin(current).count('1')
        best = current
        best_size = current_size
        T = 10.0
        T_min = 0.01
        t_start = time.time()
        
        while time.time() - t_start < time_limit:
            # Generate a neighbor: remove a random point or swap
            S_points = [i for i in range(TOTAL) if (current >> i) & 1]
            if not S_points:
                break
            
            move_type = random.random()
            if move_type < 0.7:
                # Remove a random point
                p = random.choice(S_points)
                trial = current & ~(1 << p)
                trial_size = current_size - 1
            elif move_type < 0.9:
                # Add a random point (if compatible)
                not_in_S = [i for i in range(TOTAL) if not ((current >> i) & 1)]
                if not not_in_S:
                    continue
                p = random.choice(not_in_S)
                # Check compatibility
                compatible = True
                for a in S_points:
                    if cover_mask[a][p] & current:
                        compatible = False
                        break
                if not compatible:
                    continue
                trial = current | (1 << p)
                trial_size = current_size + 1
            else:
                # Swap: remove one, add one
                if len(S_points) < 2:
                    continue
                p_remove = random.choice(S_points)
                not_in_S = [i for i in range(TOTAL) if not ((current >> i) & 1)]
                if not not_in_S:
                    continue
                p_add = random.choice(not_in_S)
                current_without_p = current & ~(1 << p_remove)
                compatible = True
                S_points_no_p = [a for a in S_points if a != p_remove]
                for a in S_points_no_p:
                    if cover_mask[a][p_add] & current_without_p:
                        compatible = False
                        break
                if not compatible:
                    continue
                trial = current_without_p | (1 << p_add)
                trial_size = current_size
            
            # Check if trial is valid (independent)
            # For efficiency, we only check independence, not maximality, in the inner loop
            # Maximality is checked in _is_maximal
            # Actually, we need the trial to be a valid no-isosceles set (independent)
            # And we want to maintain maximality eventually.
            # Let's check independence first.
            S_trial_points = [i for i in range(TOTAL) if (trial >> i) & 1]
            independent = True
            for i in range(len(S_trial_points)):
                for j in range(i + 1, len(S_trial_points)):
                    a = S_trial_points[i]
                    b = S_trial_points[j]
                    if cover_mask[a][b] & trial:
                        independent = False
                        break
                if not independent:
                    break
            
            if not independent:
                continue
            
            # Acceptance
            delta = trial_size - current_size
            if delta < 0:
                # Improvement
                current = trial
                current_size = trial_size
                if current_size < best_size:
                    best = current
                    best_size = current_size
            else:
                # Accept with probability exp(-delta/T)
                if random.random() < np.exp(-delta / max(T, 1e-9)):
                    current = trial
                    current_size = trial_size
            
            T = max(T * 0.99, T_min)
        
        return best

    best_mask = None
    best_size = TOTAL + 1

    # Initial greedy
    while time.time() < deadline:
        S_mask = _greedy_random()
        S_mask = _local_search(S_mask)
        size = bin(S_mask).count('1')
        
        if size < best_size:
            best_size = size
            best_mask = S_mask
        
        # Try simulated annealing on the best so far
        if time.time() < deadline - 10:
            improved_mask = _simulated_annealing(best_mask, time_limit=min(5.0, deadline - time.time() - 5))
            improved_size = bin(improved_mask).count('1')
            if improved_size < best_size:
                best_size = improved_size
                best_mask = improved_mask
        
        if best_size <= 12:
            # Try to continue improving
            pass
        else:
            # Continue greedy restarts
            continue

    if best_mask is None:
        # Fallback
        best_mask = _greedy_random()

    # Convert mask to list of (x, y)
    result = []
    for i in range(TOTAL):
        if (best_mask >> i) & 1:
            result.append(coords[i])
    
    return result
