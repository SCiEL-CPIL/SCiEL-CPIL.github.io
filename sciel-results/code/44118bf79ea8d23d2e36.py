"""No-isosceles set: find max points in [0,N)^2 with no three forming an isosceles triangle."""
import time
import random
import math

N = 64
TIME_LIMIT = 285  # Safety margin below 330s wall-clock limit

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    n_points = len(all_points)
    
    def dist2(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy
    
    # Precompute all pairwise squared distances
    # dist_matrix[i][j] = dist2(all_points[i], all_points[j])
    dist_matrix = [[0] * n_points for _ in range(n_points)]
    for i in range(n_points):
        for j in range(i + 1, n_points):
            d = dist2(all_points[i], all_points[j])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    
    # For each point i, group other points by distance from i
    # dist_groups[i] = dict: distance -> list of point indices at that distance from i
    dist_groups = [{} for _ in range(n_points)]
    for i in range(n_points):
        for j in range(n_points):
            if i == j:
                continue
            d = dist_matrix[i][j]
            if d not in dist_groups[i]:
                dist_groups[i][d] = []
            dist_groups[i][d].append(j)
    
    def fast_can_add(p_idx, S_indices):
        """
        Check if point p_idx can be added to set S_indices.
        Returns (valid, distances_to_S) where distances_to_S[i] = dist2(p, S[i])
        """
        k = len(S_indices)
        if k == 0:
            return True, []
        if k == 1:
            d = dist_matrix[p_idx][S_indices[0]]
            return True, [d]
        
        # Calculate distances from p to all points in S
        distances = [dist_matrix[p_idx][S_indices[i]] for i in range(k)]
        
        # Check 1: Apex at existing point S[i].
        # dist(p, S[i]) must not equal dist(S[i], S[j]) for any j != i
        # i.e., distances[i] must not be in the set of distances from S[i] to other points in S
        for i in range(k):
            d_to_p = distances[i]
            # Check if any other point in S is at distance d_to_p from S[i]
            s_i = S_indices[i]
            if d_to_p in dist_groups[s_i]:
                # There are points at this distance from S[i]; check if any are in S
                group = dist_groups[s_i][d_to_p]
                for g in group:
                    if g != p_idx and g in S_set:
                        return False, []
        
        # Check 2: Apex at p.
        # No two distances from p to S should be equal.
        if len(set(distances)) < k:
            return False, []
            
        return True, distances
    
    def greedy_construct(seed_rng, S_set=None):
        """Fast greedy construction with randomized ordering."""
        if S_set is None:
            S_set = set()
        S_indices = []
        candidates = list(range(n_points))
        seed_rng.shuffle(candidates)
        
        for idx in candidates:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            valid, distances = fast_can_add(idx, S_indices)
            if valid:
                S_indices.append(idx)
                S_set.add(idx)
        
        return S_indices, S_set
    
    def local_improvement(S_indices, S_set, seed_rng, max_iters=2000):
        """Try to add more points via local search."""
        S_indices = list(S_indices)
        S_set = set(S_set)
        remaining = [i for i in range(n_points) if i not in S_set]
        seed_rng.shuffle(remaining)
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for i in range(len(remaining)):
                p_idx = remaining[i]
                valid, distances = fast_can_add(p_idx, S_indices)
                if valid:
                    S_indices.append(p_idx)
                    S_set.add(p_idx)
                    remaining.pop(i)
                    improved = True
                    break
            if time.time() - start_time > TIME_LIMIT - 5:
                break
        
        return S_indices, S_set
    
    def swap_improvement(S_indices, S_set, seed_rng, max_iters=500):
        """Try to improve by swapping points."""
        S_indices = list(S_indices)
        S_set = set(S_set)
        current_len = len(S_indices)
        
        for _ in range(max_iters):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            # Try to add a random point not in the set
            p_add = seed_rng.randint(0, n_points - 1)
            if p_add not in S_set:
                valid, distances = fast_can_add(p_add, S_indices)
                if valid:
                    S_indices.append(p_add)
                    S_set.add(p_add)
                    current_len = len(S_indices)
                    continue
            
            # Try swap: remove a random point, then try to add a different one
            if current_len > 1:
                idx = seed_rng.randint(0, current_len - 1)
                p_remove = S_indices[idx]
                
                # Rebuild without p_remove
                new_S = S_indices[:idx] + S_indices[idx+1:]
                new_S_set = set(new_S)
                
                # Try to add a new point
                p_new = seed_rng.randint(0, n_points - 1)
                if p_new not in new_S_set:
                    valid, distances = fast_can_add(p_new, new_S)
                    if valid:
                        new_S.append(p_new)
                        new_S_set.add(p_new)
                        S_indices = new_S
                        S_set = new_S_set
                        current_len = len(S_indices)
                        continue
        
        return S_indices, S_set
    
    best_S = []
    best_S_set = set()
    best_len = 0
    
    # Phase 1: Many fast randomized greedy restarts
    num_restarts = 300
    for r in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        restart_rng = random.Random(1000 + r * 7919)
        S, S_set = greedy_construct(restart_rng)
        
        if len(S) > best_len:
            best_S = S[:]
            best_S_set = S_set.copy()
            best_len = len(S)
        
        # Local improvement on this solution if it's competitive
        if len(S) >= best_len - 3:
            improved_S, improved_S_set = local_improvement(S, S_set, restart_rng, max_iters=800)
            if len(improved_S) > best_len:
                best_S = improved_S[:]
                best_S_set = improved_S_set.copy()
                best_len = len(best_S)
        
        # Check time
        if time.time() - start_time > TIME_LIMIT - 20:
            break
    
    # Phase 2: Final refinement on best
    if time.time() - start_time < TIME_LIMIT - 10 and best_len > 0:
        final_rng = random.Random(99999)
        improved_S, improved_S_set = local_improvement(best_S, best_S_set, final_rng, max_iters=5000)
        if len(improved_S) > best_len:
            best_S = improved_S[:]
            best_S_set = improved_S_set.copy()
            best_len = len(best_S)
        
        # Swap improvement
        if time.time() - start_time < TIME_LIMIT - 5:
            swap_rng = random.Random(88888)
            swapped_S, swapped_S_set = swap_improvement(best_S, best_S_set, swap_rng, max_iters=2000)
            if len(swapped_S) > best_len:
                best_S = swapped_S[:]
                best_S_set = swapped_S_set.copy()
                best_len = len(best_S)
        
        # One more local improvement after swaps
        if time.time() - start_time < TIME_LIMIT - 3 and best_len > 0:
            final2_rng = random.Random(77777)
            improved2_S, improved2_S_set = local_improvement(best_S, best_S_set, final2_rng, max_iters=3000)
            if len(improved2_S) > best_len:
                best_S = improved2_S[:]
                best_S_set = improved2_S_set.copy()
                best_len = len(best_S)
    
    # Ensure we have at least the baseline
    if not best_S:
        best_S = [0, 1, 64]  # (0,0), (1,0), (0,1)
    
    return [all_points[i] for i in best_S]
# EVOLVE-BLOCK-END
