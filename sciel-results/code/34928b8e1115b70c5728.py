"""No-isosceles set construction with incremental SA and vectorized validity."""
import numpy as np
import time
import random

N = 64
TIME_LIMIT = 275.0

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Precompute all grid points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    all_pts_arr = np.array(all_points, dtype=np.int64)
    total_points = len(all_points)
    
    # Precompute squared distances between all pairs of grid points
    # This is N*N x N*N which is 4096x4096 = 16M entries - feasible
    # Actually let's be smarter: compute on the fly but cache for active sets
    
    # Vectorized validity check for adding point p to set represented by indices
    def can_add(point_idx, active_indices):
        """Check if adding point_idx to active set maintains no-isosceles property."""
        if len(active_indices) == 0:
            return True
        if len(active_indices) == 1:
            return True
        
        p = all_pts_arr[point_idx]
        active = all_pts_arr[active_indices]
        n = len(active)
        
        # Distances from p to all active points
        d_p = (active[:, 0] - p[0])**2 + (active[:, 1] - p[1])**2
        
        # Check 1: no two distances from p are equal
        if len(set(d_p)) != n:
            return False
        
        # Distances between active points
        P_norm_sq = np.sum(active**2, axis=1)
        Dot = active @ active.T
        D_sq = P_norm_sq[:, np.newaxis] + P_norm_sq[np.newaxis, :] - 2 * Dot
        
        # Check: for each pair (i,j) in active, d_p[i] != D_sq[i,j] and d_p[j] != D_sq[i,j]
        # Vectorized: create matrix of D_sq[i,j] for all i<j, compare with d_p[i] and d_p[j]
        for i in range(n - 1):
            js = np.arange(i + 1, n)
            if np.any(D_sq[i, js] == d_p[i]):
                return False
            if np.any(D_sq[i, js] == d_p[js]):
                return False
        
        return True

    # Fast vectorized violation count for a set of point indices
    def count_violations(indices):
        """Count isosceles triangles in the set."""
        n = len(indices)
        if n < 3:
            return 0
        
        P = all_pts_arr[indices]
        P_norm_sq = np.sum(P**2, axis=1)
        Dot = P @ P.T
        D_sq = P_norm_sq[:, np.newaxis] + P_norm_sq[np.newaxis, :] - 2 * Dot
        
        violations = 0
        for i in range(n):
            for j in range(i + 1, n):
                dij = D_sq[i, j]
                # Count k > j where D_sq[i,k]==dij or D_sq[j,k]==dij
                # Also need to check D_sq[i,k]==D_sq[j,k]
                for k in range(j + 1, n):
                    dik = D_sq[i, k]
                    djk = D_sq[j, k]
                    if dij == dik or dij == djk or dik == djk:
                        violations += 1
        return violations

    # Incremental violation tracking for SA
    # State: set of active point indices, and for each active point, 
    # the squared distances to all other active points
    class IncrementalState:
        def __init__(self):
            self.active = []  # list of point indices
            self.active_set = set()
            self.violations = 0
            
        def add(self, point_idx):
            """Add a point and update violation count incrementally."""
            if point_idx in self.active_set:
                return False
            
            p = all_pts_arr[point_idx]
            n = len(self.active)
            
            # Compute distances from p to all active points
            if n == 0:
                self.active.append(point_idx)
                self.active_set.add(point_idx)
                return True
            
            active_pts = all_pts_arr[self.active]
            d_p = (active_pts[:, 0] - p[0])**2 + (active_pts[:, 1] - p[1])**2
            
            # Check for new isosceles triangles involving p
            new_violations = 0
            
            # Type 1: d(p, a) == d(p, b) for some a, b in active
            # Count pairs with equal distance
            dist_counts = {}
            for i, d in enumerate(d_p):
                dist_counts[d] = dist_counts.get(d, 0) + 1
            for count in dist_counts.values():
                if count >= 2:
                    new_violations += count * (count - 1) // 2
            
            # Type 2: d(p, a) == d(a, b) for some a, b in active
            # Type 3: d(p, b) == d(a, b) for some a, b in active
            # These are equivalent: for each pair (a,b) in active, 
            # check if d_p[a] == D(a,b) or d_p[b] == D(a,b)
            
            P_norm_sq = np.sum(active_pts**2, axis=1)
            Dot = active_pts @ active_pts.T
            D_sq = P_norm_sq[:, np.newaxis] + P_norm_sq[np.newaxis, :] - 2 * Dot
            
            for i in range(n - 1):
                js = np.arange(i + 1, n)
                # d_p[i] == D_sq[i, j] for some j
                matches_i = np.sum(D_sq[i, js] == d_p[i])
                # d_p[j] == D_sq[i, j] for some j
                matches_j = np.sum(D_sq[i, js] == d_p[js])
                # But careful: if d_p[i] == D_sq[i,j] AND d_p[j] == D_sq[i,j], 
                # that's still one triangle (i,j,p) that's isosceles, counted once
                # Actually each pair (i,j) can create at most 1 new isosceles triangle with p
                # The triangle (i,j,p) is isosceles if any of the three distances are equal
                # We already counted d_p[i]==d_p[j] in Type 1
                # Now check d_p[i]==D_sq[i,j] or d_p[j]==D_sq[i,j]
                # But if both, it's still one triangle
                for j in js:
                    if d_p[i] == D_sq[i, j] or d_p[j] == D_sq[i, j]:
                        new_violations += 1
            
            self.violations += new_violations
            self.active.append(point_idx)
            self.active_set.add(point_idx)
            return True
        
        def remove(self, point_idx):
            """Remove a point and update violation count incrementally."""
            if point_idx not in self.active_set:
                return False
            
            pos = self.active.index(point_idx)
            n = len(self.active)
            
            # Compute distances from p to all other active points
            other_indices = [self.active[i] for i in range(n) if i != pos]
            if len(other_indices) == 0:
                self.active.pop(pos)
                self.active_set.discard(point_idx)
                return True
            
            p = all_pts_arr[point_idx]
            other_pts = all_pts_arr[other_indices]
            m = len(other_pts)
            d_p = (other_pts[:, 0] - p[0])**2 + (other_pts[:, 1] - p[1])**2
            
            # Count isosceles triangles involving p that will be removed
            removed_violations = 0
            
            # Type 1: d(p, a) == d(p, b)
            dist_counts = {}
            for d in d_p:
                dist_counts[d] = dist_counts.get(d, 0) + 1
            for count in dist_counts.values():
                if count >= 2:
                    removed_violations += count * (count - 1) // 2
            
            # Type 2/3: d(p, a) == d(a, b) or d(p, b) == d(a, b)
            P_norm_sq = np.sum(other_pts**2, axis=1)
            Dot = other_pts @ other_pts.T
            D_sq = P_norm_sq[:, np.newaxis] + P_norm_sq[np.newaxis, :] - 2 * Dot
            
            for i in range(m - 1):
                for j in range(i + 1, m):
                    if d_p[i] == D_sq[i, j] or d_p[j] == D_sq[i, j]:
                        removed_violations += 1
            
            self.violations -= removed_violations
            self.active.pop(pos)
            self.active_set.discard(point_idx)
            return True

    # Simulated annealing with incremental violation tracking
    def sa_incremental(start_size, max_iters=100000):
        """SA with incremental violation updates."""
        random.shuffle(all_points)
        
        state = IncrementalState()
        # Start with random points
        for _ in range(start_size):
            state.add(random.randint(0, total_points - 1))
        
        best_state = state.active[:]
        best_violations = state.violations
        best_count = len(state.active)
        
        T = 5.0
        cooling_rate = 0.99995
        min_T = 0.001
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if T < min_T:
                break
            
            # Choose a move: add, remove, or swap
            r = random.random()
            if r < 0.33:
                # Add a random point
                point_idx = random.randint(0, total_points - 1)
                if point_idx in state.active_set:
                    continue
                old_viol = state.violations
                old_count = len(state.active)
                state.add(point_idx)
                new_viol = state.violations
                new_count = len(state.active)
                
                # Score: minimize violations, maximize count
                old_score = old_viol - 0.5 * old_count
                new_score = new_viol - 0.5 * new_count
                delta = new_score - old_score
                
                if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                    if state.violations == 0 and len(state.active) > best_count:
                        best_state = state.active[:]
                        best_count = len(state.active)
                        best_violations = 0
                    elif state.violations < best_violations:
                        best_state = state.active[:]
                        best_count = len(state.active)
                        best_violations = state.violations
                else:
                    # Reject: remove the added point
                    state.remove(point_idx)
                    
            elif r < 0.66 and len(state.active) > 1:
                # Remove a random point
                point_idx = random.choice(state.active)
                old_viol = state.violations
                old_count = len(state.active)
                state.remove(point_idx)
                new_viol = state.violations
                new_count = len(state.active)
                
                old_score = old_viol - 0.5 * old_count
                new_score = new_viol - 0.5 * new_count
                delta = new_score - old_score
                
                if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                    if state.violations == 0 and len(state.active) > best_count:
                        best_state = state.active[:]
                        best_count = len(state.active)
                        best_violations = 0
                    elif state.violations < best_violations:
                        best_state = state.active[:]
                        best_count = len(state.active)
                        best_violations = state.violations
                else:
                    # Reject: re-add the point
                    state.add(point_idx)
                    
            else:
                # Swap: remove one, add another
                if len(state.active) == 0:
                    continue
                remove_idx = random.choice(state.active)
                add_idx = random.randint(0, total_points - 1)
                if add_idx in state.active_set and add_idx != remove_idx:
                    continue
                
                old_viol = state.violations
                old_count = len(state.active)
                state.remove(remove_idx)
                state.add(add_idx)
                new_viol = state.violations
                new_count = len(state.active)
                
                old_score = old_viol - 0.5 * old_count
                new_score = new_viol - 0.5 * new_count
                delta = new_score - old_score
                
                if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                    if state.violations == 0 and len(state.active) > best_count:
                        best_state = state.active[:]
                        best_count = len(state.active)
                        best_violations = 0
                    elif state.violations < best_violations:
                        best_state = state.active[:]
                        best_count = len(state.active)
                        best_violations = state.violations
                else:
                    # Reject: undo the swap
                    state.remove(add_idx)
                    state.add(remove_idx)
            
            T *= cooling_rate
        
        return best_state, best_count

    # Greedy construction with vectorized check
    def greedy_construct():
        pts = []
        pts_set = set()
        candidates = list(range(total_points))
        random.shuffle(candidates)
        
        for point_idx in candidates:
            if time.time() > deadline:
                break
            if point_idx in pts_set:
                continue
            if can_add(point_idx, pts):
                pts.append(point_idx)
                pts_set.add(point_idx)
        
        return pts

    # Local search: try to add valid points
    def local_search_add(indices):
        """Try to add any valid point to the set."""
        pts = list(indices)
        pts_set = set(pts)
        improved = True
        
        while improved and time.time() < deadline - 10:
            improved = False
            for point_idx in range(total_points):
                if point_idx in pts_set:
                    continue
                if can_add(point_idx, pts):
                    pts.append(point_idx)
                    pts_set.add(point_idx)
                    improved = True
                    break
        
        return pts

    # Phase 1: Multiple greedy constructions
    best_points = []
    best_count = 0
    
    for trial in range(30):
        if time.time() > deadline - 120:
            break
        
        pts = greedy_construct()
        if len(pts) > best_count:
            best_points = pts
            best_count = len(pts)
        
        # Try local search
        if time.time() < deadline - 60:
            improved = local_search_add(pts)
            if len(improved) > best_count:
                best_points = improved
                best_count = len(improved)

    # Phase 2: SA with incremental tracking
    for trial in range(15):
        if time.time() > deadline - 30:
            break
        
        start_size = max(5, min(best_count + random.randint(-5, 10), total_points))
        sa_pts, sa_count = sa_incremental(start_size, max_iters=50000)
        
        if sa_count > best_count:
            best_points = sa_pts
            best_count = sa_count
        
        # Polish with local search
        if time.time() < deadline - 15:
            improved = local_search_add(sa_pts)
            if len(improved) > best_count:
                best_points = improved
                best_count = len(improved)

    # Final polish
    if best_points and time.time() < deadline - 5:
        final_pts = local_search_add(best_points)
        if len(final_pts) > best_count:
            best_points = final_pts
            best_count = len(final_pts)

    # Convert indices back to (x, y) tuples
    result = [all_points[i] for i in best_points]
    
    # Final verification
    if len(result) >= 3:
        P = np.array(result, dtype=np.int64)
        P_norm_sq = np.sum(P**2, axis=1)
        Dot = P @ P.T
        D_sq = P_norm_sq[:, np.newaxis] + P_norm_sq[np.newaxis, :] - 2 * Dot
        n = len(P)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if D_sq[i,j] == D_sq[i,k] or D_sq[i,j] == D_sq[j,k] or D_sq[i,k] == D_sq[j,k]:
                        # Invalid - should not happen, but return empty if it does
                        return []
    
    return result

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
