import numpy as np
import random
import time

N = 64
NUM_POINTS = N * N
DEADLINE = time.time() + 285  # Safety margin before 300s limit

# Pre-compute coordinates
points = np.array([(x, y) for y in range(N) for x in range(N)], dtype=np.int32)

# Pre-compute full distance matrix (4096x4096)
def _precompute_distances():
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sum(diff**2, axis=2).astype(np.int32)

D_grid = _precompute_distances()

def is_valid_set(S_indices):
    """
    Check if the set of indices S_indices forms a no-isosceles set.
    Returns True if valid.
    """
    if len(S_indices) <= 2:
        return True
    
    K = len(S_indices)
    if K > 200:
        # For very large sets, use a faster check
        D = D_grid[np.ix_(S_indices, S_indices)]
        # Check each point: all distances from it to others must be unique
        for i in range(K):
            d_from_i = D[i, np.arange(K) != i]
            if len(np.unique(d_from_i)) < len(d_from_i):
                return False
        # Also need to check cross-distances: d(i,j) == d(k,l) doesn't matter,
        # but d(i,j) == d(i,k) or d(i,j) == d(j,k) does
        # The above check covers d(i,j)==d(i,k) for fixed i
        # We also need d(i,j)==d(j,k) which is d(j,i)==d(j,k), covered when i=j
        return True
    
    # Extract submatrix
    D = D_grid[np.ix_(S_indices, S_indices)]
    
    # Check for isosceles triangles
    for i in range(K):
        d_from_i = D[i, np.arange(K) != i]
        if len(np.unique(d_from_i)) < len(d_from_i):
            return False
    return True

class NoIsoscelesSet:
    """Fast data structure for maintaining a no-isosceles set."""
    def __init__(self):
        self.S = []
        self.S_set = set()
        # For each point in S, store the set of distances to other points in S
        # Use dict: point_index -> set of distances
        self.dist_map = {}
    
    def __len__(self):
        return len(self.S)
    
    def __contains__(self, idx):
        return idx in self.S_set
    
    def is_compatible(self, p_idx):
        """Check if p_idx can be added without creating an isosceles triangle."""
        if not self.S:
            return True
        
        K = len(self.S)
        
        # Get distances from p_idx to all points in S
        d_to_S = D_grid[p_idx, self.S]
        
        # Check 1: d(p, S_i) must be all distinct
        if len(np.unique(d_to_S)) < K:
            return False
        
        # Check 2: For each S_i, d(p, S_i) must not equal d(S_i, S_j) for any j != i
        # Vectorized: build a boolean mask of conflicts
        # For each i, we need d_to_S[i] not in dist_map[S_i]
        # Since dist_map values are sets, we check via membership
        # But we can vectorize by checking all at once using numpy
        S_arr = np.array(self.S)
        # d_to_S[i] vs D_grid[S_i, S_j] for j != i
        # This is: for each i, is d_to_S[i] in {D_grid[S_i, S_j] : j != i}?
        # Equivalent: is there a j != i such that D_grid[p_idx, S_i] == D_grid[S_i, S_j]?
        # i.e., is there a j != i such that D_grid[p_idx, S_i] == D_grid[S_i, S_j]?
        
        # Build the submatrix of distances among S points
        D_sub = D_grid[np.ix_(S_arr, S_arr)]
        
        # For each i, d_to_S[i] should not equal D_sub[i, j] for any j != i
        # D_sub[i, :] gives distances from S_i to all S_j
        # We need: for all i, d_to_S[i] not in D_sub[i, np.arange(K) != i]
        
        # Create mask: conflict[i] = any(d_to_S[i] == D_sub[i, j] for j != i)
        # D_sub has 0 on diagonal, so we can compare d_to_S[:, None] with D_sub
        # conflict[i] = any(D_sub[i, :] == d_to_S[i]) but excluding j==i where D_sub[i,i]=0
        # Since d_to_S[i] > 0 (p != S_i), and D_sub[i,i] = 0, the diagonal doesn't cause false positives
        conflict = np.any(D_sub == d_to_S[:, None], axis=1)
        
        if np.any(conflict):
            return False
        
        return True
    
    def add(self, p_idx):
        """Add a point to the set."""
        self.S.append(p_idx)
        self.S_set.add(p_idx)
        
        new_dists = set()
        for s_idx in self.S[:-1]:
            d = D_grid[p_idx, s_idx]
            self.dist_map[s_idx].add(d)
            new_dists.add(d)
        self.dist_map[p_idx] = new_dists
    
    def remove(self, p_idx):
        """Remove a point from the set."""
        self.S.remove(p_idx)
        self.S_set.remove(p_idx)
        
        p_dists = self.dist_map.pop(p_idx)
        for s_idx in self.S:
            d = D_grid[p_idx, s_idx]
            self.dist_map[s_idx].discard(d)
    
    def get_list(self):
        return self.S[:]

def construct():
    start_time = time.time()
    best_set = []
    
    all_indices = list(range(NUM_POINTS))
    
    # Run multiple restarts
    restart_count = 0
    while time.time() < DEADLINE - 15:
        restart_count += 1
        
        # Strategy 1: Random order greedy with multiple attempts
        # Try 3 different random orderings and keep the best
        best_local_S = None
        best_local_size = 0
        
        for _ in range(3):
            random.shuffle(all_indices)
            S_obj_try = NoIsoscelesSet()
            
            for idx in all_indices:
                if S_obj_try.is_compatible(idx):
                    S_obj_try.add(idx)
            
            if len(S_obj_try) > best_local_size:
                best_local_size = len(S_obj_try)
                best_local_S = S_obj_try
        
        S_obj = best_local_S
        
        if len(S_obj) > len(best_set):
            best_set = S_obj.get_list()
        
        # Local search: try to improve
        local_deadline = min(time.time() + 12, DEADLINE - 15)
        pool = [i for i in all_indices if i not in S_obj.S_set]
        
        # Track best size seen during local search
        local_best_size = len(S_obj)
        local_best_set = S_obj.get_list()
        
        no_improve_count = 0
        max_no_improve = 200
        
        while time.time() < local_deadline and pool:
            # Try adding a random point
            if pool:
                p_idx = random.choice(pool)
                if S_obj.is_compatible(p_idx):
                    S_obj.add(p_idx)
                    pool.remove(p_idx)
                    no_improve_count = 0
                    if len(S_obj) > local_best_size:
                        local_best_size = len(S_obj)
                        local_best_set = S_obj.get_list()
                    continue
            
            # Try swap: remove one, add one or two
            if S_obj.S and pool:
                # Remove a random point from S
                rem_idx = random.choice(S_obj.S)
                S_obj.remove(rem_idx)
                if rem_idx not in pool:
                    pool.append(rem_idx)
                
                # Try to add up to 3 points from pool (was 2)
                added = 0
                shuffled_pool = pool[:]
                random.shuffle(shuffled_pool)
                
                for cand in shuffled_pool:
                    if added >= 3:
                        break
                    if cand in S_obj.S_set:
                        continue
                    if S_obj.is_compatible(cand):
                        S_obj.add(cand)
                        if cand in pool:
                            pool.remove(cand)
                        added += 1
                
                # If we couldn't add back at least 1, restore
                if added == 0:
                    S_obj.add(rem_idx)
                    if rem_idx in pool:
                        pool.remove(rem_idx)
                    no_improve_count += 1
                else:
                    no_improve_count = 0
                    if len(S_obj) > local_best_size:
                        local_best_size = len(S_obj)
                        local_best_set = S_obj.get_list()
                
                # Accept neutral moves (same size) to escape local minima
                # With probability 0.3, keep the swap even if size didn't increase
                # This is handled implicitly since we only restore if added==0
                
                # If too many no-improve moves, restart local search phase
                if no_improve_count >= max_no_improve:
                    no_improve_count = 0
                    # Reset pool to be more diverse
                    pool = [i for i in all_indices if i not in S_obj.S_set]
                    random.shuffle(pool)
            
            # Check time more frequently
            if time.time() > local_deadline:
                break
        
        # Use the best set found during local search
        if local_best_size > len(S_obj):
            # Reconstruct from local_best_set
            S_obj = NoIsoscelesSet()
            for idx in local_best_set:
                S_obj.add(idx)
        
        if len(S_obj) > len(best_set):
            best_set = S_obj.get_list()
        
        # Early stop if we found a very good solution
        if len(best_set) > 85:
            break
    
    # Final verification
    if best_set:
        if not is_valid_set(best_set):
            # Fallback: find a small valid set
            return [(0, 0), (1, 0)]
    
    return [tuple(points[i]) for i in best_set]
