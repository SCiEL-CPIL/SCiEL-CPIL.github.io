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
    D = D_grid[np.ix_(S_indices, S_indices)]
    
    # Check for isosceles triangles: for each point i, all distances to other points must be unique
    for i in range(K):
        d_from_i = D[i, np.arange(K) != i]
        if len(np.unique(d_from_i)) < len(d_from_i):
            return False
    return True

class NoIsoscelesSet:
    """Fast data structure for maintaining a no-isosceles set with vectorized validity check."""
    def __init__(self):
        self.S = []
        self.S_set = set()
        # For each point in S, store the set of distances to other points in S
        self.dist_map = {}
        # Track which distances are "forbidden" for each point
        # forbidden[i] = set of distances that point i cannot have to any other point in S
        self.forbidden = {}
    
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
        # This means d(p, S_i) must not be in dist_map[S_i]
        for i, s_idx in enumerate(self.S):
            if d_to_S[i] in self.dist_map[s_idx]:
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

def greedy_initial():
    """Generate an initial solution using greedy with random order."""
    all_indices = list(range(NUM_POINTS))
    random.shuffle(all_indices)
    S_obj = NoIsoscelesSet()
    
    for idx in all_indices:
        if S_obj.is_compatible(idx):
            S_obj.add(idx)
    
    return S_obj

def simulated_annealing(S_obj, deadline):
    """
    Simulated annealing to improve the no-isosceles set.
    Moves: 
      1. Add a random point from pool
      2. Remove a random point from S
      3. Swap: remove one from S, add one from pool
    """
    best_set = S_obj.get_list()
    best_len = len(S_obj)
    current_len = len(S_obj)
    
    pool = [i for i in range(NUM_POINTS) if i not in S_obj.S_set]
    
    T = 10.0  # Initial temperature
    T_min = 0.01
    cooling = 0.999
    
    step = 0
    while time.time() < deadline and T > T_min:
        step += 1
        
        if not pool and current_len == 0:
            break
        
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.3 and pool:
            # Move 1: Add a random point from pool
            p_idx = random.choice(pool)
            if S_obj.is_compatible(p_idx):
                S_obj.add(p_idx)
                pool.remove(p_idx)
                new_len = len(S_obj)
                # Always accept improvements
                if new_len > current_len:
                    current_len = new_len
                    if new_len > best_len:
                        best_len = new_len
                        best_set = S_obj.get_list()
            else:
                # Reject
                pass
        
        elif move_type < 0.5 and current_len > 2:
            # Move 2: Remove a random point from S
            rem_idx = random.choice(S_obj.S)
            S_obj.remove(rem_idx)
            pool.append(rem_idx)
            new_len = len(S_obj)
            
            # Accept with probability based on temperature
            delta = new_len - current_len  # Negative
            if delta >= 0 or random.random() < np.exp(-(-delta) / T):
                current_len = new_len
            else:
                # Reject: restore
                S_obj.add(rem_idx)
                pool.remove(rem_idx)
        
        else:
            # Move 3: Swap - remove one from S, add one from pool
            if current_len > 0 and pool:
                rem_idx = random.choice(S_obj.S)
                S_obj.remove(rem_idx)
                pool.append(rem_idx)
                
                # Try to add a random point from pool
                cand_idx = random.choice(pool)
                if cand_idx != rem_idx and S_obj.is_compatible(cand_idx):
                    S_obj.add(cand_idx)
                    pool.remove(cand_idx)
                    new_len = len(S_obj)
                    delta = new_len - current_len
                    
                    if delta >= 0 or random.random() < np.exp(-(-delta) / T):
                        current_len = new_len
                        if new_len > best_len:
                            best_len = new_len
                            best_set = S_obj.get_list()
                    else:
                        # Reject: restore
                        if cand_idx in S_obj.S_set:
                            S_obj.remove(cand_idx)
                            pool.append(cand_idx)
                        S_obj.add(rem_idx)
                        if rem_idx in pool:
                            pool.remove(rem_idx)
                else:
                    # Couldn't add, restore
                    S_obj.add(rem_idx)
                    if rem_idx in pool:
                        pool.remove(rem_idx)
        
        # Cool down
        T *= cooling
        
        # Periodic restart if stuck
        if step % 5000 == 0 and current_len < best_len - 5:
            # Revert to best
            S_obj = NoIsoscelesSet()
            for idx in best_set:
                S_obj.add(idx)
            current_len = len(S_obj)
            pool = [i for i in range(NUM_POINTS) if i not in S_obj.S_set]
            T = 5.0  # Reheat
    
    # Try to add more points greedily
    random.shuffle(pool)
    for p_idx in pool:
        if S_obj.is_compatible(p_idx):
            S_obj.add(p_idx)
    
    if len(S_obj) > best_len:
        best_set = S_obj.get_list()
    
    return best_set

def construct():
    start_time = time.time()
    best_set = []
    
    # Run multiple SA restarts
    restart_count = 0
    while time.time() < DEADLINE - 10:
        restart_count += 1
        
        # Generate initial solution
        S_obj = greedy_initial()
        
        if len(S_obj) > len(best_set):
            best_set = S_obj.get_list()
        
        # Run simulated annealing
        sa_deadline = min(time.time() + 20, DEADLINE - 10)
        improved_set = simulated_annealing(S_obj, sa_deadline)
        
        if len(improved_set) > len(best_set):
            best_set = improved_set
        
        # Early stop if we found a very good solution
        if len(best_set) > 90:
            break
    
    # Final verification
    if best_set:
        if not is_valid_set(best_set):
            # Fallback: find a small valid set
            return [(0, 0), (1, 0)]
    
    return [tuple(points[i]) for i in best_set]
