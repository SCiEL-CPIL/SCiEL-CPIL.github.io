"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 295
    
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = N * N
    
    xs = np.array([p[0] for p in points], dtype=np.int32)
    ys = np.array([p[1] for p in points], dtype=np.int32)
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2 = (dx * dx + dy * dy).astype(np.int32)
    
    max_dist2 = 2 * (N - 1) * (N - 1)
    
    # used_dist[i, d] is True if distance d is used by point i to some point in S
    used_dist = np.zeros((num_points, max_dist2 + 1), dtype=bool)
    
    S = []
    S_set = set()
    
    def can_add(P_idx):
        if P_idx in S_set:
            return False
        
        if not S:
            return True
        
        S_arr = np.array(S, dtype=np.int32)
        dists = dist2[P_idx, S_arr]
        
        # Check 1: Distances from P to S must be unique
        if len(np.unique(dists)) != len(dists):
            return False
        
        # Check 2: For each Q in S, d(P, Q) must not be in used_dist[Q]
        for j, Q_idx in enumerate(S):
            d_val = dists[j]
            if used_dist[Q_idx, d_val]:
                return False
        
        return True

    def add(P_idx):
        S_set.add(P_idx)
        S.append(P_idx)
        
        if not S:
            return
        
        S_arr = np.array(S, dtype=np.int32)
        dists_P = dist2[P_idx, S_arr]
        
        # Mark P's distances to S
        used_dist[P_idx, dists_P] = True
        
        # Mark Q's distances to P
        used_dist[S_arr, dists_P] = True

    def remove(P_idx):
        if P_idx not in S_set:
            return
        
        S_set.remove(P_idx)
        S.remove(P_idx)
        
        if not S:
            used_dist[P_idx, :] = False
            return
        
        S_arr = np.array(S, dtype=np.int32)
        dists_P = dist2[P_idx, S_arr]
        
        # Clear P's distances to S
        used_dist[P_idx, dists_P] = False
        
        # Clear Q's distances to P
        used_dist[S_arr, dists_P] = False

    best_S = []
    best_score = 0
    
    rng = random.Random(42)
    
    # Greedy Initialization with multiple random orders
    for trial in range(50):
        S.clear()
        S_set.clear()
        used_dist[:] = False
        
        order = list(range(num_points))
        rng.shuffle(order)
        
        for idx in order:
            if can_add(idx):
                add(idx)
        
        if len(S) > best_score:
            best_score = len(S)
            best_S = S.copy()
    
    # Rebuild state for best_S
    S = best_S.copy()
    S_set = set(S)
    used_dist[:] = False
    
    # Rebuild used_dist
    for i_idx in S:
        for j_idx in S:
            if i_idx != j_idx:
                d = dist2[i_idx, j_idx]
                used_dist[i_idx, d] = True

    # Local Search: Add and Kick (Remove + Add)
    start_time = time.time()
    no_improve = 0
    
    while time.time() < deadline - 10:
        if no_improve > 2000:
            break
        
        move_type = rng.random()
        
        if move_type < 0.7:
            # Try Add
            i_in = rng.randint(0, num_points - 1)
            if i_in not in S_set and can_add(i_in):
                add(i_in)
                if len(S) > best_score:
                    best_score = len(S)
                    best_S = S.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
        else:
            # Kick: Remove one, then try to add one
            if len(S) > 0:
                i_out = rng.choice(S)
                remove(i_out)
                
                added = False
                for _ in range(5):
                    i_in = rng.randint(0, num_points - 1)
                    if i_in not in S_set and can_add(i_in):
                        add(i_in)
                        added = True
                        break
                
                if added:
                    no_improve = 0
                else:
                    add(i_out)
                    no_improve += 1
    
    return [points[i] for i in best_S]
# EVOLVE-BLOCK-END
