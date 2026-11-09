import time
import random
import numpy as np

N = 64
NUM_POINTS = N * N
MAX_TIME = 280.0

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME

    # Precompute all squared distances between points
    # d2[i, j] = squared distance between point i and point j
    xs = np.arange(N)
    ys = np.arange(N)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing='ij')
    coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    # Compute pairwise squared distances
    # d2 = coords[:, 0:1]**2 + coords[:, 1:2]**2 - 2 * coords @ coords.T
    # But this might have floating point issues. Let's use integer arithmetic.
    d2 = (coords[:, 0].reshape(-1, 1) - coords[0:1, 0].reshape(1, -1))**2 + \
         (coords[:, 1].reshape(-1, 1) - coords[0:1, 1].reshape(1, -1))**2
    
    # Actually, let's just compute it properly
    dx = coords[:, 0].reshape(-1, 1) - coords[:, 0].reshape(1, -1)
    dy = coords[:, 1].reshape(-1, 1) - coords[:, 1].reshape(1, -1)
    d2 = dx**2 + dy**2
    
    # d2[i, j] is the squared distance from point i to point j
    # Note: d2[i, i] = 0
    
    # Strategy: Start with a random valid set, then use local search to grow it.
    # A point p can be added to set S if:
    # 1. d2[p, i] is unique for all i in S (no two points in S are at the same distance from p)
    # 2. For each i in S, d2[p, i] is not in the set of distances from i to other points in S
    
    # Let's maintain for each point i in S, a set of distances d(i, j) for j in S, j != i
    # We'll use a dictionary: dist_sets[i] = set of squared distances from i to other points in S
    
    def can_add(p, S, dist_sets):
        if not S:
            return True
        # Check 1: All distances from p to points in S must be unique
        d2s = d2[p, S]
        if len(set(d2s)) != len(d2s):
            return False
        # Check 2: For each i in S, d2[p, i] must not be in dist_sets[i]
        for i, d in zip(S, d2s):
            if d in dist_sets[i]:
                return False
        return True
    
    def add_point(p, S, dist_sets):
        S.append(p)
        dist_sets[p] = set()
        # Update dist_sets for all existing points
        for i in S[:-1]:
            d = d2[p, i]
            dist_sets[i].add(d)
            dist_sets[p].add(d)
    
    def remove_point(p, S, dist_sets):
        idx = S.index(p)
        S.pop(idx)
        for i in S:
            dist_sets[i].discard(d2[p, i])
        del dist_sets[p]
    
    best_S = []
    best_dist_sets = {}
    
    # Initial construction: random greedy
    random.seed(42)
    all_indices = list(range(NUM_POINTS))
    
    # Try multiple random restarts
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        random.shuffle(all_indices)
        S = []
        dist_sets = {}
        
        for p in all_indices:
            if time.time() > deadline - 5:
                break
            if can_add(p, S, dist_sets):
                add_point(p, S, dist_sets)
        
        if len(S) > len(best_S):
            best_S = S[:]
            best_dist_sets = {k: v.copy() for k, v in dist_sets.items()}
        
        # Local search: try to swap points to enable adding more
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try swapping each point in S with a random point not in S
            random.shuffle(S)
            for i, p in enumerate(S):
                if time.time() > deadline - 10:
                    break
                q = random.choice(all_indices)
                if q in S:
                    continue
                # Try swapping p with q
                remove_point(p, S, dist_sets)
                if can_add(q, S, dist_sets):
                    add_point(q, S, dist_sets)
                    improved = True
                    if len(S) > len(best_S):
                        best_S = S[:]
                        best_dist_sets = {k: v.copy() for k, v in dist_sets.items()}
                else:
                    # Revert
                    add_point(p, S, dist_sets)
            
            # Also try just adding more points if possible
            for p in all_indices:
                if p not in S and can_add(p, S, dist_sets):
                    add_point(p, S, dist_sets)
                    improved = True
                    if len(S) > len(best_S):
                        best_S = S[:]
                        best_dist_sets = {k: v.copy() for k, v in dist_sets.items()}
    
    # Final attempt: try to add points to best_S
    if time.time() < deadline - 5:
        S = best_S[:]
        dist_sets = {k: v.copy() for k, v in best_dist_sets.items()}
        remaining = [p for p in all_indices if p not in S]
        random.shuffle(remaining)
        for p in remaining:
            if time.time() > deadline - 2:
                break
            if can_add(p, S, dist_sets):
                add_point(p, S, dist_sets)
                if len(S) > len(best_S):
                    best_S = S[:]
                    best_dist_sets = {k: v.copy() for k, v in dist_sets.items()}
    
    return [tuple(coords[i]) for i in best_S]
