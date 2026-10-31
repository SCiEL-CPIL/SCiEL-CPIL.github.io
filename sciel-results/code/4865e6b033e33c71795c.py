import numpy as np
import itertools
import time
import random

D = 11

def _gen_weight4():
    """Generate all weight-4 vectors in {-1,0,1}^11 (norm^2=4)."""
    out = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            out.append(v)
    return np.array(out, dtype=np.int64)

def _gen_weight1():
    """Generate all weight-1 vectors in {-2,0,2}^11 (norm^2=4)."""
    out = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            out.append(v)
    return np.array(out, dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Generate candidate pools
    v4_candidates = _gen_weight4()
    v1_candidates = _gen_weight1()
    
    N4 = len(v4_candidates)
    
    if N4 == 0:
        return v1_candidates
    
    # Compute conflict graph for weight-4 vectors
    # For weight-4 vectors in {-1,0,1}^11, norm^2 = 4
    # dist^2 = 4 + 4 - 2*u.v = 8 - 2*u.v
    # Need dist^2 >= max_norm^2 = 4, so 8 - 2*u.v >= 4 => u.v <= 2
    # Conflict if u.v > 2
    dots = v4_candidates @ v4_candidates.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Build adjacency lists
    adj = [np.where(conflicts[i])[0] for i in range(N4)]
    degrees = conflicts.sum(axis=1).astype(np.int32)
    
    # Greedy MIS with minimum degree selection (Tomita-style)
    alive = np.ones(N4, dtype=bool)
    greedy_selected = []
    rng = random.Random(42)
    
    while np.any(alive) and time.time() < deadline - 25:
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        
        # Pick vertex with minimum degree among alive vertices
        min_deg = np.min(degrees[alive_indices])
        cands = alive_indices[degrees[alive_indices] == min_deg]
        chosen = cands[rng.randrange(len(cands))]
        
        greedy_selected.append(chosen)
        
        # Remove chosen and its neighbors
        neighbors = adj[chosen]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        alive[to_remove] = False
        
        # Update degrees
        rem = np.where(alive)[0]
        if len(rem) > 0:
            degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
    
    best_set = set(greedy_selected)
    best_size = len(best_set)
    
    # Local improvement: try to add vertices that don't conflict with current set
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for v in range(N4):
            if v in best_set:
                continue
            # Check if v can be added (no conflicts with current set)
            v_conflicts = conflicts[v]
            if np.any(v_conflicts[list(best_set)]):
                continue
            best_set.add(v)
            best_size += 1
            improved = True
    
    # Try a few random restarts with different initial selections if time permits
    if time.time() < deadline - 10:
        for restart in range(3):
            if time.time() >= deadline - 8:
                break
            alive_r = np.ones(N4, dtype=bool)
            greedy_r = []
            degrees_r = degrees.copy()
            rng_r = random.Random(42 + restart * 1000)
            
            while np.any(alive_r) and time.time() < deadline - 8:
                alive_indices = np.where(alive_r)[0]
                if len(alive_indices) == 0:
                    break
                min_deg = np.min(degrees_r[alive_indices])
                cands = alive_indices[degrees_r[alive_indices] == min_deg]
                chosen = cands[rng_r.randrange(len(cands))]
                greedy_r.append(chosen)
                neighbors = adj[chosen]
                to_remove = np.unique(np.concatenate(([chosen], neighbors)))
                alive_r[to_remove] = False
                rem = np.where(alive_r)[0]
                if len(rem) > 0:
                    degrees_r[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
            
            # Local improvement
            set_r = set(greedy_r)
            imp_r = True
            while imp_r and time.time() < deadline - 8:
                imp_r = False
                for v in range(N4):
                    if v in set_r:
                        continue
                    if np.any(conflicts[v][list(set_r)]):
                        continue
                    set_r.add(v)
                    imp_r = True
            
            if len(set_r) > best_size:
                best_set = set_r
                best_size = len(set_r)
    
    result_v4 = v4_candidates[np.array(list(best_set))]
    
    # All 22 V1 vectors are mutually compatible and compatible with V4
    final_vectors = np.vstack([result_v4, v1_candidates])
    
    return final_vectors
