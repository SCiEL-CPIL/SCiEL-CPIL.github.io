import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill
    
    # Pre-compute all weight-4 candidates: 4 non-zero entries of +/-1
    candidates = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
    candidates = np.array(candidates)
    num_candidates = len(candidates)
    
    # Pre-compute all pairwise dot products for conflict detection
    # dot[i,j] > 2 means conflict
    # This is a 5280x5280 matrix, might be large but let's try
    # Actually, let's compute on the fly or use a sparse approach
    
    # Better: pre-compute dot products as a matrix for fast lookup
    # 5280^2 = ~27.9M entries, each int8 is fine
    dots = np.zeros((num_candidates, num_candidates), dtype=np.int8)
    for i in range(num_candidates):
        for j in range(i, num_candidates):
            d = int(np.dot(candidates[i], candidates[j]))
            dots[i, j] = d
            dots[j, i] = d
    
    # Conflict: dots[i,j] > 2
    # We want max independent set in graph where edge = conflict
    
    # Strategy: local search with add/remove moves
    # Start with greedy, then iteratively try to improve
    
    def greedy_init(order):
        selected = set()
        for idx in order:
            # Check if idx conflicts with any selected
            conflict = False
            for s in selected:
                if dots[idx, s] > 2:
                    conflict = True
                    break
            if not conflict:
                selected.add(idx)
        return selected
    
    def local_search(initial_set, max_iters=1000):
        current = set(initial_set)
        best = set(initial_set)
        best_size = len(best)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try to add a vertex
            # For each unselected vertex, count conflicts with current
            # Add the one with fewest conflicts (if 0 conflicts, add it)
            best_add = -1
            min_conflicts = len(current) + 1
            for idx in range(num_candidates):
                if idx in current:
                    continue
                conflicts = 0
                for s in current:
                    if dots[idx, s] > 2:
                        conflicts += 1
                        if conflicts >= min_conflicts:
                            break
                if conflicts < min_conflicts:
                    min_conflicts = conflicts
                    best_add = idx
            
            if best_add >= 0 and min_conflicts == 0:
                current.add(best_add)
                improved = True
                if len(current) > best_size:
                    best = set(current)
                    best_size = len(best)
            
            # Try to remove a vertex and add multiple
            # For each vertex in current, remove it and see if we can add more
            if not improved:
                for rem in list(current):
                    trial = set(current)
                    trial.discard(rem)
                    # Try to add vertices
                    added = 0
                    for idx in range(num_candidates):
                        if idx in trial:
                            continue
                        # Check conflicts with trial
                        can_add = True
                        for s in trial:
                            if dots[idx, s] > 2:
                                can_add = False
                                break
                        if can_add:
                            trial.add(idx)
                            added += 1
                    if len(trial) > len(current):
                        current = trial
                        improved = True
                        if len(current) > best_size:
                            best = set(current)
                            best_size = len(best)
                        break
            
            if not improved:
                break
        
        return best
    
    # Run multiple random restarts with local search
    np.random.seed(42)
    best_result_set = set()
    best_k = 0
    
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        order = np.random.permutation(num_candidates)
        init_set = greedy_init(order)
        
        # Local search
        result_set = local_search(init_set, max_iters=200)
        
        if len(result_set) > best_k:
            best_k = len(result_set)
            best_result_set = set(result_set)
    
    # Convert to array
    if best_k == 0:
        # Fallback to seed
        seed_vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    seed_vectors.append(v)
        return np.array(seed_vectors, dtype=np.int64)
    
    result = candidates[list(best_result_set)]
    return result
