import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    D = 11
    
    # 1. Generate Candidate Vectors
    # We look for integer vectors u, v such that ||u-v|| >= ||u|| and ||u-v|| >= ||v||.
    # This is equivalent to ||u-v||^2 >= max(||u||^2, ||v||^2).
    # Let M = max(||u||^2, ||v||^2).
    # ||u||^2 + ||v||^2 - 2 u.v >= M
    # 2 u.v <= ||u||^2 + ||v||^2 - M
    
    # To maximize the count, we want vectors with small norms.
    # Norm^2 = 2: Weight 2, entries +/-1. (220 vectors)
    # Norm^2 = 4: Weight 1, entries +/-2 (22 vectors) OR Weight 4, entries +/-1 (5280 vectors).
    
    # Let's combine:
    # V1: Weight 1, entries +/-2. Norm^2 = 4.
    # V2: Weight 2, entries +/-1. Norm^2 = 2.
    # V4: Weight 4, entries +/-1. Norm^2 = 4.
    
    v_list = []
    
    # V1: 22 vectors
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v_list.append(v)
            
    # V2: 220 vectors
    for pos1 in range(D):
        for pos2 in range(pos1 + 1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * D
                    v[pos1] = s1
                    v[pos2] = s2
                    v_list.append(v)
                    
    # V4: 5280 vectors
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(combo, signs):
                v[pos] = sign
            v_list.append(v)
            
    V_all = np.array(v_list, dtype=np.int8)
    N = len(V_all)
    
    # 2. Compute Conflict Matrix
    # Conflict if ||u-v||^2 < max(||u||^2, ||v||^2)
    # i.e., ||u||^2 + ||v||^2 - 2 u.v < max(||u||^2, ||v||^2)
    # 2 u.v > ||u||^2 + ||v||^2 - max(||u||^2, ||v||^2)
    # 2 u.v > min(||u||^2, ||v||^2)
    
    NormSq = np.sum(V_all.astype(np.float64) ** 2, axis=1)
    Dots = V_all.astype(np.float64) @ V_all.astype(np.float64).T
    
    # Threshold T[i,j] = min(NormSq[i], NormSq[j]) / 2
    # Conflict if Dots[i,j] > T[i,j]
    
    MinNormSq = np.minimum(NormSq[:, None], NormSq[None, :])
    T = MinNormSq / 2.0
    
    Conflicts = (Dots > T)
    np.fill_diagonal(Conflicts, False)
    
    # 3. Maximum Independent Set (MIS)
    # We want to maximize the number of non-conflicting vectors.
    
    rng = np.random.default_rng(42)
    
    best_selected = None
    best_k = 0
    
    def run_mis(conflicts, deadline, rng):
        N = len(conflicts)
        alive = np.ones(N, dtype=bool)
        # deg[i] is the number of conflicts with alive nodes
        deg = conflicts.sum(axis=1).astype(np.int32)
        
        selected = []
        
        while np.any(alive):
            if time.time() > deadline:
                break
            
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            # Find min degree among alive nodes
            # We use a simple scan. N is ~5500, so this is fast.
            # To speed up, we could use a heap, but numpy min is quite fast.
            min_deg = np.min(deg[alive_indices])
            
            # Pick a random node among those with min degree
            candidates = alive_indices[deg[alive_indices] == min_deg]
            u = candidates[rng.integers(len(candidates))]
            
            selected.append(u)
            
            # Remove u and its neighbors
            # neighbors = np.where(conflicts[u])[0]
            # to_remove = np.union1d([u], neighbors)
            
            # Efficient update:
            # 1. Mark u and neighbors as not alive
            # 2. Update degrees of remaining alive nodes
            
            # Get neighbors
            neighbors = np.where(conflicts[u])[0]
            to_remove = np.concatenate(([u], neighbors))
            
            # Update alive mask
            alive[to_remove] = False
            
            # Update degrees
            # For each remaining alive node v, deg[v] decreases by the number of its neighbors in to_remove
            remaining_alive = np.where(alive)[0]
            if len(remaining_alive) > 0:
                # Count conflicts between remaining alive and removed nodes
                # conflicts[remaining_alive, to_remove] is a matrix
                # Sum along axis 1 gives the reduction in degree for each remaining node
                deg[remaining_alive] -= conflicts[remaining_alive][:, to_remove].sum(axis=1)
                
        return selected

    # Run multiple restarts
    max_restarts = 500
    mis_deadline = deadline - 2.0
    
    for restart in range(max_restarts):
        if time.time() > mis_deadline:
            break
        
        selected = run_mis(Conflicts, mis_deadline, rng)
        
        k = len(selected)
        if k > best_k:
            best_k = k
            best_selected = selected.copy()
            
    if best_selected is None:
        # Fallback: return just V1 (which is always valid? No, V1 vectors conflict with each other if opposite signs?
        # u = [2,0..], v = [-2,0..]. ||u-v|| = 4, max norm = 2. 4 >= 2. OK.
        # u = [2,0..], v = [2,0..]. Same vector. Not allowed (distinct vectors? "pairwise distance" implies distinct indices).
        # If we pick distinct indices, u and v are different.
        # Wait, if we pick u=[2,0..] and v=[2,0..] from the list, they are at different indices but same value.
        # Distance is 0. 0 < 2. Conflict.
        # So V1 alone is not an independent set if it contains both +2 and -2 at same pos?
        # u=[2,0..], v=[-2,0..]. Dot = -4. MinNormSq = 4. T = 2. -4 > 2 is False. No conflict.
        # u=[2,0..], v=[2,0..]. Dot = 4. MinNormSq = 4. T = 2. 4 > 2 is True. Conflict.
        # So we can't pick both +2 and -2 at same pos?
        # Wait, the list V1 has both. So V1 is not an independent set.
        # Fallback: return empty or a single vector.
        return V_all[:1].astype(np.int64)
        
    result = V_all[best_selected].astype(np.int64)
    
    # Optional: Local Search to improve
    # Given the time constraints and the complexity of local search on a 5500-node graph,
    # simple greedy restarts are often quite effective for this type of random-like graph.
    # The graph structure here is deterministic but sparse enough for greedy to work well.
    
    return result
