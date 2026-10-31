import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
    
    # Compute dot products
    dots = vectors_v4 @ vectors_v4.T
    
    # Conflict if dot > 2
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists for each node
    # conflicts is a boolean matrix N4 x N4
    # For each node i, neighbors are indices j where conflicts[i,j] is True
    neighbor_lists = []
    for i in range(N4):
        neighbor_lists.append(np.where(conflicts[i])[0].tolist())
    
    best_mis_size = 0
    best_selected = []
    rng = np.random.default_rng(42)
    
    # Try multiple random restarts of greedy MIS
    max_restarts = 20
    restart_time_budget = (deadline - start_time - 5) / max_restarts if restart_time_budget > 0 else 0.5
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        # Greedy MIS with random ordering
        alive = [True] * N4
        selected = []
        # Create a random priority for each node
        priorities = rng.random(N4)
        # Sort indices by priority (higher priority first)
        # Actually, let's use a min-heap approach or just random ordering
        
        # Simpler: randomize the order in which we consider nodes
        order = list(range(N4))
        rng.shuffle(order)
        
        # Greedy: iterate through nodes in random order, add if no selected neighbor
        selected_set = set()
        # We need to check if a node conflicts with any already selected node
        # To make this efficient, we can track which nodes are "blocked"
        
        blocked = [False] * N4
        
        for node in order:
            if time.time() > deadline - 5:
                break
            if blocked[node]:
                continue
            # Select this node
            selected.append(node)
            selected_set.add(node)
            # Block this node and all its neighbors
            blocked[node] = True
            for nb in neighbor_lists[node]:
                if not blocked[nb]:
                    blocked[nb] = True
        
        if len(selected) > best_mis_size:
            best_mis_size = len(selected)
            best_selected = selected[:]
            
            # Check if we have time for another pass
            if len(selected) >= 500:  # Heuristic threshold
                break
    
    # Also try a deterministic min-degree greedy for comparison
    if time.time() < deadline - 10:
        alive = [True] * N4
        selected = []
        deg = [len(neighbor_lists[i]) for i in range(N4)]
        
        while True:
            if time.time() > deadline - 10:
                break
            # Find alive node with min degree
            min_deg = float('inf')
            cands = []
            for i in range(N4):
                if alive[i] and deg[i] < min_deg:
                    min_deg = deg[i]
                    cands = [i]
                elif alive[i] and deg[i] == min_deg:
                    cands.append(i)
            if not cands:
                break
            chosen = cands[0]  # Deterministic
            selected.append(chosen)
            alive[chosen] = False
            # Update degrees
            for nb in neighbor_lists[chosen]:
                if alive[nb]:
                    deg[nb] -= 1
                    alive[nb] = False  # Remove neighbors
            
            if time.time() > deadline - 10:
                break
        
        if len(selected) > best_mis_size:
            best_mis_size = len(selected)
            best_selected = selected[:]
    
    if not best_selected:
        return vectors_v1
    
    result_v4 = vectors_v4[best_selected]
    
    # Combine V4 and V1
    # V1 vectors are all compatible with V4 and each other
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
