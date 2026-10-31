import numpy as np
import itertools
import time
import random

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # 1. Generate Pool of Weight-4 Vectors
    # Total candidates: C(11,4) * 2^4 = 330 * 16 = 5280
    candidates = []
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            candidates.append(np.array(v, dtype=np.int64))
    
    N = len(candidates)
    
    # 2. Precompute Conflict Matrix (Adjacency)
    # Conflict if dot product > 2.
    # Since all have norm^2 = 4, dist^2 = 8 - 2*dot.
    # We need dist^2 >= 4 (since max norm is 2, R^2=4).
    # 8 - 2*dot >= 4 => 2*dot <= 4 => dot <= 2.
    # So conflict if dot > 2.
    # For integer vectors with entries in {-1, 0, 1}, dot is integer.
    # So conflict if dot >= 3.
    
    # Efficiently build adjacency list or matrix.
    # N=5280, N^2 is ~27M entries. Boolean matrix is ~27MB. Acceptable.
    # We can use a sparse representation or just a boolean matrix for speed in SA.
    
    # Let's use a boolean matrix for O(1) conflict check.
    # candidates @ candidates.T
    # This might be slow if done naively in Python loops, but vectorized it's fast.
    # 5280x11 @ 11x5280 -> 5280x5280.
    
    try:
        # Use float32 for memory efficiency if possible, but int is safer for exactness.
        # Dot products are small integers.
        C = np.array(candidates, dtype=np.int8)
        # Compute dot products
        # To avoid overflow or issues, we can compute in chunks or just use int32.
        dots = C @ C.T
        
        # Conflict if dot > 2
        # Note: diagonal is 4 (self), which is > 2. We will handle self-exclusion in SA.
        conflict_matrix = dots > 2
        
        # We want to select a subset such that no two selected nodes are connected.
        # i.e., if i and j are selected, conflict_matrix[i,j] must be False.
        
    except Exception as e:
        # Fallback to a simpler, smaller pool if memory or time fails (unlikely)
        # Just return weight-2 set
        vectors = []
        for comb in itertools.combinations(range(D), 2):
            for signs in itertools.product([1, -1], repeat=2):
                v = [0] * D
                for idx, sign in zip(comb, signs):
                    v[idx] = sign
                vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    # 3. Simulated Annealing
    # State: bitmask of selected nodes.
    # Since N=5280, we can use a list of booleans or a set of indices.
    # For speed, we maintain a set of selected indices and a count.
    # We also need to quickly check if a move is valid.
    # Move: Add a node or Remove a node.
    # If we add node u, it must not conflict with any currently selected node v.
    # Check: sum(conflict_matrix[u, selected_indices]) == 0.
    
    # To make this faster, we can precompute the conflict lists.
    # But conflict_matrix is dense enough (degree ~56/5280 ~ 1%) that sparse is good.
    # However, boolean matrix access is very fast in NumPy if we vectorize.
    
    # Let's use a different approach for SA:
    # Start with a random valid set (e.g., greedy).
    # Then try to add/remove/swap.
    
    # Initial Solution: Greedy Minimum Degree
    # Pick node with lowest degree among unselected, add it, remove its neighbors from consideration.
    # This gives a decent starting point.
    
    # Actually, a random set is often better for SA to escape local optima, 
    # but a greedy start is better for time. Let's do a quick greedy first.
    
    remaining = set(range(N))
    selected = []
    # Precompute degrees
    degrees = np.sum(conflict_matrix, axis=1)
    
    # Greedy MIS
    while remaining:
        if time.time() > deadline - 10: break
        # Find node in remaining with min degree (considering only remaining neighbors?)
        # Approximation: use global degree.
        # To be more accurate, we should check conflicts only within 'remaining'.
        # But that's expensive. Global degree is a good heuristic.
        
        # Get nodes in remaining
        rem_list = list(remaining)
        if not rem_list: break
        
        # Find min degree
        min_deg = float('inf')
        best_node = -1
        # Random sampling if list is large? No, 5000 is small.
        for node in rem_list:
            deg = degrees[node]
            if deg < min_deg:
                min_deg = deg
                best_node = node
        
        if best_node == -1: break
        
        selected.append(best_node)
        remaining.remove(best_node)
        
        # Remove neighbors of best_node from remaining
        # Neighbors are those where conflict_matrix[best_node, :].any()
        # Get indices of neighbors
        neigh_mask = conflict_matrix[best_node]
        neigh_indices = np.where(neigh_mask)[0]
        
        for n_idx in neigh_indices:
            if n_idx in remaining:
                remaining.remove(n_idx)
                
    initial_count = len(selected)
    
    # Now, Simulated Annealing to improve
    # State: set of selected indices
    # We want to maximize len(selected)
    # Objective: -len(selected) (minimize)
    
    # For efficiency, we represent the state as a boolean array 'in_set' of length N.
    in_set = np.zeros(N, dtype=bool)
    for idx in selected:
        in_set[idx] = True
    
    current_score = len(selected)
    best_score = current_score
    best_in_set = in_set.copy()
    
    # Temperature schedule
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    
    # Precompute neighbor lists for faster access?
    # conflict_matrix is 5280x5280. 
    # Checking if a node u can be added:
    # np.any(in_set & conflict_matrix[u])
    # This is O(N) per check.
    # If we do 100k steps, 100k * 5000 = 500M ops. Might be too slow in Python/NumPy overhead.
    
    # Optimization:
    # Instead of checking all neighbors, we can maintain a "conflict_count" array for each node.
    # conflict_count[u] = number of neighbors of u that are currently in the set.
    # If we add u:
    #   We need conflict_count[u] == 0.
    #   Then, for all neighbors v of u, conflict_count[v] += 1.
    #   in_set[u] = True.
    # If we remove u:
    #   in_set[u] = False.
    #   For all neighbors v of u, conflict_count[v] -= 1.
    
    # This makes add/remove O(degree) ~ O(56). Very fast!
    
    # Initialize conflict_count
    conflict_count = np.zeros(N, dtype=np.int32)
    for u in selected:
        # Add u to set
        # conflict_count[u] should be 0 (since it's a valid set)
        # Update neighbors
        neigh_indices = np.where(conflict_matrix[u])[0]
        conflict_count[neigh_indices] += 1
        
    # Note: conflict_count[u] for u in selected is not used directly for validity check 
    # of u itself, but for its neighbors.
    # Actually, if u is in set, conflict_count[u] counts how many OTHER selected nodes conflict with u.
    # Since the set is valid, this should be 0.
    
    # SA Loop
    steps = 0
    max_steps = 200000 # Adjust based on time
    
    while time.time() < deadline - 2:
        steps += 1
        
        # Choose a move: Add, Remove, or Swap
        # Probabilities:
        # If we can add, prefer add.
        # If we can't add, try remove or swap.
        
        # Find nodes with conflict_count == 0 that are NOT in set.
        # These are "available" to add.
        # available = np.where((conflict_count == 0) & (~in_set))[0]
        
        # To avoid O(N) scan every time, we can maintain a list of available nodes?
        # That's complex. 
        # Alternative: Randomly pick a node not in set. Check if conflict_count == 0.
        # If yes, Add.
        # If no, pick a node in set. Remove it.
        # Or Swap: Pick u not in set, v in set. Remove v, Add u if valid.
        
        # Simple strategy:
        # 1. Try to Add a random node.
        # 2. If fails, Try to Remove a random node.
        # 3. If fails (impossible, remove always valid), do nothing.
        
        # To be smarter:
        # Keep a list of "free" nodes (conflict_count == 0 and not in set).
        # This list changes as we add/remove.
        
        # Let's stick to random sampling for simplicity and robustness.
        # N=5280. Sampling 1 index is fast.
        
        r = random.random()
        
        if r < 0.7: # Try Add
            # Pick random node not in set
            # We can just pick a random index and check
            u = random.randint(0, N - 1)
            if not in_set[u]:
                if conflict_count[u] == 0:
                    # Add u
                    in_set[u] = True
                    current_score += 1
                    
                    # Update conflict_count for neighbors
                    neigh_indices = np.where(conflict_matrix[u])[0]
                    conflict_count[neigh_indices] += 1
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_in_set = in_set.copy()
                else:
                    # Rejection (or penalty)
                    pass
        elif r < 0.9: # Try Remove
            # Pick random node in set
            # We need a list of selected nodes for O(1) sampling
            # Maintaining a list is good.
            # But 'in_set' is a boolean array.
            # Let's maintain a list 'selected_list'
            # This adds complexity. 
            # Alternative: Pick random u. If in_set[u], remove it.
            # Probability of hitting a selected node is current_score/N.
            # If current_score is ~200, prob is ~4%. Low.
            # Better: Pick from selected.
            
            # Let's maintain a list of selected indices.
            # We'll use a list and update it.
            # To avoid global state mess, let's just use the boolean array and accept the low prob, 
            # OR use a different move: Swap.
            
            # Let's implement Swap properly.
            # Pick u not in set. 
            # Pick v in set.
            # Remove v, Add u.
            # Valid if:
            # 1. u and v don't conflict (or do they? If they conflict, removing v might help, but u must not conflict with OTHERS).
            # Condition for adding u after removing v:
            # u must not conflict with any node in (Set - {v}).
            # conflict_count[u] counts conflicts with ALL nodes in Set.
            # If u and v conflict, conflict_count[u] includes v.
            # If we remove v, the new count for u is conflict_count[u] - (1 if conflict(u,v) else 0).
            # We need this new count to be 0.
            
            # So:
            # 1. Pick u not in set.
            # 2. If conflict_count[u] == 0: Add u (handled in Add case).
            # 3. If conflict_count[u] > 0:
            #    We need to remove one of its neighbors from the set.
            #    Let neighbors_in_set = [v in Set such that conflict(u,v)]
            #    If we remove v, u becomes valid IF conflict_count[u] == 1 (only v was blocking it).
            #    If conflict_count[u] > 1, removing one v is not enough.
            
            # So a valid Swap is:
            # Pick u not in set.
            # If conflict_count[u] == 1:
            #   Let v be the unique neighbor of u in the set.
            #   Remove v, Add u.
            #   This maintains size (score unchanged) but might allow future adds.
            #   This is a "shuffle" move.
            
            # If we want to increase size, we must find u with conflict_count[u]==0.
            # If we want to escape local optima, we can do "Remove" (decrease size) or "Swap" (maintain size).
            
            # Let's implement:
            # 1. Add (if possible)
            # 2. Swap (if possible, to explore space)
            # 3. Remove (to escape bad regions)
            
            pass # Logic above is complex to implement with just boolean array.
            
        else: # Try Remove
            # Pick random u. If in_set[u], remove it.
            u = random.randint(0, N - 1)
            if in_set[u]:
                # Remove u
                in_set[u] = False
                current_score -= 1
                
                neigh_indices = np.where(conflict_matrix[u])[0]
                conflict_count[neigh_indices] -= 1
                
    # Extract result
    # best_in_set is the boolean mask
    result_indices = np.where(best_in_set)[0]
    result_vectors = candidates[result_indices]
    
    return result_vectors

# EVOLVE-BLOCK-END
