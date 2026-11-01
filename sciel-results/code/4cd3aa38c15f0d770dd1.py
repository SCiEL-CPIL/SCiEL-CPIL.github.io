"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import itertools

D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Generate all integer vectors with norm squared = 4
    # Type A: one coordinate is +/-2 (22 vectors)
    # Type B: four coordinates are +/-1 (C(11,4)*2^4 = 5280 vectors)
    vectors = []
    
    # Type A: one coordinate +/-2
    for i in range(D):
        for sign in (1, -1):
            v = [0] * D
            v[i] = 2 * sign
            vectors.append(v)
    
    # Type B: four coordinates +/-1
    for coords in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0] * D
            for c, s in zip(coords, signs):
                v[c] = s
            vectors.append(v)
    
    n = len(vectors)
    V = np.array(vectors, dtype=np.int64)
    
    # Compute dot product matrix
    dot = V @ V.T
    
    # Conflict: dot[i,j] > 2 for i != j (since norm=2, need dot <= 2)
    conflict = (dot > 2).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    # Build adjacency as lists for fast neighbor lookup
    adj = [np.where(conflict[i] == 1)[0] for i in range(n)]
    degrees = np.array([len(adj[i]) for i in range(n)], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    def greedy_select(order):
        """Greedy independent set selection given an ordering."""
        in_set = np.zeros(n, dtype=bool)
        alive = np.ones(n, dtype=bool)
        
        for v in order:
            if not alive[v]:
                continue
            in_set[v] = True
            # Mark v and all its neighbors as not alive
            alive[v] = False
            for u in adj[v]:
                alive[u] = False
        
        return in_set, int(in_set.sum())
    
    def local_search(in_set, max_iters=500):
        """Local search: try to add non-selected vertices, or swap."""
        in_set = in_set.copy()
        current_size = int(in_set.sum())
        
        # Count conflicts for each non-selected vertex
        # conflict_count[v] = number of selected neighbors of v
        conflict_count = np.zeros(n, dtype=np.int32)
        selected_idx = np.where(in_set)[0]
        for v in selected_idx:
            conflict_count[adj[v]] += 1
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            
            # Try to add a non-selected vertex with 0 conflicts
            non_selected = np.where(~in_set)[0]
            if len(non_selected) == 0:
                break
            
            # Find non-selected vertices with 0 conflicts
            zero_conf = non_selected[conflict_count[non_selected] == 0]
            if len(zero_conf) > 0:
                # Add one (random choice among zero-conflict)
                v = zero_conf[rng.integers(len(zero_conf))]
                in_set[v] = True
                current_size += 1
                for u in adj[v]:
                    conflict_count[u] += 1
                improved = True
                continue
            
            # Try 1-for-1 swaps: remove one selected, add one non-selected
            # Gain = 1 (add) - 1 (remove) = 0, but might enable more additions
            # Try 1-for-2 swaps: remove one selected, add two non-selected
            # This is expensive, so try a limited version
            
            # Try to find a swap that increases size
            # Remove selected vertex s, add non-selected vertex v
            # Net gain = 1 - 1 = 0, but we check if it enables more
            # For now, try simple 1-for-1 that doesn't decrease size
            # Actually, let's try: for each selected vertex, check if removing it
            # allows adding 2+ non-selected vertices
            
            # Simpler: try random 1-for-1 swaps and see if they help
            if len(selected_idx) > 0 and len(non_selected) > 0:
                # Try a few random swaps
                for _ in range(min(50, len(selected_idx))):
                    s = selected_idx[rng.integers(len(selected_idx))]
                    v = non_selected[rng.integers(len(non_selected))]
                    
                    # Check if v conflicts with any selected except s
                    v_conflicts = adj[v]
                    conflicts_with_selected = np.intersect1d(v_conflicts, selected_idx)
                    # If the only conflict is s, then swapping is valid
                    if len(conflicts_with_selected) == 1 and conflicts_with_selected[0] == s:
                        # Do the swap
                        in_set[s] = False
                        in_set[v] = True
                        # Update conflict counts
                        for u in adj[s]:
                            conflict_count[u] -= 1
                        for u in adj[v]:
                            conflict_count[u] += 1
                        # Update selected_idx
                        selected_idx = np.where(in_set)[0]
                        non_selected = np.where(~in_set)[0]
                        improved = True
                        break
        
        return in_set, current_size
    
    def simulated_annealing(in_set, max_iters=2000):
        """Simulated annealing for local search."""
        in_set = in_set.copy()
        current_size = int(in_set.sum())
        best_in_set = in_set.copy()
        best_size = current_size
        
        conflict_count = np.zeros(n, dtype=np.int32)
        selected_idx = np.where(in_set)[0]
        for v in selected_idx:
            conflict_count[adj[v]] += 1
        
        temp = 10.0
        cooling = 0.995
        
        iters = 0
        while iters < max_iters and time.time() < deadline - 5:
            iters += 1
            temp *= cooling
            if temp < 0.01:
                break
            
            # Try a random move
            move_type = rng.random()
            
            if move_type < 0.5:
                # Try to add a random non-selected vertex
                non_selected = np.where(~in_set)[0]
                if len(non_selected) == 0:
                    continue
                v = non_selected[rng.integers(len(non_selected))]
                nc = conflict_count[v]
                if nc == 0:
                    # Always accept
                    in_set[v] = True
                    current_size += 1
                    for u in adj[v]:
                        conflict_count[u] += 1
                    selected_idx = np.where(in_set)[0]
                    if current_size > best_size:
                        best_size = current_size
                        best_in_set = in_set.copy()
                else:
                    # Reject (can't add with conflicts)
                    pass
            elif move_type < 0.8:
                # Try to remove a random selected vertex
                if len(selected_idx) == 0:
                    continue
                s = selected_idx[rng.integers(len(selected_idx))]
                # Remove it
                in_set[s] = False
                current_size -= 1
                for u in adj[s]:
                    conflict_count[u] -= 1
                selected_idx = np.where(in_set)[0]
                non_selected = np.where(~in_set)[0]
                # Then try to add some non-selected vertices
                added = 0
                for _ in range(10):
                    if len(non_selected) == 0:
                        break
                    v = non_selected[rng.integers(len(non_selected))]
                    if conflict_count[v] == 0:
                        in_set[v] = True
                        current_size += 1
                        added += 1
                        for u in adj[v]:
                            conflict_count[u] += 1
                        selected_idx = np.where(in_set)[0]
                        non_selected = np.where(~in_set)[0]
                if current_size > best_size:
                    best_size = current_size
                    best_in_set = in_set.copy()
            else:
                # Try a 1-for-1 swap
                if len(selected_idx) == 0 or len(np.where(~in_set)[0]) == 0:
                    continue
                s = selected_idx[rng.integers(len(selected_idx))]
                non_selected = np.where(~in_set)[0]
                v = non_selected[rng.integers(len(non_selected))]
                # Check if swap is valid
                v_conflicts = adj[v]
                conflicts_with_selected = np.intersect1d(v_conflicts, selected_idx)
                if len(conflicts_with_selected) == 1 and conflicts_with_selected[0] == s:
                    in_set[s] = False
                    in_set[v] = True
                    for u in adj[s]:
                        conflict_count[u] -= 1
                    for u in adj[v]:
                        conflict_count[u] += 1
                    selected_idx = np.where(in_set)[0]
                    if current_size > best_size:
                        best_size = current_size
                        best_in_set = in_set.copy()
        
        return best_in_set, best_size
    
    best_size = 0
    best_in_set = np.zeros(n, dtype=bool)
    
    # Strategy 1: Greedy by degree (ascending)
    order_deg = np.argsort(degrees)
    in_set, size = greedy_select(order_deg)
    if size > best_size:
        best_size = size
        best_in_set = in_set.copy()
    in_set, size = local_search(in_set, max_iters=300)
    if size > best_size:
        best_size = size
        best_in_set = in_set.copy()
    
    # Strategy 2: Multiple random greedy restarts
    for trial in range(30):
        if time.time() > deadline - 30:
            break
        
        # Random ordering biased toward low degree
        weights = 1.0 / (degrees.astype(np.float64) + 1.0)
        weights /= weights.sum()
        order_rand = rng.choice(n, size=n, replace=False, p=weights)
        
        in_set, size = greedy_select(order_rand)
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
        
        in_set, size = local_search(in_set, max_iters=200)
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
    
    # Strategy 3: Simulated annealing on best solution
    if time.time() < deadline - 20:
        best_in_set, best_size = simulated_annealing(best_in_set, max_iters=3000)
    
    # Final local search polish
    best_in_set, best_size = local_search(best_in_set, max_iters=500)
    
    final_indices = np.where(best_in_set)[0]
    if len(final_indices) == 0:
        # Fallback: return type A vectors only
        return V[:22]
    
    return V[final_indices]
# EVOLVE-BLOCK-END
