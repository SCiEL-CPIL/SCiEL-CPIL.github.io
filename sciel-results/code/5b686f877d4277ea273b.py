<<<<<<< SEARCH
import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Generate candidates with squared norm 4
    candidates = []
    
    # Type 1: One 2, rest 0
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four 1s, rest 0
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute conflict matrix
    # M[i, j] = candidates[i] . candidates[j]
    # Conflict if M[i, j] > 2
    M = candidates @ candidates.T
    C = (M > 2).astype(np.int8)
    # Set diagonal to 0 (no self-conflict in logic, but we don't pick same node)
    np.fill_diagonal(C, 0)
    
    # Precompute degrees for sorting (optional, but helpful for greedy)
    degrees = C.sum(axis=1)
    
    best_set = []
    rng = np.random.default_rng(42)
    
    # Helper: Greedy completion from a starting set
    def greedy_complete(start_indices, order=None):
        if order is None:
            # Sort by degree (ascending) to pick "easy" nodes first?
            # Or random?
            # Let's use random for ILS diversity
            order = rng.permutation(N)
            
        selected_mask = np.zeros(N, dtype=bool)
        selected = []
        
        # Initialize counts based on start_indices
        if start_indices:
            selected_mask[start_indices] = True
            selected = list(start_indices)
            # counts[j] = sum of C[i, j] for i in start_indices
            counts = C[start_indices, :].sum(axis=0).astype(np.int32)
        else:
            counts = np.zeros(N, dtype=np.int32)
            
        for idx in order:
            if selected_mask[idx]:
                continue
            if counts[idx] > 0:
                continue
            
            # Add idx
            selected.append(idx)
            selected_mask[idx] = True
            counts += C[idx, :]
            
        return selected

    # Initial greedy passes with different orderings
    for _ in range(10):
        if time.time() > deadline:
            break
        # Try degree-sorted order (lowest degree first)
        order = np.argsort(degrees)
        s = greedy_complete([], order)
        if len(s) > len(best_set):
            best_set = s
            
    # Iterated Local Search
    # Strategy: Remove k random nodes, re-greedy
    k_removal = 2
    while time.time() < deadline:
        if not best_set:
            break
            
        # Perturb: remove k random nodes
        start_indices = list(best_set)
        if len(start_indices) <= k_removal:
            # If set is too small, just restart
            s = greedy_complete([])
        else:
            # Remove k random nodes
            to_remove = rng.choice(len(start_indices), size=k_removal, replace=False)
            to_remove_indices = [start_indices[i] for i in to_remove]
            start_indices = [x for x in start_indices if x not in set(to_remove_indices)]
            
            # Re-greedy
            # Use random order for re-greedy to escape local optima
            s = greedy_complete(start_indices)
            
        if len(s) > len(best_set):
            best_set = s
            # Optional: reset k_removal or adapt?
            # For now, keep fixed.
            
    if not best_set:
        # Fallback
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return candidates[best_set].astype(np.int64)
=======
import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Generate candidates with squared norm 4
    candidates = []
    
    # Type 1: One 2, rest 0
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four 1s, rest 0
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute conflict matrix
    # M[i, j] = candidates[i] . candidates[j]
    # Conflict if M[i, j] > 2
    M = candidates @ candidates.T
    C = (M > 2).astype(np.int8)
    # Set diagonal to 0 (no self-conflict in logic, but we don't pick same node)
    np.fill_diagonal(C, 0)
    
    # Precompute degrees for sorting (optional, but helpful for greedy)
    degrees = C.sum(axis=1)
    
    best_set = []
    rng = np.random.default_rng(42)
    
    # Helper: Greedy completion from a starting set
    def greedy_complete(start_indices, order=None):
        if order is None:
            # Sort by degree (ascending) to pick "easy" nodes first?
            # Or random?
            # Let's use random for ILS diversity
            order = rng.permutation(N)
            
        selected_mask = np.zeros(N, dtype=bool)
        selected = []
        
        # Initialize counts based on start_indices
        if start_indices:
            selected_mask[start_indices] = True
            selected = list(start_indices)
            # counts[j] = sum of C[i, j] for i in start_indices
            counts = C[start_indices, :].sum(axis=0).astype(np.int32)
        else:
            counts = np.zeros(N, dtype=np.int32)
            
        for idx in order:
            if selected_mask[idx]:
                continue
            if counts[idx] > 0:
                continue
            
            # Add idx
            selected.append(idx)
            selected_mask[idx] = True
            counts += C[idx, :]
            
        return selected

    # Initial greedy passes with different orderings
    for _ in range(10):
        if time.time() > deadline:
            break
        # Try degree-sorted order (lowest degree first)
        order = np.argsort(degrees)
        s = greedy_complete([], order)
        if len(s) > len(best_set):
            best_set = s
            
    # Simulated Annealing on the Independent Set
    # Objective: Maximize number of selected nodes (size of independent set)
    # We model this as: start with a random independent set, then try to add/remove/swap nodes.
    
    if best_set:
        current_set = set(best_set)
        current_size = len(best_set)
        best_set_sa = list(best_set)
        best_size_sa = current_size
        
        # Temperature schedule
        T = 1.0
        T_min = 0.001
        cooling = 0.995
        
        # Precompute neighbor lists for faster access? 
        # C is int8, N is small enough (~3000?) so direct access is fine.
        
        while time.time() < deadline:
            if T < T_min:
                # Reset temperature if stuck? Or break?
                # Let's try to reset if we haven't improved for a while
                # For now, simple break or restart
                break
                
            # Generate a neighbor state
            # Move types:
            # 1. Add a node (if it doesn't conflict with current set)
            # 2. Remove a node
            # 3. Swap a node (remove one, add another)
            
            move_type = rng.random()
            
            if move_type < 0.4:
                # Try to add a node
                # Find nodes that don't conflict with current set
                # Conflicts: for each node in current_set, its neighbors are forbidden.
                # Forbidden = union of C[i, :] for i in current_set
                # This is expensive if done every step.
                # Optimization: Maintain a "conflict_count" array for all nodes.
                # conflict_count[j] = number of nodes in current_set that conflict with j.
                # If conflict_count[j] == 0, j can be added.
                
                # Let's maintain this array globally for SA
                if not hasattr(construct, '_sa_conflict_count'):
                    construct._sa_conflict_count = None
                    construct._sa_current_set = None
                    construct._sa_best_set = None
                    construct._sa_best_size = 0
                    construct._sa_T = 1.0
                    construct._sa_T_min = 0.001
                    construct._sa_cooling = 0.995
                    construct._sa_rng = np.random.default_rng(42)
                    construct._sa_deadline = deadline
                    construct._sa_best_set_initial = list(best_set)
                    construct._sa_best_size_initial = len(best_set)
                    
                # Initialize SA state if first call or reset
                if construct._sa_conflict_count is None:
                    construct._sa_conflict_count = np.zeros(N, dtype=np.int32)
                    for idx in current_set:
                        construct._sa_conflict_count += C[idx, :]
                    construct._sa_current_set = set(current_set)
                    construct._sa_best_set = list(best_set_sa)
                    construct._sa_best_size = best_size_sa
                    construct._sa_T = T
                    construct._sa_T_min = T_min
                    construct._sa_cooling = cooling
                    construct._sa_rng = rng
                    construct._sa_deadline = deadline
                    construct._sa_best_set_initial = list(best_set)
                    construct._sa_best_size_initial = len(best_set)
                
                # Use stored state
                cc = construct._sa_conflict_count
                curr_set = construct._sa_current_set
                T = construct._sa_T
                
                # Find candidates to add: nodes with cc[j] == 0 and j not in curr_set
                # We can just pick a random node and check
                # Or better, pick from non-conflicting nodes efficiently?
                # For now, random pick and check is O(1)
                
                # To make it faster, let's just pick a random node
                candidate_idx = rng.integers(0, N)
                
                if candidate_idx in curr_set:
                    # Already in set, skip
                    continue
                    
                if cc[candidate_idx] > 0:
                    # Conflicts, skip
                    continue
                
                # Valid move: Add candidate_idx
                new_size = len(curr_set) + 1
                delta = 1 # Improvement
                
                # Accept (always accept improvement)
                curr_set.add(candidate_idx)
                cc += C[candidate_idx, :]
                
                if new_size > construct._sa_best_size:
                    construct._sa_best_size = new_size
                    construct._sa_best_set = list(curr_set)
                
                # Cool down
                construct._sa_T *= construct._sa_cooling
                
            elif move_type < 0.7:
                # Try to remove a node
                if not curr_set:
                    continue
                    
                # Pick a random node to remove
                # Converting set to list every time is slow. 
                # Let's keep a list representation in sync? 
                # Or just sample from the set.
                # Python sets don't support random indexing.
                # Let's use a list for the current set in SA state.
                
                # Refactor: Use list for curr_set in SA state
                # If we change to list, we need to update the logic above.
                # Let's restart SA with a cleaner implementation below.
                pass # Skip this branch for now, will rewrite
            
            else:
                # Try swap
                pass # Skip for now

            # Update global best
            if construct._sa_best_size > len(best_set_sa):
                best_set_sa = list(construct._sa_best_set)
                best_size_sa = construct._sa_best_size
            if best_size_sa > len(best_set):
                best_set = list(construct._sa_best_set)
                
            # Check time
            if time.time() > deadline:
                break
                
            # If temperature is too low, restart SA with new random start
            if T < T_min:
                # Restart
                # Pick a random independent set of similar size?
                # Or just restart from best?
                # Let's restart from best set with new temperature
                construct._sa_conflict_count = np.zeros(N, dtype=np.int32)
                curr_set = set(construct._sa_best_set)
                for idx in curr_set:
                    construct._sa_conflict_count += C[idx, :]
                construct._sa_current_set = curr_set
                construct._sa_T = 1.0
                construct._sa_best_set = list(construct._sa_best_set)
                construct._sa_best_size = construct._sa_best_size
                
    return candidates[best_set].astype(np.int64)
>>>>>>> REPLACE

The previous attempt was a bit messy with state management inside the function. Let's write a cleaner version of the Simulated Annealing approach that is structurally distinct from the Greedy ILS.

The key structural change is moving from **Greedy + Perturbation** to **Simulated Annealing** on the Independent Set problem.
Instead of maintaining a hard constraint (independent set) and only moving to valid states, we can use a penalty-based objective or simply stick to valid states (which is easier for Independent Set) but allow "bad" moves (removing nodes) to escape local optima where we are stuck with a small set but can't add any more.

Actually, the previous code snippet I wrote in the SEARCH block was incomplete/buggy. I will provide a clean, complete replacement for the entire `construct` function that implements a proper Simulated Annealing solver for the Maximum Independent Set on the conflict graph.

**Plan:**
1.  Generate candidates (same as before).
2.  Build Conflict Matrix `C`.
3.  Initialize a random independent set using a greedy pass.
4.  Run Simulated Annealing:
    *   State: `selected_mask` (boolean array), `conflict_count` (int array, number of conflicts for each node with current set).
    *   Objective: Maximize `sum(selected_mask)`.
    *   Moves:
        *   **Add**: Pick a random node `i` not in set. If `conflict_count[i] == 0`, it's a valid add. (Always accept).
        *   **Remove**: Pick a random node `i` in set. Remove it. (Accept with probability `exp(-delta/T)` where `delta = -1`, so `exp(1/T)`? No, standard SA minimizes energy. Here we maximize size. So `delta_obj = -1`. `accept_prob = exp(delta_obj / T)` is for minimization. For maximization, we accept if `delta > 0` or with prob `exp(delta/T)` where `delta` is the change in objective. If we remove, `delta = -1`. `exp(-1/T)`. This allows us to remove nodes to escape local maxima).
        *   **Swap**: Pick `i` in set, `j` not in set. If `j` conflicts only with `i` (i.e., `conflict_count[j] == 1` and `C[i,j]==1`), we can swap. `delta = 0`. Always accept (or accept with prob 1). This moves us to another local optimum of the same size.
    *   Cooling: Reduce T.

This is a true stochastic local search that can escape local optima by accepting "worse" states (smaller sets) temporarily.

