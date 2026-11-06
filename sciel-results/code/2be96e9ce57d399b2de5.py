# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = time.time() + 140.0  # Safety margin

    def _greedy_numpy(order, counts_arr):
        chosen = []
        for x in order:
            if counts_arr[2 * x] >= 2:
                continue
            valid = True
            for y in chosen:
                s = x + y
                if counts_arr[s] >= 2:
                    valid = False
                    break
            if valid:
                chosen.append(x)
                counts_arr[2 * x] += 1
                for y in chosen[:-1]:
                    s = x + y
                    counts_arr[s] += 1
        return chosen

    # 1. Deterministic Greedy Baseline
    order1 = list(range(1, 1001))
    counts1 = np.zeros(2001, dtype=np.int32)
    best_set = _greedy_numpy(order1, counts1)
    best_size = len(best_set)
    
    # 2. Branch and Bound Search
    # State:
    # - selected: list of selected numbers
    # - counts: array of sum counts
    # - current_size: len(selected)
    # - best_size: global best
    
    def _branch_and_bound(initial_set, time_limit):
        # Initialize state
        selected = list(initial_set)
        counts = np.zeros(2001, dtype=np.int32)
        for i, x in enumerate(selected):
            counts[2*x] += 1
            for y in selected[i+1:]:
                counts[x+y] += 1
        
        current_size = len(selected)
        best_local_size = current_size
        best_local_set = selected.copy()
        
        # Candidates are numbers 1..1000 not in selected
        # We will iterate through candidates in a specific order (e.g., descending or random)
        # To make it a "search", we allow backtracking.
        # However, full backtracking on 1000 nodes is too slow.
        # We use a "Greedy with Random Restarts and Local Repair" approach which is a heuristic BnB.
        
        # Actually, let's implement a proper heuristic BnB for the "independent set" on the conflict graph.
        # Conflict graph: edge (u, v) exists if adding u to a set containing v (or vice versa)
        # causes a violation? No, the constraint is global.
        
        # Let's use the "Repair" strategy from the prompt's inspiration.
        # 1. Start with a set.
        # 2. If valid, try to add elements.
        # 3. If invalid, remove elements to repair.
        
        # We will simulate a search tree where at each step we pick a candidate to add.
        # If adding causes violation, we branch:
        #   Branch A: Do not add candidate.
        #   Branch B: Add candidate AND remove one conflicting element (Repair).
        
        # To keep it fast, we limit the depth of repair or the number of branches.
        
        def _is_valid(counts):
            return np.max(counts) <= 2
        
        def _try_add(x, selected, counts):
            # Returns True if x can be added, False otherwise
            # Also updates counts if added? No, let's check first.
            if counts[2*x] >= 2:
                return False
            for y in selected:
                if counts[x+y] >= 2:
                    return False
            return True

        def _add(x, selected, counts):
            counts[2*x] += 1
            for y in selected:
                counts[x+y] += 1
            selected.append(x)

        def _remove(x, selected, counts):
            counts[2*x] -= 1
            new_selected = []
            for y in selected:
                if y != x:
                    new_selected.append(y)
                    counts[x+y] -= 1
            return new_selected

        # Heuristic BnB
        # We maintain a stack of states.
        # State: (selected, counts, candidates, depth)
        # This might be too memory intensive.
        
        # Alternative: Iterative Deepening A*? Or just a smart local search with "repair" moves.
        # The prompt suggests "Model ... as MIS ... Use fast vectorized BnB".
        # A true BnB on 1000 nodes is infeasible. 
        # But a BnB on a *subset* or a *heuristic* BnB is possible.
        
        # Let's interpret "BnB" as a guided search that prunes branches where the upper bound
        # (current_size + remaining_candidates) <= best_size.
        
        # Since the graph is dense (B2[2] sets are sparse), the "remaining candidates" bound is loose.
        # We need a tighter bound.
        
        # Let's stick to the "Repair" local search but structured as a tree search.
        # We will perform a DFS with a depth limit.
        
        max_depth = 20 # Max number of "decisions" (adds or repairs) to explore in one branch
        
        def _search(selected, counts, candidates, depth, current_size):
            nonlocal best_local_size, best_local_set
            
            if time.time() > time_limit:
                return
            
            if current_size > best_local_size:
                best_local_size = current_size
                best_local_set = selected.copy()
                
            if depth >= max_depth:
                return
                
            if not candidates:
                return
                
            # Pruning: If current_size + len(candidates) <= best_local_size, prune
            # This is a weak bound, but better than nothing.
            if current_size + len(candidates) <= best_local_size:
                return

            # Pick a candidate to decide on
            # Heuristic: Pick the one with highest "value" (e.g., smallest sum counts)
            # Or just pick randomly? Random is faster.
            idx = np.random.randint(len(candidates))
            x = candidates[idx]
            new_candidates = candidates[:idx] + candidates[idx+1:]
            
            # Option 1: Include x
            if _try_add(x, selected, counts):
                # Add x
                _add(x, selected, counts)
                _search(selected, counts, new_candidates, depth + 1, current_size + 1)
                # Backtrack
                _remove(x, selected, counts)
            else:
                # Cannot add x directly. 
                # Option 1a: Skip x
                _search(selected, counts, new_candidates, depth + 1, current_size)
                
                # Option 1b: Repair - Remove a conflicting element and add x
                # Find conflicts
                conflicts = []
                if counts[2*x] >= 2:
                    # Conflicts with pairs (a, b) such that a+b=2x? No, 2x is just x+x.
                    # If counts[2x] >= 2, it means there are already 2 pairs summing to 2x.
                    # Since x is not in selected, these pairs don't involve x.
                    # So adding x makes counts[2x] become 3.
                    # To repair, we must remove one of the pairs.
                    # This is complex.
                    pass
                else:
                    for y in selected:
                        if counts[x+y] >= 2:
                            conflicts.append(y)
                
                if conflicts:
                    # Try removing one conflict
                    # Pick a random conflict
                    y = conflicts[np.random.randint(len(conflicts))]
                    # Remove y
                    new_selected = _remove(y, selected, counts)
                    # Now try to add x
                    if _try_add(x, new_selected, counts):
                        _add(x, new_selected, counts)
                        _search(new_selected, counts, new_candidates, depth + 1, current_size) # Size unchanged
                        _remove(x, new_selected, counts)
            
            # Option 2: Exclude x (already covered by Option 1a if include fails, but we should always consider exclusion)
            # Actually, the logic above handles "Include" and "Exclude".
            # If Include fails, we tried Exclude and Repair.
            # If Include succeeds, we tried Include. We should also try Exclude?
            # Yes, standard BnB branches on Include/Exclude.
            
            # Let's restructure:
            # Branch 1: Include x (if valid)
            # Branch 2: Exclude x
            
            # My current code:
            # If valid: Include
            # Else: Exclude, Repair
            
            # This misses "Exclude" when "Include" is valid.
            # We should add that.
            
            if _try_add(x, selected, counts):
                 # We already did Include. Now do Exclude.
                 # But we backtracked. So we can do Exclude now.
                 _search(selected, counts, new_candidates, depth + 1, current_size)

        # Run search
        # Candidates are all numbers not in selected
        candidates = [x for x in range(1, 1001) if x not in set(selected)]
        
        # Shuffle candidates to break symmetry
        np.random.shuffle(candidates)
        
        _search(selected, counts, candidates, 0, current_size)
        
        return best_local_set, best_local_size

    # Run BnB from the greedy solution
    # We might run multiple instances with different shuffles
    
    num_restarts = 5
    time_per_restart = (deadline - time.time()) / num_restarts
    
    best_global_set = best_set
    best_global_size = best_size
    
    rng = np.random.default_rng(42)
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        time_limit = time.time() + time_per_restart
        
        # Perturb the initial set slightly?
        # Or just run on the greedy set with a different shuffle.
        # The shuffle in _search is hardcoded to np.random. 
        # We should pass in a seed or shuffle outside.
        
        # Let's create a perturbed start set for variety
        start_set = best_global_set.copy()
        if i > 0:
            # Remove a few random elements
            for _ in range(5):
                if start_set:
                    idx = rng.integers(0, len(start_set))
                    start_set.pop(idx)
        
        run_set, run_size = _branch_and_bound(start_set, time_limit)
        
        if run_size > best_global_size:
            best_global_size = run_size
            best_global_set = run_set
    
    return best_global_set

# EVOLVE-BLOCK-END
