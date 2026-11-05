<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    n = 256
    elements = list(range(1, 257))
    
    # Precompute conflict matrix: conflict[i][j] = True if (a*b) % (a+b) == 0
    # where a = elements[i], b = elements[j]
    # We want to find maximum independent set in this graph.
    # Conflict graph: edge between i,j if they CANNOT both be in the set.
    
    # Use bitset representation for speed
    # conflict_mask[i] is a bitset where bit j is set if i and j conflict
    
    # First, let's build the conflict structure
    # For efficiency, we'll work with numpy arrays
    
    a = np.array(elements, dtype=np.int64)
    
    # Build conflict matrix as a set of pairs
    # conflict[i][j] = 1 if a[i]*a[j] % (a[i]+a[j]) == 0
    # This is O(n^2) = 256^2 = 65536 checks, very fast
    
    conflict = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i+1, n):
            ai = a[i]
            aj = a[j]
            if (ai * aj) % (ai + aj) == 0:
                conflict[i][j] = 1
                conflict[j][i] = 1
    
    # Now we need to find maximum independent set.
    # This is equivalent to maximum clique in the complement graph,
    # or we can formulate as MILP:
    # Maximize sum(x_i)
    # Subject to: x_i + x_j <= 1 for all (i,j) in conflict edges
    #             x_i in {0,1}
    
    # Count conflict edges
    conflict_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if conflict[i][j]:
                conflict_edges.append((i, j))
    
    m = len(conflict_edges)
    
    # Build MILP
    # c = -1 for all variables (we minimize -sum(x))
    c = -np.ones(n)
    
    # Constraints: x_i + x_j <= 1 for each conflict edge
    if m > 0:
        A = np.zeros((m, n), dtype=np.int8)
        for idx, (i, j) in enumerate(conflict_edges):
            A[idx, i] = 1
            A[idx, j] = 1
        lb = np.full(m, 0, dtype=np.int8)
        ub = np.full(m, 1, dtype=np.int8)
        constraints = LinearConstraint(A, lb, ub)
    else:
        constraints = LinearConstraint(np.zeros((1, n)), 0, 1)
    
    integrality = np.ones(n, dtype=np.int8)
    bounds = Bounds(0, 1)
    
    # Try MILP
    try:
        result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if result.status == 0 and result.x is not None:
            selected = [elements[i] for i in range(n) if result.x[i] > 0.5]
            if len(selected) > 0:
                # Verify the solution
                valid = True
                sel_set = set(selected)
                for idx_i in range(len(selected)):
                    for idx_j in range(idx_i+1, len(selected)):
                        ai = selected[idx_i]
                        aj = selected[idx_j]
                        if (ai * aj) % (ai + aj) == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return selected
    except Exception:
        pass
    
    # Fallback: greedy approach with local search
    # Greedy: pick elements in order of fewest conflicts first
    degree = np.sum(conflict, axis=1)
    
    # Sort by degree (ascending) - pick low-degree vertices first
    order = np.argsort(degree)
    
    selected = []
    selected_set = set()
    for idx in order:
        if time.time() > deadline - 5:
            break
        # Check if this vertex conflicts with any already selected
        can_add = True
        for sel_idx in selected_set:
            if conflict[idx, sel_idx]:
                can_add = False
                break
        if can_add:
            selected.append(elements[idx])
            selected_set.add(idx)
    
    # Local search: try to improve by swapping
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            current_set = set(selected)
            current_idx_set = selected_set.copy()
            
            # Try adding one element
            for idx in range(n):
                if idx in current_idx_set:
                    continue
                can_add = True
                for sel_idx in current_idx_set:
                    if conflict[idx, sel_idx]:
                        can_add = False
                        break
                if can_add:
                    selected.append(elements[idx])
                    current_idx_set.add(idx)
                    improved = True
                    break
            
            # Try swapping: remove one, add another that was previously blocked
            if not improved:
                for rem_idx in list(current_idx_set):
                    # Try to add something that conflicts with rem_idx
                    for add_idx in range(n):
                        if add_idx in current_idx_set:
                            continue
                        # Check if add_idx only conflicts with rem_idx among selected
                        only_conflicts_rem = True
                        for sel_idx in current_idx_set:
                            if sel_idx != rem_idx and conflict[add_idx, sel_idx]:
                                only_conflicts_rem = False
                                break
                        if only_conflicts_rem:
                            # Do the swap
                            selected.remove(elements[rem_idx])
                            current_idx_set.remove(rem_idx)
                            selected.append(elements[add_idx])
                            current_idx_set.add(add_idx)
                            improved = True
                            break
                    if improved:
                        break
    
    if len(selected) == 0:
        return [1]
    
    return selected
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    n = 256
    elements = list(range(1, 257))
    
    # Precompute conflict matrix
    a = np.array(elements, dtype=np.int64)
    
    conflict = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i+1, n):
            ai = a[i]
            aj = a[j]
            if (ai * aj) % (ai + aj) == 0:
                conflict[i][j] = 1
                conflict[j][i] = 1
    
    degree = np.sum(conflict, axis=1)
    
    # Try MILP first
    conflict_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if conflict[i][j]:
                conflict_edges.append((i, j))
    
    m = len(conflict_edges)
    
    c = -np.ones(n)
    
    if m > 0:
        A = np.zeros((m, n), dtype=np.int8)
        for idx, (i, j) in enumerate(conflict_edges):
            A[idx, i] = 1
            A[idx, j] = 1
        lb = np.full(m, 0, dtype=np.int8)
        ub = np.full(m, 1, dtype=np.int8)
        constraints = LinearConstraint(A, lb, ub)
    else:
        constraints = LinearConstraint(np.zeros((1, n)), 0, 1)
    
    integrality = np.ones(n, dtype=np.int8)
    bounds = Bounds(0, 1)
    
    best_solution = []
    best_size = 0
    
    # Try MILP
    try:
        result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if result.status == 0 and result.x is not None:
            selected = [elements[i] for i in range(n) if result.x[i] > 0.5]
            if len(selected) > 0:
                valid = True
                for idx_i in range(len(selected)):
                    for idx_j in range(idx_i+1, len(selected)):
                        ai = selected[idx_i]
                        aj = selected[idx_j]
                        if (ai * aj) % (ai + aj) == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_solution = selected
                    best_size = len(selected)
    except Exception:
        pass
    
    # Multi-start greedy with local search
    # Use different orderings to explore diverse basins
    
    # Strategy 1: degree-sorted greedy
    if time.time() < deadline - 15:
        order = np.argsort(degree)
        selected_idx = []
        selected_set = set()
        for idx in order:
            if time.time() > deadline - 10:
                break
            can_add = True
            for sel_idx in selected_set:
                if conflict[idx, sel_idx]:
                    can_add = False
                    break
            if can_add:
                selected_idx.append(idx)
                selected_set.add(idx)
        
        # Local search refinement
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            current_idx_set = set(selected_idx)
            
            # Try adding one element
            for idx in range(n):
                if idx in current_idx_set:
                    continue
                can_add = True
                for sel_idx in current_idx_set:
                    if conflict[idx, sel_idx]:
                        can_add = False
                        break
                if can_add:
                    selected_idx.append(idx)
                    current_idx_set.add(idx)
                    improved = True
                    break
            
            # Try 1-for-1 swap
            if not improved:
                for rem_idx in list(current_idx_set):
                    for add_idx in range(n):
                        if add_idx in current_idx_set:
                            continue
                        only_conflicts_rem = True
                        for sel_idx in current_idx_set:
                            if sel_idx != rem_idx and conflict[add_idx, sel_idx]:
                                only_conflicts_rem = False
                                break
                        if only_conflicts_rem:
                            selected_idx.remove(rem_idx)
                            selected_idx.append(add_idx)
                            current_idx_set.remove(rem_idx)
                            current_idx_set.add(add_idx)
                            improved = True
                            break
                    if improved:
                        break
            
            # Try 2-for-1: remove two, add one
            if not improved and time.time() < deadline - 12:
                rem_list = list(current_idx_set)
                for i in range(len(rem_list)):
                    for j in range(i+1, len(rem_list)):
                        r1 = rem_list[i]
                        r2 = rem_list[j]
                        temp_set = current_idx_set - {r1, r2}
                        for add_idx in range(n):
                            if add_idx in current_idx_set:
                                continue
                            can_add = True
                            for sel_idx in temp_set:
                                if conflict[add_idx, sel_idx]:
                                    can_add = False
                                    break
                            if can_add:
                                # Do the swap: remove r1, r2; add add_idx
                                selected_idx.remove(r1)
                                selected_idx.remove(r2)
                                selected_idx.append(add_idx)
                                current_idx_set.remove(r1)
                                current_idx_set.remove(r2)
                                current_idx_set.add(add_idx)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        if len(selected_idx) > best_size:
            best_solution = [elements[i] for i in selected_idx]
            best_size = len(selected_idx)
    
    # Strategy 2: Randomized greedy with multiple restarts
    if time.time() < deadline - 15:
        random.seed(42)
        for restart in range(10):
            if time.time() > deadline - 10:
                break
            
            # Random ordering with bias toward low-degree vertices
            weights = 1.0 / (1.0 + degree.astype(np.float64))
            total_w = np.sum(weights)
            probs = weights / total_w
            order = np.random.choice(n, size=n, replace=False, p=probs)
            
            selected_idx = []
            selected_set = set()
            for idx in order:
                if time.time() > deadline - 10:
                    break
                can_add = True
                for sel_idx in selected_set:
                    if conflict[idx, sel_idx]:
                        can_add = False
                        break
                if can_add:
                    selected_idx.append(idx)
                    selected_set.add(idx)
            
            # Local search refinement
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                current_idx_set = set(selected_idx)
                
                # Try adding one element
                for idx in range(n):
                    if idx in current_idx_set:
                        continue
                    can_add = True
                    for sel_idx in current_idx_set:
                        if conflict[idx, sel_idx]:
                            can_add = False
                            break
                    if can_add:
                        selected_idx.append(idx)
                        current_idx_set.add(idx)
                        improved = True
                        break
                
                # Try 1-for-1 swap
                if not improved:
                    for rem_idx in list(current_idx_set):
                        for add_idx in range(n):
                            if add_idx in current_idx_set:
                                continue
                            only_conflicts_rem = True
                            for sel_idx in current_idx_set:
                                if sel_idx != rem_idx and conflict[add_idx, sel_idx]:
                                    only_conflicts_rem = False
                                    break
                            if only_conflicts_rem:
                                selected_idx.remove(rem_idx)
                                selected_idx.append(add_idx)
                                current_idx_set.remove(rem_idx)
                                current_idx_set.add(add_idx)
                                improved = True
                                break
                        if improved:
                            break
                
                # Try 2-for-1
                if not improved and time.time() < deadline - 12:
                    rem_list = list(current_idx_set)
                    for i in range(len(rem_list)):
                        for j in range(i+1, len(rem_list)):
                            r1 = rem_list[i]
                            r2 = rem_list[j]
                            temp_set = current_idx_set - {r1, r2}
                            for add_idx in range(n):
                                if add_idx in current_idx_set:
                                    continue
                                can_add = True
                                for sel_idx in temp_set:
                                    if conflict[add_idx, sel_idx]:
                                        can_add = False
                                        break
                                if can_add:
                                    selected_idx.remove(r1)
                                    selected_idx.remove(r2)
                                    selected_idx.append(add_idx)
                                    current_idx_set.remove(r1)
                                    current_idx_set.remove(r2)
                                    current_idx_set.add(add_idx)
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                
                # Try 1-for-2: remove one, add two
                if not improved and time.time() < deadline - 12:
                    for rem_idx in list(current_idx_set):
                        temp_set = current_idx_set - {rem_idx}
                        # Find two elements that can both be added
                        add_candidates = []
                        for add1 in range(n):
                            if add1 in current_idx_set:
                                continue
                            can_add1 = True
                            for sel_idx in temp_set:
                                if conflict[add1, sel_idx]:
                                    can_add1 = False
                                    break
                            if can_add1:
                                add_candidates.append(add1)
                        
                        for i in range(len(add_candidates)):
                            for j in range(i+1, len(add_candidates)):
                                add1 = add_candidates[i]
                                add2 = add_candidates[j]
                                if conflict[add1, add2]:
                                    continue
                                # Check both can be added
                                can_add_both = True
                                for sel_idx in temp_set:
                                    if conflict[add1, sel_idx] or conflict[add2, sel_idx]:
                                        can_add_both = False
                                        break
                                if can_add_both:
                                    selected_idx.remove(rem_idx)
                                    selected_idx.append(add1)
                                    selected_idx.append(add2)
                                    current_idx_set.remove(rem_idx)
                                    current_idx_set.add(add1)
                                    current_idx_set.add(add2)
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
            
            if len(selected_idx) > best_size:
                best_solution = [elements[i] for i in selected_idx]
                best_size = len(selected_idx)
    
    # Strategy 3: Simulated annealing-style local search
    if time.time() < deadline - 15 and best_size > 0:
        current_idx_set = set()
        for val in best_solution:
            current_idx_set.add(elements.index(val))
        
        best_local = len(current_idx_set)
        best_local_set = current_idx_set.copy()
        
        # Hill climbing with random restarts within the local search
        for iteration in range(20):
            if time.time() > deadline - 10:
                break
            
            improved = False
            for _ in range(100):  # Inner loop iterations
                if time.time() > deadline - 10:
                    break
                
                # Try random moves
                move_type = random.choice(['add', 'swap11', 'swap21', 'swap12'])
                
                if move_type == 'add':
                    candidates = [i for i in range(n) if i not in current_idx_set]
                    if candidates:
                        idx = random.choice(candidates)
                        can_add = True
                        for sel_idx in current_idx_set:
                            if conflict[idx, sel_idx]:
                                can_add = False
                                break
                        if can_add:
                            current_idx_set.add(idx)
                            improved = True
                
                elif move_type == 'swap11':
                    if current_idx_set:
                        rem = random.choice(list(current_idx_set))
                        candidates = [i for i in range(n) if i not in current_idx_set]
                        if candidates:
                            add = random.choice(candidates)
                            only_conflicts_rem = True
                            for sel_idx in current_idx_set:
                                if sel_idx != rem and conflict[add, sel_idx]:
                                    only_conflicts_rem = False
                                    break
                            if only_conflicts_rem:
                                current_idx_set.remove(rem)
                                current_idx_set.add(add)
                                improved = True
                
                elif move_type == 'swap21':
                    rem_list = list(current_idx_set)
                    if len(rem_list) >= 2:
                        r1, r2 = random.sample(rem_list, 2)
                        candidates = [i for i in range(n) if i not in current_idx_set]
                        if candidates:
                            add = random.choice(candidates)
                            temp_set = current_idx_set - {r1, r2}
                            can_add = True
                            for sel_idx in temp_set:
                                if conflict[add, sel_idx]:
                                    can_add = False
                                    break
                            if can_add:
                                current_idx_set.remove(r1)
                                current_idx_set.remove(r2)
                                current_idx_set.add(add)
                                # This reduces size, so only accept if it enables future improvements
                                # For now, skip this in hill climbing
                
                elif move_type == 'swap12':
                    if current_idx_set:
                        rem = random.choice(list(current_idx_set))
                        candidates = [i for i in range(n) if i not in current_idx_set]
                        if len(candidates) >= 2:
                            add1, add2 = random.sample(candidates, 2)
                            if not conflict[add1, add2]:
                                temp_set = current_idx_set - {rem}
                                can_add_both = True
                                for sel_idx in temp_set:
                                    if conflict[add1, sel_idx] or conflict[add2, sel_idx]:
                                        can_add_both = False
                                        break
                                if can_add_both:
                                    current_idx_set.remove(rem)
                                    current_idx_set.add(add1)
                                    current_idx_set.add(add2)
                                    improved = True
                
                if improved:
                    break
            
            if len(current_idx_set) > best_local:
                best_local = len(current_idx_set)
                best_local_set = current_idx_set.copy()
            
            # Random restart: perturb the current solution
            if random.random() < 0.3 and best_local_set:
                perturb_idx = random.choice(list(best_local_set))
                current_idx_set = best_local_set - {perturb_idx}
            else:
                current_idx_set = best_local_set.copy()
        
        if best_local > best_size:
            best_solution = [elements[i] for i in best_local_set]
            best_size = best_local
    
    if len(best_solution) == 0:
        return [1]
    
    # Final verification
    valid = True
    for i in range(len(best_solution)):
        for j in range(i+1, len(best_solution)):
            if (best_solution[i] * best_solution[j]) % (best_solution[i] + best_solution[j]) == 0:
                valid = False
                break
        if not valid:
            break
    
    if valid:
        return best_solution
    
    # Fallback: return a valid small set
    return [1]
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
