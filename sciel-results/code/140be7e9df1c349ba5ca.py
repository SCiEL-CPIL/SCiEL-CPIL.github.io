import numpy as np
from math import lcm, gcd
import time
import sys
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 256
    candidates = list(range(2, N + 1))
    n = len(candidates)
    
    # Precompute conflict bitsets for fast checking
    # conflict_bits[i] is a bitset of nodes that conflict with node i
    conflict_bits = [0] * n
    for i in range(n):
        a = candidates[i]
        for j in range(i + 1, n):
            b = candidates[j]
            if lcm(a, b) <= 256:
                conflict_bits[i] |= (1 << j)
                conflict_bits[j] |= (1 << i)
    
    # Helper: check if adding node idx to set (represented as bitmask) is valid
    def can_add(idx, selected_mask):
        # Check if idx conflicts with any selected node
        # We need to check if any bit in conflict_bits[idx] is set in selected_mask
        return (conflict_bits[idx] & selected_mask) == 0
    
    # Helper: compute score from selected indices
    def compute_score(selected_indices):
        total = Fraction(0, 1)
        for idx in selected_indices:
            total += Fraction(1, candidates[idx])
        return total
    
    def compute_score_float(selected_indices):
        return sum(1.0 / candidates[idx] for idx in selected_indices)
    
    # Greedy initialization with different orderings
    def greedy_init(order):
        selected_mask = 0
        selected_indices = []
        for idx in order:
            if can_add(idx, selected_mask):
                selected_mask |= (1 << idx)
                selected_indices.append(idx)
        return selected_indices
    
    # Local search: 1-swap, 1-add, 2-for-1, 1-for-2
    def local_search(initial_indices, time_limit):
        selected = list(initial_indices)
        selected_set = set(selected)
        selected_mask = 0
        for idx in selected:
            selected_mask |= (1 << idx)
        
        current_score = compute_score_float(selected)
        improved = True
        t0 = time.time()
        
        while improved and time.time() < time_limit:
            improved = False
            
            # Try 1-add: add any node that fits
            for idx in range(n):
                if idx in selected_set:
                    continue
                if can_add(idx, selected_mask):
                    new_score = current_score + 1.0 / candidates[idx]
                    if new_score > current_score + 1e-12:
                        selected.append(idx)
                        selected_set.add(idx)
                        selected_mask |= (1 << idx)
                        current_score = new_score
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try 1-swap: remove one, add one
            best_swap_gain = 0
            best_swap = None
            for rem in list(selected):
                rem_score = 1.0 / candidates[rem]
                temp_mask = selected_mask & ~(1 << rem)
                for add in range(n):
                    if add in selected_set:
                        continue
                    if can_add(add, temp_mask):
                        gain = (1.0 / candidates[add]) - rem_score
                        if gain > best_swap_gain + 1e-12:
                            best_swap_gain = gain
                            best_swap = (rem, add)
            
            if best_swap:
                rem, add = best_swap
                selected.remove(rem)
                selected_set.discard(rem)
                selected_mask &= ~(1 << rem)
                selected.append(add)
                selected_set.add(add)
                selected_mask |= (1 << add)
                current_score += best_swap_gain
                improved = True
                continue
            
            # Try 2-for-1: remove one, add two
            best_2for1_gain = 0
            best_2for1 = None
            for rem in list(selected):
                rem_score = 1.0 / candidates[rem]
                temp_mask = selected_mask & ~(1 << rem)
                
                # Find all candidates that can be added to temp_mask
                candidates_for_add = []
                for add in range(n):
                    if add in selected_set:
                        continue
                    if can_add(add, temp_mask):
                        candidates_for_add.append(add)
                
                # Try pairs from candidates_for_add
                for i in range(len(candidates_for_add)):
                    a = candidates_for_add[i]
                    for j in range(i + 1, len(candidates_for_add)):
                        b = candidates_for_add[j]
                        # Check if a and b conflict with each other
                        if (conflict_bits[a] >> b) & 1:
                            continue
                        gain = (1.0 / candidates[a] + 1.0 / candidates[b]) - rem_score
                        if gain > best_2for1_gain + 1e-12:
                            best_2for1_gain = gain
                            best_2for1 = (rem, a, b)
            
            if best_2for1:
                rem, a, b = best_2for1
                selected.remove(rem)
                selected_set.discard(rem)
                selected_mask &= ~(1 << rem)
                selected.append(a)
                selected_set.add(a)
                selected_mask |= (1 << a)
                selected.append(b)
                selected_set.add(b)
                selected_mask |= (1 << b)
                current_score += best_2for1_gain
                improved = True
                continue
            
            # Try 1-for-2: remove two, add one
            best_1for2_gain = 0
            best_1for2 = None
            sel_list = list(selected)
            for i in range(len(sel_list)):
                for j in range(i + 1, len(sel_list)):
                    rem1 = sel_list[i]
                    rem2 = sel_list[j]
                    rem_score = 1.0 / candidates[rem1] + 1.0 / candidates[rem2]
                    temp_mask = selected_mask & ~(1 << rem1) & ~(1 << rem2)
                    
                    for add in range(n):
                        if add in selected_set:
                            continue
                        if can_add(add, temp_mask):
                            gain = (1.0 / candidates[add]) - rem_score
                            if gain > best_1for2_gain + 1e-12:
                                best_1for2_gain = gain
                                best_1for2 = (rem1, rem2, add)
            
            if best_1for2:
                rem1, rem2, add = best_1for2
                selected.remove(rem1)
                selected_set.discard(rem1)
                selected_mask &= ~(1 << rem1)
                selected.remove(rem2)
                selected_set.discard(rem2)
                selected_mask &= ~(1 << rem2)
                selected.append(add)
                selected_set.add(add)
                selected_mask |= (1 << add)
                current_score += best_1for2_gain
                improved = True
                continue
        
        return selected, current_score
    
    # Try MILP first
    best_solution = None
    best_score = 0.0
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        weights = np.array([1.0 / x for x in candidates])
        c = -weights
        
        # Build edge list
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (conflict_bits[i] >> j) & 1:
                    edges.append((i, j))
        
        if len(edges) == 0:
            best_solution = list(range(n))
            best_score = sum(weights)
        else:
            A = np.zeros((len(edges), n))
            for k, (i, j) in enumerate(edges):
                A[k, i] = 1
                A[k, j] = 1
            
            constraints = [LinearConstraint(A, -np.inf, np.ones(len(edges)))]
            bounds = Bounds(0, 1)
            integrality = np.ones(n)
            
            # Give MILP up to 20 seconds
            milp_deadline = start_time + 20.0
            res = milp(c, constraints=constraints, integrality=integrality, 
                       bounds=bounds, options={"time_limit": 20})
            
            if res.status == 0 and res.x is not None:
                sol_indices = [i for i in range(n) if res.x[i] > 0.5]
                if len(sol_indices) > 0:
                    # Verify validity
                    valid = True
                    for i in range(len(sol_indices)):
                        for j in range(i + 1, len(sol_indices)):
                            if (conflict_bits[sol_indices[i]] >> sol_indices[j]) & 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        sol_score = compute_score_float(sol_indices)
                        if sol_score > best_score:
                            best_score = sol_score
                            best_solution = sol_indices
    
    except Exception:
        pass
    
    # Local search from MILP solution or greedy
    if time.time() < deadline - 5.0:
        if best_solution is None:
            # Use greedy initialization
            order = list(range(n))  # ascending order = largest weights first
            best_solution = greedy_init(order)
            best_score = compute_score_float(best_solution)
        
        # Run local search
        ls_deadline = min(deadline - 2.0, start_time + 60.0)
        best_solution, best_score = local_search(best_solution, ls_deadline)
    
    # Randomized restarts if time permits
    if time.time() < deadline - 10.0:
        import random
        random.seed(42)
        
        restart_deadline = deadline - 5.0
        num_restarts = 0
        
        while time.time() < restart_deadline:
            num_restarts += 1
            
            # Random perturbation of the current best
            perturbed = list(best_solution)
            
            # Randomly remove 1-3 elements
            num_remove = min(len(perturbed), random.randint(1, 3))
            for _ in range(num_remove):
                if perturbed:
                    idx = random.randint(0, len(perturbed) - 1)
                    perturbed.pop(idx)
            
            # Randomly add 1-2 random valid elements
            num_add = random.randint(1, 2)
            for _ in range(num_add):
                # Pick a random candidate
                add_idx = random.randint(0, n - 1)
                if add_idx not in perturbed:
                    temp_mask = 0
                    for idx in perturbed:
                        temp_mask |= (1 << idx)
                    if can_add(add_idx, temp_mask):
                        perturbed.append(add_idx)
            
            # Run local search from perturbed state
            ls_deadline = min(restart_deadline, time.time() + 10.0)
            new_solution, new_score = local_search(perturbed, ls_deadline)
            
            if new_score > best_score + 1e-12:
                best_solution = new_solution
                best_score = new_score
    
    # Final polish: try to add any remaining valid elements
    if time.time() < deadline - 1.0:
        selected_set = set(best_solution)
        selected_mask = 0
        for idx in best_solution:
            selected_mask |= (1 << idx)
        
        for idx in range(n):
            if idx in selected_set:
                continue
            if can_add(idx, selected_mask):
                best_solution.append(idx)
                selected_set.add(idx)
                selected_mask |= (1 << idx)
                best_score += 1.0 / candidates[idx]
    
    # Convert to values
    result = [candidates[i] for i in best_solution]
    
    # Final validation
    if len(result) == 0:
        # Should not happen, but fallback
        result = [256]
    
    return result
