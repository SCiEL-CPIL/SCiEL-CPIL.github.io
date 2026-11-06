"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, lcm
from fractions import Fraction

def construct():
    start_time = time.time()
    TIME_LIMIT = 85.0  # safety margin before 90s kill
    
    # Candidates: 2 to 256
    candidates = list(range(2, 257))
    n = len(candidates)
    
    # Precompute compatibility: comp[i][j] = True if lcm(candidates[i], candidates[j]) > 256
    # For efficiency, precompute for all pairs
    comp = [[True] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = candidates[i], candidates[j]
            if lcm(a, b) <= 256:
                comp[i][j] = False
                comp[j][i] = False
    
    # Sort by reciprocal descending (smaller numbers first)
    order = sorted(range(n), key=lambda i: candidates[i])
    # order[i] is the index into candidates of the i-th candidate to consider
    
    # We'll use branch and bound
    # State: current selected set (as a set of indices into candidates)
    # We process candidates in order (smallest first = highest reciprocal first)
    
    best_sum = Fraction(0, 1)
    best_set = set()
    
    # For speed, use float for bounds but Fraction for exact comparison
    # Actually, let's use Fraction throughout for correctness, but it may be slow.
    # Let's use float for the bound and track the exact sum separately.
    
    best_sum_float = 0.0
    
    # Precompute reciprocal as float
    recip = [1.0 / candidates[i] for i in range(n)]
    
    # For the bound: at step k (considering order[k]), the optimistic bound is
    # current_sum + sum of recip[i] for i in order[k:] that are compatible with all selected
    
    def compatible_with_selected(i, selected_indices):
        """Check if candidate i is compatible with all in selected_indices."""
        for j in selected_indices:
            if not comp[i][j]:
                return False
        return True
    
    # Branch and bound with recursion
    # To avoid deep recursion issues, we'll use an iterative approach or limit depth
    # Actually, n=255 is too large for full B&B. We need smarter pruning.
    
    # Key insight: the number of candidates that can be included is limited.
    # Let's think about which numbers are "heavy" (small, high reciprocal) and which are "light".
    
    # Alternative approach: since the constraint is pairwise lcm > 256, let's think
    # about which small numbers are mutually compatible.
    
    # Let me try a different strategy: 
    # 1. Identify a small core set of high-value numbers that are mutually compatible
    # 2. Then greedily add the best remaining compatible numbers
    
    # First, let's find the best subset among the small numbers (say, 2-50) using B&B
    # then extend greedily.
    
    # Actually, let's do a smarter B&B. The key is that the search space is large
    # but the constraint structure allows good pruning.
    
    # Let me try: process in order of decreasing reciprocal (increasing number)
    # At each step, we have a choice: include or exclude.
    
    # For the bound, compute the max possible sum from remaining candidates
    # that are compatible with current selection.
    
    # Let's use a recursive B&B with memoization on the "frontier" of conflicts.
    
    # Given the complexity, let me try a practical approach:
    # 1. Start with an initial solution from greedy (but optimized)
    # 2. Use local search / hill climbing to improve
    
    # First, get a good initial solution
    def greedy_solution():
        selected = []
        for i in order:  # smallest first
            if all(comp[i][j] for j in selected):
                selected.append(i)
        return selected
    
    init_sol = greedy_solution()
    init_sum = sum(recip[i] for i in init_sol)
    best_sum = init_sum
    best_set = set(init_sol)
    
    # Now try to improve with local search
    # Try swapping: remove one element, add one or more
    
    def can_add(i, selected):
        return all(comp[i][j] for j in selected)
    
    def solution_sum(selected):
        return sum(recip[i] for i in selected)
    
    improved = True
    while improved:
        improved = False
        current = list(best_set)
        # Try adding each candidate not in current
        for i in range(n):
            if i in best_set:
                continue
            if can_add(i, best_set):
                new_set = best_set | {i}
                new_sum = solution_sum(new_set)
                if new_sum > best_sum + 1e-15:
                    best_sum = new_sum
                    best_set = new_set
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try 1-for-1 swaps
        for i in list(best_set):
            for j in range(n):
                if j in best_set:
                    continue
                new_set = (best_set - {i}) | {j}
                if can_add(j, new_set):
                    new_sum = solution_sum(new_set)
                    if new_sum > best_sum + 1e-15:
                        best_sum = new_sum
                        best_set = new_set
                        improved = True
                        break
            if improved:
                break
        
        if improved:
            continue
        
        # Try 2-for-1 swaps (remove 2, add 1)
        current_list = list(best_set)
        for a_idx in range(len(current_list)):
            for b_idx in range(a_idx + 1, len(current_list)):
                a, b = current_list[a_idx], current_list[b_idx]
                base = best_set - {a, b}
                for j in range(n):
                    if j in base:
                        continue
                    if can_add(j, base):
                        new_set = base | {j}
                        new_sum = solution_sum(new_set)
                        if new_sum > best_sum + 1e-15:
                            best_sum = new_sum
                            best_set = new_set
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    
    # Now try a B&B approach on a reduced problem
    # Focus on the "interesting" candidates: those with high reciprocal
    # and those that conflict with the current solution
    
    # Let's try a proper B&B on a subset of candidates
    # Consider candidates 2 to 100 (high value region)
    
    if time.time() - start_time < TIME_LIMIT - 5.0:
        # B&B on first 60 candidates (2 to ~61)
        subset_size = 60
        subset_indices = order[:subset_size]
        
        # For B&B, we need to consider interactions with the rest of the solution
        # This is complex. Let's try a different approach.
        
        # Try simulated annealing-like local search
        import random
        random.seed(42)
        
        current_set = set(best_set)
        current_sum = best_sum
        
        for iteration in range(50000):
            if time.time() - start_time > TIME_LIMIT - 3.0:
                break
            
            # Random move: add, remove, or swap
            move_type = random.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add':
                # Try to add a random candidate
                candidates_to_add = [i for i in range(n) if i not in current_set and can_add(i, current_set)]
                if not candidates_to_add:
                    continue
                i = random.choice(candidates_to_add)
                new_set = current_set | {i}
                new_sum = current_sum + recip[i]
                if new_sum > current_sum:
                    current_set = new_set
                    current_sum = new_sum
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_set = set(current_set)
            
            elif move_type == 'remove':
                if len(current_set) <= 1:
                    continue
                i = random.choice(list(current_set))
                new_set = current_set - {i}
                new_sum = current_sum - recip[i]
                # Only accept if it enables better additions later
                # For now, skip removals unless we can immediately add something better
                # Try to add something after removal
                candidates_to_add = [j for j in range(n) if j not in new_set and can_add(j, new_set)]
                best_add = None
                best_add_val = -1
                for j in candidates_to_add:
                    if recip[j] > recip[i] + 1e-15:
                        if best_add is None or recip[j] > best_add_val:
                            best_add = j
                            best_add_val = recip[j]
                if best_add is not None:
                    new_set = new_set | {best_add}
                    new_sum = new_sum + recip[best_add]
                    if new_sum > current_sum:
                        current_set = new_set
                        current_sum = new_sum
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_set = set(current_set)
            
            elif move_type == 'swap':
                if not current_set:
                    continue
                i = random.choice(list(current_set))
                candidates_to_swap = [j for j in range(n) if j not in current_set]
                if not candidates_to_swap:
                    continue
                j = random.choice(candidates_to_swap)
                new_set = (current_set - {i}) | {j}
                if can_add(j, new_set):
                    new_sum = current_sum - recip[i] + recip[j]
                    if new_sum > current_sum:
                        current_set = new_set
                        current_sum = new_sum
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_set = set(current_set)
    
    # Convert back to values
    result = [candidates[i] for i in best_set]
    return result
# EVOLVE-BLOCK-END
