import time
import random
import numpy as np


def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 55  # 5s safety margin before 60s kill

    # Precompute all arithmetic progressions
    progressions = []
    prog_by_index = [[] for _ in range(n)]

    pid = 0
    for step in range(1, n):
        for start in range(step):
            max_len = (n - 1 - start) // step + 1
            for length in range(1, max_len + 1):
                indices = tuple(start + k * step for k in range(length))
                progressions.append(indices)
                for idx in indices:
                    prog_by_index[idx].append(pid)
                pid += 1

    num_progs = len(progressions)

    # For each index, precompute which progressions it belongs to and its contribution pattern
    # When we assign sign s to index i, for each progression p containing i, the sum changes by s
    # We need to track current partial sums of each progression

    best_signs = None
    best_max_abs = float('inf')

    # CSP with backtracking and forward checking
    # Strategy: try bounds from 1 upward, use backtracking with pruning
    
    def try_csp(max_bound, time_limit):
        """Try to find assignment where all progression sums are in [-max_bound, max_bound]"""
        nonlocal best_signs, best_max_abs
        
        signs = [-1] * n  # -1 means unassigned, 0 or 1 means assigned sign
        # Actually use: signs[i] = 0 (unassigned), 1 (+1), -1 (-1)
        signs = [0] * n
        
        # Current sum for each progression (only counting assigned elements)
        prog_sums = [0] * num_progs
        # Count of assigned elements in each progression
        prog_counts = [0] * num_progs
        
        # For forward checking: for each unassigned index, track which progressions
        # would be violated if we assign +1 or -1
        # We'll use a simpler approach: at each step, check if current partial sums
        # can still be kept within bounds
        
        # Variable ordering: use most-constrained-first heuristic
        # Precompute for each index, the number of progressions it's in
        index_deg = [len(prog_by_index[i]) for i in range(n)]
        
        # Sort indices by degree (most constrained first)
        order = sorted(range(n), key=lambda i: -index_deg[i])
        
        # We'll assign variables in this order
        # But we need to be careful: progressions connect non-adjacent indices
        
        # Alternative: use natural order but with good pruning
        # Let's use natural order for simplicity, with strong pruning
        
        # Recursive backtracking
        def backtrack(pos):
            nonlocal best_signs, best_max_abs
            
            if time.time() > time_limit:
                return False
            
            if pos == n:
                # All assigned, check if valid
                # Actually we should have pruned already
                best_signs = signs[:]
                best_max_abs = max_bound
                return True
            
            idx = pos  # Use natural order
            # Try +1 first, then -1
            for s in [1, -1]:
                # Check if assigning s to idx keeps all progressions feasible
                feasible = True
                
                # Update progressions containing idx
                affected = prog_by_index[idx]
                for p in affected:
                    new_sum = prog_sums[p] + s
                    # Check if this violates the bound
                    if abs(new_sum) > max_bound:
                        feasible = False
                        break
                    # Forward checking: check if remaining unassigned elements
                    # in this progression can keep it within bounds
                    # Remaining elements in progression p:
                    prog_indices = progressions[p]
                    remaining = 0
                    for j in prog_indices:
                        if j > idx and signs[j] == 0:
                            remaining += 1
                    
                    # Current sum after this assignment: new_sum
                    # Max possible future change: remaining (all could be +1 or -1)
                    # So final sum will be in [new_sum - remaining, new_sum + remaining]
                    # We need this interval to intersect [-max_bound, max_bound]
                    if new_sum + remaining < -max_bound or new_sum - remaining > max_bound:
                        feasible = False
                        break
                
                if not feasible:
                    continue
                
                # Apply assignment
                signs[idx] = s
                for p in affected:
                    prog_sums[p] += s
                    prog_counts[p] += 1
                
                # Recurse
                if backtrack(pos + 1):
                    return True
                
                # Undo
                for p in affected:
                    prog_sums[p] -= s
                    prog_counts[p] -= 1
                signs[idx] = 0
        
        return backtrack(0)
    
    # Try increasing bounds
    for bound in range(1, 10):
        if time.time() > deadline - 2:
            break
        
        time_limit = min(deadline - 1, time.time() + 8)
        if try_csp(bound, time_limit):
            break
    
    # If CSP didn't find anything, fall back to random
    if best_signs is None:
        rng = random.Random(42)
        best_signs = [rng.choice([-1, 1]) for _ in range(n)]
        # Compute actual max for reporting
        actual_max = 0
        for p in progressions:
            s = sum(best_signs[i] for i in p)
            actual_max = max(actual_max, abs(s))
        best_max_abs = actual_max
    
    return [int(x) for x in best_signs]
