"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

_START_TIME = None

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _v2(n):
    """Calculate the 2-adic valuation of n."""
    count = 0
    while n % 2 == 0:
        count += 1
        n //= 2
    return count

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    deadline = _START_TIME + 85.0  # safety margin
    
    # Candidate moduli: All even numbers from 2 to 128
    # We know we cannot include odds if 2 is included, and 2 is in the evens set.
    # The set of evens is the largest pairwise non-coprime set.
    target_moduli = [m for m in range(2, 129) if m % 2 == 0]
    
    # Pre-compute gcds between all pairs of target moduli
    # To speed up constraint checking
    n = len(target_moduli)
    idx_to_mod = {m: i for i, m in enumerate(target_moduli)}
    mod_to_idx = {i: m for i, m in enumerate(target_moduli)}
    
    # gcd_matrix[i][j] = gcd(target_moduli[i], target_moduli[j])
    gcd_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            g = _gcd(target_moduli[i], target_moduli[j])
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g
    
    # Order moduli by increasing v2(m), then by m
    # Moduli with low v2 have smaller gcds with others (often just 2),
    # leading to tighter constraints (more forbidden residues relative to domain).
    ordered_moduli = sorted(target_moduli, key=lambda m: (_v2(m), m))
    
    # Map original index to ordered position?
    # We will work with the ordered list.
    # assigned[i] = residue for ordered_moduli[i]
    
    best_solution = []
    best_size = 0
    
    num_trials = 500
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        # Backtracking search
        # state: list of assigned residues, parallel to ordered_moduli
        # domains: for each unassigned modulus, a set of allowed residues
        # To be fast, we don't maintain full sets, we just check on the fly.
        # But forward checking helps prune.
        
        # Let's use a simple backtracking with randomization.
        # We'll use a recursive function or iterative with stack.
        # Given the small size (64), recursion depth is fine.
        
        assigned = [-1] * n
        # We need to map ordered index to original index for gcd lookup
        ordered_indices = [idx_to_mod[m] for m in ordered_moduli]
        
        def backtrack(depth):
            if time.time() > deadline:
                return False
            
            if depth == n:
                return True
            
            m = ordered_moduli[depth]
            orig_idx = ordered_indices[depth]
            
            # Generate candidate residues
            # Heuristic: Randomize order
            residues = list(range(m))
            random.shuffle(residues)
            
            for r in residues:
                # Check against all previously assigned moduli
                valid = True
                for k in range(depth):
                    m_prev = ordered_moduli[k]
                    orig_idx_prev = ordered_indices[k]
                    r_prev = assigned[k]
                    
                    g = gcd_matrix[orig_idx][orig_idx_prev]
                    # Check if r and r_prev are compatible
                    if (r - r_prev) % g == 0:
                        valid = False
                        break
                
                if valid:
                    assigned[depth] = r
                    if backtrack(depth + 1):
                        return True
                    assigned[depth] = -1
            
            return False
        
        # Start backtracking
        # We might not find a solution for all 64.
        # We want to maximize the count.
        # So we should try to include as many as possible.
        # Strategy: Try to solve for all. If fails, drop the last one (or a random one) and retry?
        # Or, run backtracking that stops when it finds a conflict, and returns the partial solution?
        
        # Better: Run backtracking that tries to assign all.
        # If it fails, we get a partial solution from the deepest point reached?
        # The `backtrack` function above returns False if it can't complete.
        # It doesn't track the best partial solution.
        
        # Let's modify: We'll just run the greedy backtracking.
        # If it fails, we might have a partial assignment in `assigned` that is valid for the prefix.
        # But `assigned` is overwritten.
        
        # Alternative: Use a greedy approach with backtracking limited to a small depth?
        # No, let's just do the full backtracking. If it's too slow, we'll timeout.
        # But 64 variables is small.
        
        if backtrack(0):
            # Full solution found
            current_sol = []
            for i in range(n):
                current_sol.append([assigned[i], ordered_moduli[i]])
            if len(current_sol) > best_size:
                best_size = len(current_sol)
                best_solution = current_sol
            break # Found max possible
        
        else:
            # Backtracking failed to find a full solution.
            # This implies no solution exists for all 64 evens?
            # Or the search space was too large for the time limit?
            # If time limit wasn't hit, it means no solution exists.
            # If time limit was hit, we have partial results?
            # The `backtrack` function doesn't return partial results.
            
            # Let's try a different strategy for this trial:
            # Greedy with randomization, but with a "look-ahead" or just simple greedy.
            # Since backtracking failed (or timed out), let's just do a simple greedy 
            # to get *some* solution, even if small, to update best_size if it's larger.
            
            current = []
            used_orig_indices = set()
            
            # Try to add moduli in the ordered sequence
            for i in range(n):
                if time.time() > deadline:
                    break
                m = ordered_moduli[i]
                orig_idx = ordered_indices[i]
                
                # Try random residues
                residues = list(range(m))
                random.shuffle(residues)
                
                found = False
                for r in residues:
                    valid = True
                    for (r_prev, m_prev) in current:
                        # Find orig index of m_prev
                        # This is O(n) lookup, inefficient.
                        # Let's store orig_idx in the tuple
                        pass
                    
                    # Redo with stored idx
                    valid = True
                    for (r_prev, m_prev, idx_prev) in current:
                        g = gcd_matrix[orig_idx][idx_prev]
                        if (r - r_prev) % g == 0:
                            valid = False
                            break
                    
                    if valid:
                        current.append((r, m, orig_idx))
                        found = True
                        break
                
                if found:
                    # Update best if current is larger
                    if len(current) > best_size:
                        best_size = len(current)
                        best_solution = [[r, m] for (r, m, _) in current]
                else:
                    # Skip this modulus
                    pass
            
    if not best_solution:
        return [[0, 2]]
        
    return best_solution

# EVOLVE-BLOCK-END
