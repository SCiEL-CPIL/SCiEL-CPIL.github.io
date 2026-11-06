import math
import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s limit
    
    max_m = 128
    
    # Precompute GCDs
    gcds = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(2, max_m + 1):
        for j in range(2, max_m + 1):
            gcds[i][j] = math.gcd(i, j)
    
    # Candidate moduli: all even numbers from 2 to 128
    # These are the ones with pairwise gcd >= 2, so disjointness is possible
    all_even_moduli = list(range(2, max_m + 1, 2))
    
    # For each modulus m, precompute which residues are "compatible" with another modulus m'
    # Two residue classes (r, m) and (r', m') are disjoint iff (r - r') % gcd(m, m') != 0
    
    # Strategy: Use a smart backtracking with forward checking
    # Key insight: Process moduli in order of increasing number of "conflicts" first
    # A modulus m has more conflicts with already-assigned moduli if it shares larger gcds
    
    # Actually, let's think about this differently.
    # For two even moduli m and m', gcd(m, m') >= 2.
    # The residue classes (r, m) and (r', m') intersect iff r ≡ r' (mod gcd(m, m')).
    # So they are disjoint iff r ≠ r' (mod gcd(m, m')).
    
    # We need to assign residues r_m for each even modulus m such that
    # for all pairs m, m': r_m ≠ r_m' (mod gcd(m, m'))
    
    # This is a constraint satisfaction problem.
    # Let's use a backtracking solver with good heuristics.
    
    def is_valid(r, m, assigned):
        """Check if residue r for modulus m is valid given current assignments."""
        for r_i, m_i in assigned:
            g = gcds[m][m_i]
            if (r - r_i) % g == 0:
                return False
        return True
    
    def solve(moduli_order, assigned, remaining_set):
        """Backtracking solver."""
        if time.time() > deadline:
            return None
        
        if not remaining_set:
            return assigned
        
        # MRV: pick the modulus with fewest valid residues
        best_m = None
        best_valid = None
        min_count = float('inf')
        
        for m in remaining_set:
            valid_r = []
            for r in range(m):
                if is_valid(r, m, assigned):
                    valid_r.append(r)
            if len(valid_r) < min_count:
                min_count = len(valid_r)
                best_m = m
                best_valid = valid_r
            if min_count == 0:
                return None
            if min_count <= 1:
                break
        
        if best_m is None:
            return None
        
        # Try residues for best_m
        # Use a deterministic order but with some randomization via shuffling
        remaining_set.discard(best_m)
        
        # Sort valid residues to try "easy" ones first (smaller residues might be more constrained)
        # Actually, let's just try them in order, or shuffle for diversity
        shuffled = list(best_valid)
        random.shuffle(shuffled)
        
        for r in shuffled:
            new_assigned = assigned + [[r, best_m]]
            res = solve(moduli_order, new_assigned, remaining_set)
            if res is not None:
                return res
        
        remaining_set.add(best_m)
        return None
    
    # Order moduli by increasing "freedom" - moduli with smaller values have fewer residues
    # but also may have more constraints. Let's try processing in order of increasing modulus.
    # Actually, MRV will handle the ordering dynamically.
    
    best_solution = []
    best_count = 0
    
    # Try multiple random restarts with different initial orderings
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Shuffle the order in which we consider moduli
        moduli_order = all_even_moduli[:]
        random.shuffle(moduli_order)
        
        # Use a set for remaining for O(1) operations
        remaining = set(moduli_order)
        
        # Try to solve
        result = solve(moduli_order, [], remaining)
        
        if result is not None:
            count = len(result)
            if count > best_count:
                best_count = count
                best_solution = result
                if best_count == len(all_even_moduli):
                    break
    
    # If we didn't get all 64, try a greedy approach as fallback
    if best_count < len(all_even_moduli):
        # Greedy: assign residues one by one, choosing the modulus with fewest options
        for attempt in range(2000):
            if time.time() > deadline:
                break
            
            assigned = []
            remaining = set(all_even_moduli)
            
            while remaining:
                # Find modulus with fewest valid residues
                best_m = None
                best_valid = None
                min_count = float('inf')
                
                for m in remaining:
                    valid_r = []
                    for r in range(m):
                        if is_valid(r, m, assigned):
                            valid_r.append(r)
                    if len(valid_r) < min_count:
                        min_count = len(valid_r)
                        best_m = m
                        best_valid = valid_r
                    if min_count == 0:
                        break
                
                if best_m is None or not best_valid:
                    break
                
                r = random.choice(best_valid)
                assigned.append([r, best_m])
                remaining.discard(best_m)
            
            if len(assigned) > best_count:
                best_count = len(assigned)
                best_solution = assigned
                if best_count == len(all_even_moduli):
                    break
    
    # Ensure we have at least one valid solution
    if not best_solution:
        # Fallback: just use modulus 2 with residue 0
        best_solution = [[0, 2]]
    
    return best_solution
