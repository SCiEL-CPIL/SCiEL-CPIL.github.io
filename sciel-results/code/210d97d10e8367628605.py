import time
from math import gcd
from collections import defaultdict
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    candidates = list(range(20, 201))
    n = len(candidates)

    # Precompute gcd matrix
    gcd_matrix = [[gcd(candidates[i], candidates[j]) for j in range(n)] for i in range(n)]

    # We want to maximize sum of 1/m over selected classes.
    # This is a CSP: variables are moduli, domains are residues, constraints are pairwise.
    # Strategy: Use backtracking with MRV (Minimum Remaining Values) heuristic.
    # But since we want to maximize the sum, we should prioritize including small moduli.
    # 
    # Key insight: We don't need to include ALL moduli. We want to maximize sum(1/m).
    # So we should try to include as many small moduli as possible.
    # 
    # Approach:
    # 1. Sort moduli by increasing m (small m first, since 1/m is larger).
    # 2. Use backtracking to find a valid assignment for a subset of moduli.
    # 3. To maximize the score, we want to include moduli with small m.
    # 4. We can use a greedy approach with backtracking:
    #    - Try to include moduli in order of increasing m.
    #    - If we get stuck, backtrack and try a different residue or skip a modulus.
    # 
    # However, full backtracking on 181 variables is too expensive.
    # Instead, let's use a smarter approach:
    # 
    # 1. First, try a greedy assignment in order of increasing m.
    # 2. If some moduli can't be assigned, try to swap residues to free up space.
    # 3. Use local search to improve the solution.
    # 
    # Actually, let's use a CSP solver with forward checking and MRV.
    # But we need to be careful about the time limit.
    # 
    # Let's try a hybrid approach:
    # 1. Greedy assignment in order of increasing m.
    # 2. For each modulus, pick the residue that minimizes future conflicts.
    # 3. If no valid residue exists, skip the modulus.
    # 4. After the greedy pass, try to add back skipped moduli by swapping residues.
    
    best_score = 0.0
    best_solution = []
    
    # Greedy pass with lookahead
    def greedy_assign(order):
        """Assign residues in the given order, picking the residue that minimizes future conflicts."""
        assignment = {}  # m -> r
        for m in order:
            m_idx = candidates.index(m)
            valid_rs = []
            for r in range(m):
                valid = True
                for m2, r2 in assignment.items():
                    m2_idx = candidates.index(m2)
                    g = gcd_matrix[m_idx][m2_idx]
                    if r % g == r2 % g:
                        valid = False
                        break
                if valid:
                    valid_rs.append(r)
            
            if not valid_rs:
                continue
            
            # Pick the residue that minimizes future conflicts
            # For each future modulus m2, count how many residues are blocked.
            # A residue r blocks residue r2 in m2 if r2 % gcd(m, m2) == r % gcd(m, m2).
            # Number of blocked residues in m2 is m2 / gcd(m, m2).
            # So the cost is sum over future m2 of (m2 / gcd(m, m2)).
            # But this is independent of r! So we need a different heuristic.
            # 
            # Let's try: pick the residue that is "most spread out" from existing residues.
            # Or just pick randomly.
            r = random.choice(valid_rs)
            assignment[m] = r
        
        return assignment
    
    # Local search to improve the solution
    def local_search(assignment):
        """Try to add back skipped moduli by swapping residues."""
        current_assignment = dict(assignment)
        current_score = sum(1.0/m for m in current_assignment)
        
        # Find skipped moduli
        skipped = [m for m in candidates if m not in current_assignment]
        
        # Try to add each skipped modulus by swapping residues
        for m in skipped:
            m_idx = candidates.index(m)
            # Find a valid residue for m
            valid_rs = []
            for r in range(m):
                valid = True
                for m2, r2 in current_assignment.items():
                    m2_idx = candidates.index(m2)
                    g = gcd_matrix[m_idx][m2_idx]
                    if r % g == r2 % g:
                        valid = False
                        break
                if valid:
                    valid_rs.append(r)
            
            if valid_rs:
                # Try each valid residue and see if it improves the score
                best_r = None
                best_new_score = current_score
                for r in valid_rs:
                    # Check if adding (m, r) is valid
                    valid = True
                    for m2, r2 in current_assignment.items():
                        m2_idx = candidates.index(m2)
                        g = gcd_matrix[m_idx][m2_idx]
                        if r % g == r2 % g:
                            valid = False
                            break
                    if valid:
                        new_score = current_score + 1.0/m
                        if new_score > best_new_score:
                            best_new_score = new_score
                            best_r = r
                
                if best_r is not None:
                    current_assignment[m] = best_r
                    current_score = best_new_score
        
        return current_assignment, current_score
    
    # Run multiple restarts with different random seeds
    for restart in range(100):
        if time.time() > deadline:
            break
        
        random.seed(restart)
        
        # Greedy assignment in order of increasing m
        order = sorted(candidates)
        assignment = greedy_assign(order)
        
        # Local search to improve
        assignment, score = local_search(assignment)
        
        if score > best_score:
            best_score = score
            best_solution = {m: r for m, r in assignment.items()}
    
    # If no solution found, return a trivial solution
    if not best_solution:
        return [[20, 0]]
    
    return [[m, r] for m, r in best_solution.items()]
