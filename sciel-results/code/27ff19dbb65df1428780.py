"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290  # Safety margin before 300s kill
    
    # Preprocessing
    m = len(clauses)
    # Map variable index (0-based) to list of (clause_index, literal_value)
    # literal_value is the signed literal from the input
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            if 0 <= v < n:
                var_to_clauses[v].append((ci, lit))
    
    # Helper to check if a clause is satisfied
    def is_satisfied(ci, assignment):
        for lit in clauses[ci]:
            if assignment[abs(lit) - 1] == (lit > 0):
                return True
        return False

    def solve_one_restart(rng, max_flips, noise_prob):
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initialize satisfaction state
        sat = [False] * m
        unsat_indices = []
        for ci in range(m):
            if is_satisfied(ci, assignment):
                sat[ci] = True
            else:
                unsat_indices.append(ci)
        
        if not unsat_indices:
            return assignment, True
        
        current_unsat_count = len(unsat_indices)
        
        for _ in range(max_flips):
            if time.time() > deadline:
                return assignment, False
            
            if not unsat_indices:
                return assignment, True
            
            # Pick a random unsatisfied clause
            # Using a list and random choice is O(1) average if we just pick index
            # But we need to keep unsat_indices updated.
            # For efficiency, we can just pick a random index from the current unsat_indices list.
            # If the list gets large, it's still fine for N=50.
            ci = unsat_indices[rng.randrange(len(unsat_indices))]
            clause = clauses[ci]
            
            # Calculate gain/loss for each literal in the clause
            best_score = -1e9
            best_lit = clause[0]
            candidates = []
            
            for lit in clause:
                v = abs(lit) - 1
                if v < 0 or v >= n:
                    continue
                
                gain = 0
                loss = 0
                
                # Iterate over all clauses containing variable v
                for (uci, ulit) in var_to_clauses[v]:
                    if ulit == lit:
                        # Same sign as lit.
                        # If we flip v to make lit true, ulit also becomes true.
                        # If uci is currently unsatisfied, this is a gain.
                        # Note: uci could be ci itself, but ci is unsat, so it's a gain (we fix the chosen clause).
                        if not sat[uci]:
                            gain += 1
                    else:
                        # Opposite sign to lit.
                        # If we flip v to make lit true, ulit becomes false.
                        # If uci is currently satisfied, it might become unsatisfied.
                        # It becomes unsatisfied ONLY IF ulit was the ONLY literal satisfying uci.
                        # We don't track "only literal" easily without more state.
                        # Standard WalkSAT approximation: Count it as a loss if it is currently satisfied.
                        # This is a conservative estimate.
                        if sat[uci]:
                            loss += 1
                
                score = gain - loss
                candidates.append((score, lit))
                
                if score > best_score:
                    best_score = score
                    best_lit = lit
            
            if not candidates:
                # Fallback: pick random literal from clause
                lit = rng.choice(clause)
            else:
                # With probability noise_prob, pick random candidate
                # Otherwise, pick best score (with random tie-breaking)
                if rng.random() < noise_prob:
                    _, lit = rng.choice(candidates)
                else:
                    max_score = best_score
                    tied = [c for c in candidates if c[0] == max_score]
                    _, lit = rng.choice(tied)
            
            # Flip the variable
            v = abs(lit) - 1
            old_val = assignment[v]
            assignment[v] = not old_val
            
            # Update satisfaction state for all clauses containing v
            for (uci, ulit) in var_to_clauses[v]:
                if sat[uci]:
                    # Clause was satisfied. Check if it is still satisfied.
                    # It is no longer satisfied if ulit was the only literal making it true.
                    # We can just re-check the whole clause.
                    new_sat = is_satisfied(uci, assignment)
                    if not new_sat:
                        sat[uci] = False
                        unsat_indices.append(uci)
                        current_unsat_count += 1
                else:
                    # Clause was unsatisfied. Check if it is now satisfied.
                    new_sat = is_satisfied(uci, assignment)
                    if new_sat:
                        sat[uci] = True
                        # Remove uci from unsat_indices
                        # List removal is O(N), but N is small (50 vars, ~200 clauses)
                        # To avoid O(N^2) over many flips, we can use a set or just leave it and filter?
                        # For N=50, M~250, O(M) removal is fine.
                        try:
                            unsat_indices.remove(uci)
                        except ValueError:
                            pass # Should not happen if logic is correct
                        current_unsat_count -= 1
            
            if current_unsat_count == 0:
                return assignment, True
        
        return assignment, (current_unsat_count == 0)

    rng = random.Random(0)
    best_assign = None
    best_unsat = float('inf')
    
    # Strategy: WalkSAT with restarts
    # Typical WalkSAT parameters for N=50
    # Flip budget starts small, increases.
    # Noise probability usually around 0.05 - 0.3
    
    restarts = 0
    base_flips = 500
    max_restarts = 10000 # High limit, time will stop it
    
    while time.time() < deadline:
        restarts += 1
        # Increase flip budget as we go, but cap it
        # If we've had many restarts, maybe increase budget
        cur_flips = base_flips + int(restarts * 10)
        if cur_flips > 10000:
            cur_flips = 10000
            
        # Vary noise slightly to escape local minima
        # Base noise 0.1, vary between 0.05 and 0.2
        noise = 0.1 + 0.05 * math.sin(restarts * 0.1)
        
        assignment, solved = solve_one_restart(rng, cur_flips, noise)
        
        if solved:
            return assignment
        
        # Track best unsat count for fallback
        unsat_cnt = sum(1 for ci in range(m) if not is_satisfied(ci, assignment))
        if unsat_cnt < best_unsat:
            best_unsat = unsat_cnt
            best_assign = assignment[:]
            
        # Early termination if we are stuck at high unsat count for too long?
        # Or just let time handle it.
        
    if best_assign is not None:
        return best_assign
    
    # Fallback random
    return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
