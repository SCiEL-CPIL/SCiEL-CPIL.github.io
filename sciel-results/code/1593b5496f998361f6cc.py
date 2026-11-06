"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    start_time = time.time()
    timeout = 320.0  # Leave margin for overhead
    
    m = len(clauses)
    if m == 0:
        return [True] * n
        
    # Preprocess
    # var_clauses: list of lists, var_clauses[v] = [clause_idx1, clause_idx2, ...]
    var_clauses = [[] for _ in range(n)]
    
    # Store clause literals as (var_idx, sign) where sign is 1 for positive, -1 for negative
    clause_lits = []
    
    for i, c in enumerate(clauses):
        lits = []
        for lit in c:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            lits.append((v, s))
            var_clauses[v].append(i)
        clause_lits.append(lits)
        
    # Map variable to list of (clause_idx, sign)
    var_clause_info = [[] for _ in range(n)]
    for i, c in enumerate(clause_lits):
        for lit in c:
            v, s = lit
            var_clause_info[v].append((i, s))

    def walk_sat(initial_assignment, max_steps, seed, p):
        rng = random.Random(seed)
        assignment = initial_assignment[:]
        
        # clause_sat: bool array
        clause_sat = [False] * m
        # clause_true_count: int array
        clause_true_count = [0] * m
        
        # Initialize
        for i, c in enumerate(clause_lits):
            t_count = 0
            for lit in c:
                v, s = lit
                if (s == 1 and assignment[v]) or (s == -1 and not assignment[v]):
                    t_count += 1
            clause_true_count[i] = t_count
            clause_sat[i] = (t_count > 0)
            
        unsat_count = m - sum(clause_sat)
        
        if unsat_count == 0:
            return assignment[:]
            
        for step in range(max_steps):
            if time.time() - start_time > timeout:
                return None
                
            if unsat_count == 0:
                return assignment[:]
                
            # Find an unsatisfied clause
            # Random sampling
            unsat_idx = -1
            for _ in range(10):
                idx = rng.randint(0, m - 1)
                if not clause_sat[idx]:
                    unsat_idx = idx
                    break
            
            if unsat_idx == -1:
                # If we couldn't find one in 10 tries, just pick random
                unsat_idx = rng.randint(0, m - 1)
                
            c_lits = clause_lits[unsat_idx]
            
            # Evaluate variables in this clause
            # We want to minimize losses (clauses that become unsat)
            # and maximize gains (clauses that become sat)
            
            best_var = -1
            best_loss = float('inf')
            best_gain = -1
            best_rand = 0
            
            candidates = []
            for lit in c_lits:
                v, s = lit
                losses = 0
                gains = 0
                
                # Check all clauses containing v
                for (ci, si) in var_clause_info[v]:
                    # Check if literal (v, si) is currently true
                    is_true = (si == 1 and assignment[v]) or (si == -1 and not assignment[v])
                    
                    if is_true:
                        # Flipping v makes it false.
                        # Clause ci loses a true literal.
                        # It becomes unsat if it had exactly 1 true literal.
                        if clause_true_count[ci] == 1:
                            losses += 1
                    else:
                        # Flipping v makes it true.
                        # Clause ci gains a true literal.
                        # It becomes sat if it had 0 true literals.
                        if clause_true_count[ci] == 0:
                            gains += 1
                            
                candidates.append((v, losses, gains))
                
            if rng.random() < p:
                # Random choice
                v, losses, gains = candidates[rng.randint(0, 2)]
            else:
                # Best choice: min losses, then max gains
                # Sort by (losses, -gains)
                # Since only 3, manual min
                best_cand = candidates[0]
                min_loss = best_cand[1]
                max_gain = best_cand[2]
                
                for cand in candidates[1:]:
                    if cand[1] < min_loss:
                        min_loss = cand[1]
                        max_gain = cand[2]
                        best_cand = cand
                    elif cand[1] == min_loss and cand[2] > max_gain:
                        max_gain = cand[2]
                        best_cand = cand
                        
                v, losses, gains = best_cand
                
            # Flip v
            assignment[v] = not assignment[v]
            
            # Update clause stats for clauses containing v
            for (ci, si) in var_clause_info[v]:
                # Recalculate true count for clause ci
                t_count = 0
                for lit in clause_lits[ci]:
                    vv, ss = lit
                    if (ss == 1 and assignment[vv]) or (ss == -1 and not assignment[vv]):
                        t_count += 1
                clause_true_count[ci] = t_count
                clause_sat[ci] = (t_count > 0)
                
            # Update unsat_count
            # Instead of scanning all m clauses, we can update based on changes.
            # But we didn't track old_sat.
            # Scanning m=450 is fast.
            unsat_count = 0
            for i in range(m):
                if not clause_sat[i]:
                    unsat_count += 1
                    
        return None

    rng = random.Random(42)
    
    # Parameters
    p = 0.2  # Randomization probability
    max_steps_per_restart = 10000
    
    restarts = 0
    while time.time() - start_time < timeout - 5:
        initial = [rng.random() < 0.5 for _ in range(n)]
        result = walk_sat(initial, max_steps_per_restart, seed=restarts, p=p)
        
        if result is not None:
            return result
            
        restarts += 1
        
    return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
