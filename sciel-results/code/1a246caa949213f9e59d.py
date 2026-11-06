"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0  # safety margin before 330s kill
    
    # Preprocessing: pure literal elimination and unit propagation
    # Build clause data structures for efficient propagation
    # Each clause is a list of signed literals (1-based)
    
    # We'll work with a mutable assignment: -1 unknown, 0 False, 1 True
    assignment = [-1] * n
    
    # Build index: literal -> list of (clause_idx, position)
    lit_to_clauses = defaultdict(list)
    for ci, clause in enumerate(clauses):
        for pi, lit in enumerate(clause):
            lit_to_clauses[lit].append((ci, pi))
    
    # Track clause states: for each clause, count of unassigned literals and satisfied
    clause_unassigned = [len(c) for c in clauses]
    clause_satisfied = [False] * len(clauses)
    
    def is_satisfied(clause):
        return any(assignment[abs(l)-1] == (1 if l > 0 else 0) for l in clause)
    
    def propagate():
        """Unit propagation and pure literal elimination.
        Returns True if consistent, False if contradiction."""
        changed = True
        while changed:
            changed = False
            # Unit propagation
            for ci in range(len(clauses)):
                if clause_satisfied[ci]:
                    continue
                if clause_unassigned[ci] == 0:
                    return False  # contradiction
                if clause_unassigned[ci] == 1:
                    # Find the unassigned literal
                    for lit in clauses[ci]:
                        var = abs(lit) - 1
                        if assignment[var] == -1:
                            val = 1 if lit > 0 else 0
                            if assignment[var] != -1 and assignment[var] != val:
                                return False
                            assignment[var] = val
                            clause_unassigned[ci] = 0
                            clause_satisfied[ci] = True
                            changed = True
                            # Update other clauses containing this variable
                            for olit in [lit, -lit]:
                                for (oci, opi) in lit_to_clauses[olit]:
                                    if not clause_satisfied[oci]:
                                        if assignment[abs(olit)-1] == (1 if olit > 0 else 0):
                                            clause_satisfied[oci] = True
                                            clause_unassigned[oci] = 0
                                        else:
                                            clause_unassigned[oci] -= 1
                            break
                    break
            # Pure literal elimination
            if not changed:
                # Check for pure literals
                for lit in lit_to_clauses:
                    var = abs(lit) - 1
                    if assignment[var] != -1:
                        continue
                    # Check if the opposite literal appears in any unsatisfied clause
                    opposite = -lit
                    if opposite in lit_to_clauses:
                        has_opposite = False
                        for (oci, opi) in lit_to_clauses[opposite]:
                            if not clause_satisfied[oci]:
                                has_opposite = True
                                break
                        if not has_opposite:
                            # Pure literal: assign it
                            assignment[var] = 1 if lit > 0 else 0
                            changed = True
                            for olit in [lit, -lit]:
                                for (oci, opi) in lit_to_clauses[olit]:
                                    if not clause_satisfied[oci]:
                                        if assignment[abs(olit)-1] == (1 if olit > 0 else 0):
                                            clause_satisfied[oci] = True
                                            clause_unassigned[oci] = 0
                                        else:
                                            clause_unassigned[oci] -= 1
                            break
    
    # Run initial propagation
    if not propagate():
        # Contradiction from preprocessing? Shouldn't happen for SAT instances, but handle gracefully
        pass
    
    # Check if already solved
    all_satisfied = all(clause_satisfied[ci] for ci in range(len(clauses)))
    if all_satisfied:
        return [assignment[i] == 1 for i in range(n)]
    
    # WalkSAT with restarts, clause-weighted selection, and dynamic noise
    rng = random.Random(42)
    
    # Phase saving: track how often each variable was set to True/False
    phase_true = [0] * n
    phase_false = [0] * n
    
    def init_assignment():
        """Initialize assignment using phase saving or random."""
        nonlocal assignment, clause_unassigned, clause_satisfied
        assignment = [-1] * n
        clause_unassigned = [len(c) for c in clauses]
        clause_satisfied = [False] * len(clauses)
        
        # Assign using phase saving bias
        for i in range(n):
            if phase_true[i] + phase_false[i] > 0:
                assignment[i] = 1 if phase_true[i] >= phase_false[i] else 0
            else:
                assignment[i] = 1 if rng.random() < 0.5 else 0
        
        # Update clause states
        for ci, clause in enumerate(clauses):
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (1 if lit > 0 else 0):
                    clause_satisfied[ci] = True
                    break
        
        # Run unit propagation on new assignment
        # Rebuild lit_to_clauses references are still valid
        # But we need to recompute clause_unassigned properly
        for ci in range(len(clauses)):
            if not clause_satisfied[ci]:
                count = 0
                for lit in clauses[ci]:
                    var = abs(lit) - 1
                    if assignment[var] != (1 if lit > 0 else 0):
                        count += 1
                clause_unassigned[ci] = count
    
    def count_unsat():
        return sum(1 for ci in range(len(clauses)) if not clause_satisfied[ci])
    
    def walksat_steps(max_steps, noise_start, noise_end):
        """Run WalkSAT for max_steps. Returns (solved, best_unsat_count)."""
        nonlocal assignment, clause_unassigned, clause_satisfied, phase_true, phase_false
        
        # Compute initial unsat count
        unsat_count = count_unsat()
        best_unsat = unsat_count
        best_assignment = assignment[:]
        
        for step in range(max_steps):
            if unsat_count == 0:
                return True, 0
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
            
            # Check deadline
            if time.time() > deadline - 5.0:
                break
            
            # Progressively decrease noise
            progress = step / max_steps
            noise = noise_start + (noise_end - noise_start) * progress
            
            # Select an unsatisfied clause
            # Weighted by number of unassigned literals (fewer = more urgent)
            unsat_clauses = [ci for ci in range(len(clauses)) if not clause_satisfied[ci]]
            if not unsat_clauses:
                return True, 0
            
            # Weighted selection: prefer clauses with fewer unassigned literals
            weights = []
            for ci in unsat_clauses:
                w = 1.0 / (clause_unassigned[ci] + 1)
                weights.append(w)
            
            # Normalize and pick
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0
            selected_ci = unsat_clauses[0]
            for idx, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    selected_ci = unsat_clauses[idx]
                    break
            
            clause = clauses[selected_ci]
            
            # Select literal to flip
            # With probability noise, flip a random literal
            # Otherwise, flip the literal that minimizes damage
            if rng.random() < noise:
                lit = rng.choice(clause)
            else:
                # Compute damage for each literal
                best_damage = float('inf')
                best_lit = clause[0]
                for lit in clause:
                    var = abs(lit) - 1
                    old_val = assignment[var]
                    new_val = 1 - old_val
                    
                    # Update phase
                    if new_val == 1:
                        phase_true[var] += 1
                    else:
                        phase_false[var] += 1
                    assignment[var] = new_val
                    
                    # Compute damage: how many previously satisfied clauses become unsatisfied
                    damage = 0
                    for olit in [lit, -lit]:
                        for (oci, opi) in lit_to_clauses[olit]:
                            if clause_satisfied[oci]:
                                # Check if this clause is still satisfied after flip
                                if assignment[abs(olit)-1] != (1 if olit > 0 else 0):
                                    # Check if any other literal in the clause is satisfied
                                    still_sat = False
                                    for l2 in clauses[oci]:
                                        if l2 == olit:
                                            continue
                                        v2 = abs(l2) - 1
                                        if assignment[v2] == (1 if l2 > 0 else 0):
                                            still_sat = True
                                            break
                                    if not still_sat:
                                        damage += 1
                    # Revert
                    assignment[var] = old_val
                    if new_val == 1:
                        phase_true[var] -= 1
                    else:
                        phase_false[var] -= 1
                    
                    if damage < best_damage:
                        best_damage = damage
                        best_lit = lit
                lit = best_lit
            
            # Flip the literal
            var = abs(lit) - 1
            old_val = assignment[var]
            new_val = 1 - old_val
            assignment[var] = new_val
            
            if new_val == 1:
                phase_true[var] += 1
            else:
                phase_false[var] += 1
            
            # Update clause states
            for olit in [lit, -lit]:
                for (oci, opi) in lit_to_clauses[olit]:
                    if assignment[abs(olit)-1] == (1 if olit > 0 else 0):
                        if not clause_satisfied[oci]:
                            clause_satisfied[oci] = True
                            unsat_count -= 1
                    else:
                        if clause_satisfied[oci]:
                            # Check if clause is still satisfied by another literal
                            still_sat = False
                            for l2 in clauses[oci]:
                                v2 = abs(l2) - 1
                                if assignment[v2] == (1 if l2 > 0 else 0):
                                    still_sat = True
                                    break
                            if not still_sat:
                                clause_satisfied[oci] = False
                                unsat_count += 1
                        # Update unassigned count
                        if assignment[abs(olit)-1] == -1:
                            pass  # shouldn't happen after init
    
            # Update clause_unassigned for the flipped variable
            for olit in [lit, -lit]:
                for (oci, opi) in lit_to_clauses[olit]:
                    if not clause_satisfied[oci]:
                        var_oci = abs(olit) - 1
                        # Recompute unassigned count if needed
                        pass  # Simplified: skip exact tracking for speed
    
        return False, best_unsat
    
    # Main search loop with restarts
    init_assignment()
    unsat_count = count_unsat()
    if unsat_count == 0:
        return [assignment[i] == 1 for i in range(n)]
    
    best_unsat_overall = unsat_count
    best_assignment_overall = assignment[:]
    
    restart_count = 0
    max_restarts = 5000
    
    while time.time() < deadline - 10.0 and restart_count < max_restarts:
        restart_count += 1
        
        # Increase steps per restart as we go, but cap total time
        steps = 2000 + restart_count * 50
        if steps > 50000:
            steps = 50000
        
        noise_start = 0.5
        noise_end = 0.3
        
        solved, best_unsat = walksat_steps(steps, noise_start, noise_end)
        
        if solved:
            return [assignment[i] == 1 for i in range(n)]
        
        if best_unsat < best_unsat_overall:
            best_unsat_overall = best_unsat
            best_assignment_overall = assignment[:]
        
        # Restart with new random assignment
        if time.time() > deadline - 15.0:
            break
        init_assignment()
    
    # Return best assignment found
    if best_unsat_overall < count_unsat():
        return [best_assignment_overall[i] == 1 for i in range(n)]
    return [assignment[i] == 1 for i in range(n)]

# EVOLVE-BLOCK-END
