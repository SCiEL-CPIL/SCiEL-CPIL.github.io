"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def solve(instance):
    """
    Solves a 3-SAT instance using an optimized WalkSAT algorithm with restarts.
    """
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline: 280 seconds (leaving 20s margin for safety)
    deadline = time.time() + 280.0
    
    # Pre-process clauses for faster access
    # Represent each clause as a tuple of (var_index, sign) where sign is 1 for positive, -1 for negative
    # var_index is 0-based
    processed_clauses = []
    for c in clauses:
        pc = []
        for lit in c:
            if lit > 0:
                pc.append((lit - 1, 1))
            else:
                pc.append((lit - 1, -1))
        processed_clauses.append(tuple(pc))
        
    m = len(processed_clauses)
    
    # Random seed for reproducibility across runs if needed, but we want robustness.
    # Using a fixed seed might be safer for evaluation consistency, but random is often better for generalization.
    # Let's use a seed based on current time or a fixed one? The prompt says "locally generated planted".
    # Usually, these are easy. Let's use a fixed seed for determinism in debugging, 
    # but actually, for SAT, random restarts are key. Let's use a global RNG.
    rng = random.Random(42)
    
    best_assignment = None
    best_unsat_count = m + 1
    
    # Parameters
    max_steps_per_restart = 5000
    max_restarts = 100
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Track clause satisfaction status to speed up updates?
        # For N=150, M~450, simple recalculation might be fast enough.
        # Let's try a simple WalkSAT loop.
        
        unsat_clauses = []
        for i, c in enumerate(processed_clauses):
            is_sat = False
            for v, s in c:
                if assignment[v] == (s == 1):
                    is_sat = True
                    break
            if not is_sat:
                unsat_clauses.append(i)
                
        if not unsat_clauses:
            return assignment
            
        # WalkSAT loop
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                return best_assignment if best_assignment else assignment
                
            if not unsat_clauses:
                return assignment
                
            # Pick a random unsatisfied clause
            idx = rng.choice(unsat_clauses)
            clause = processed_clauses[idx]
            
            # Choose a variable to flip
            # Heuristic: Flip the variable that minimizes the number of newly unsatisfied clauses
            # This is the "Walk" part of WalkSAT.
            # With probability p, choose the best variable (minimizes damage).
            # With probability 1-p, choose a random variable from the clause.
            p = 0.55  # Typical value for WalkSAT
            
            if rng.random() < p:
                # Find variable that minimizes newly unsatisfied clauses
                best_var = -1
                best_damage = float('inf')
                
                # We need to know which clauses contain each variable.
                # To avoid building an index every restart, we can just evaluate the cost for vars in the clause.
                # Cost = number of clauses that are currently satisfied but will become unsatisfied if we flip var.
                # Since we only care about the clause we picked, we can just look at the other 2 literals.
                # Actually, the standard WalkSAT heuristic looks at the number of *other* clauses that contain this variable.
                # Pre-computing a variable-to-clauses map is efficient.
                
                # Let's do a quick check: if we flip a variable in the current clause,
                # how many other clauses does it affect?
                # We don't have the map here. Building it is O(M*3).
                # Let's build it once outside the restart loop.
                
                # Wait, I should build the index once.
                pass # Placeholder for logic below
                
            # Since I didn't build the index in this draft, let's refactor to build it once.
            # I will rewrite the solve function to include the index.
            break # Break inner loop to force rewrite with index

    # Fallback return
    return best_assignment if best_assignment else [False] * n

# Refactored solve with proper index and logic
def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    deadline = time.time() + 280.0
    
    # Build variable to clauses index
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for lit in c:
            var_to_clauses[abs(lit) - 1].append(i)
            
    processed_clauses = []
    for c in clauses:
        pc = []
        for lit in c:
            if lit > 0:
                pc.append((lit - 1, 1))
            else:
                pc.append((lit - 1, -1))
        processed_clauses.append(tuple(pc))
        
    m = len(processed_clauses)
    rng = random.Random(42)
    
    best_assignment = None
    best_unsat_count = m + 1
    
    max_steps_per_restart = 2000
    max_restarts = 200
    p_walk = 0.55
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initial check
        unsat_clauses = []
        for i, c in enumerate(processed_clauses):
            is_sat = False
            for v, s in c:
                if assignment[v] == (s == 1):
                    is_sat = True
                    break
            if not is_sat:
                unsat_clauses.append(i)
                
        if not unsat_clauses:
            return assignment
            
        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                if best_assignment is None or len(unsat_clauses) < best_unsat_count:
                    best_assignment = assignment[:]
                    best_unsat_count = len(unsat_clauses)
                return best_assignment
                
            if not unsat_clauses:
                return assignment
                
            # Update best if better
            if len(unsat_clauses) < best_unsat_count:
                best_unsat_count = len(unsat_clauses)
                best_assignment = assignment[:]
                
            # Pick random unsat clause
            idx = rng.choice(unsat_clauses)
            clause = processed_clauses[idx]
            
            # Choose variable to flip
            if rng.random() < p_walk:
                # Greedy: minimize damage
                best_var = -1
                best_damage = float('inf')
                
                for v, s in clause:
                    # Calculate damage: number of clauses containing v that are satisfied and will become unsatisfied
                    # A clause is satisfied if it has at least one true literal.
                    # If we flip v, clauses containing v might change status.
                    # Specifically, if a clause C containing v is currently satisfied ONLY by v, it becomes unsatisfied.
                    # If a clause C containing v is currently unsatisfied, it might become satisfied (if v is the only false one? No, if v becomes true).
                    # We only care about "damage" to satisfied clauses.
                    
                    damage = 0
                    for c_idx in var_to_clauses[v]:
                        if c_idx == idx:
                            continue
                        c = processed_clauses[c_idx]
                        # Check if c is currently satisfied
                        is_sat = False
                        for vv, ss in c:
                            if assignment[vv] == (ss == 1):
                                is_sat = True
                                break
                        if is_sat:
                            # Check if v is the ONLY satisfying literal in c
                            # If so, flipping v makes it unsat.
                            only_sat = True
                            for vv, ss in c:
                                if vv == v:
                                    continue
                                if assignment[vv] == (ss == 1):
                                    only_sat = False
                                    break
                            if only_sat:
                                damage += 1
                                
                    if damage < best_damage:
                        best_damage = damage
                        best_var = v
                        if damage == 0:
                            break # Can't do better than 0
                
                flip_var = best_var
            else:
                # Random walk
                v, s = rng.choice(clause)
                flip_var = v
                
            # Flip
            assignment[flip_var] = not assignment[flip_var]
            
            # Update unsat_clauses efficiently
            # Remove clauses that are now satisfied
            new_unsat = []
            for c_idx in unsat_clauses:
                if c_idx == idx:
                    continue
                # Check if this clause is still unsat
                c = processed_clauses[c_idx]
                is_sat = False
                for v, s in c:
                    if assignment[v] == (s == 1):
                        is_sat = True
                        break
                if not is_sat:
                    new_unsat.append(c_idx)
            
            # Add clauses that were satisfied but are now unsat
            # These are clauses containing flip_var that were NOT in unsat_clauses
            for c_idx in var_to_clauses[flip_var]:
                if c_idx in unsat_clauses:
                    continue
                c = processed_clauses[c_idx]
                is_sat = False
                for v, s in c:
                    if assignment[v] == (s == 1):
                        is_sat = True
                        break
                if not is_sat:
                    new_unsat.append(c_idx)
            
            unsat_clauses = new_unsat
            
    if best_assignment is None:
        return [False] * n
    return best_assignment

# EVOLVE-BLOCK-END
