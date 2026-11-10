"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Pre-compute variables in each clause for faster access
    clause_vars = []
    for c in clauses:
        vars_in_c = [abs(l) - 1 for l in c]
        clause_vars.append(vars_in_c)
        
    def is_satisfied(assignment, clause):
        for l in clause:
            if assignment[abs(l) - 1] == (l > 0):
                return True
        return False

    def count_unsat_fast(assignment):
        count = 0
        for c in clauses:
            sat = False
            for l in c:
                if assignment[abs(l) - 1] == (l > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    # Build conflict graph for spectral initialization
    # Edge between var u and var v with weight w if they appear together in a clause
    # The sign of the "conflict" depends on literal signs:
    # If both literals have same sign (both positive or both negative), they conflict
    # (Max-Cut: want them on opposite sides). If opposite signs, they agree (want same side).
    # We build a weighted graph where edge weight reflects the "pressure" for variables to differ.
    
    # For spectral bisection: build Laplacian L = D - A
    # A[u][v] = sum over clauses containing both u,v of:
    #   +1 if literals conflict (same sign) -> want different assignments
    #   -1 if literals agree (opposite sign) -> want same assignment
    # Then Fiedler vector gives initial partition.
    
    A = np.zeros((n, n))
    for c in clauses:
        if len(c) < 2:
            continue
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                li, lj = c[i], c[j]
                ui, uj = abs(li) - 1, abs(lj) - 1
                if ui == uj:
                    continue
                # Determine if literals conflict
                # li > 0 means x_i, li < 0 means not x_i
                # They conflict if (li > 0 and lj > 0) or (li < 0 and lj < 0)
                # i.e., same sign -> they can't both be true -> want different values
                # They agree if opposite signs -> can both be true -> want same value
                if (li > 0) == (lj > 0):
                    # Conflict: want different assignments -> positive edge in cut formulation
                    A[ui][uj] += 1.0
                    A[uj][ui] += 1.0
                else:
                    # Agree: want same assignment -> negative edge
                    A[ui][uj] -= 1.0
                    A[uj][ui] -= 1.0
    
    # Build Laplacian
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    
    # Compute Fiedler vector (second smallest eigenvalue's eigenvector)
    # For n=50, full eigendecomposition is fine
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is eigenvector of second smallest eigenvalue
        fiedler = eigvecs[:, 1]
    except Exception:
        fiedler = np.zeros(n)
    
    # Convert Fiedler vector to initial assignment
    # Positive entries -> True, negative -> False
    # The sign of the Fiedler vector determines the partition
    spectral_assignment = [fiedler[i] > 0 for i in range(n)]
    
    # Also prepare a few alternative spectral initializations with random sign flips
    # to diversify restarts
    spectral_assignments = []
    spectral_assignments.append(list(spectral_assignment))
    spectral_assignments.append([not x for x in spectral_assignment])
    
    # Add a few perturbed spectral assignments
    for _ in range(3):
        perturbed = list(spectral_assignment)
        for i in range(n):
            if random.random() < 0.1:
                perturbed[i] = not perturbed[i]
        spectral_assignments.append(perturbed)
    
    def walksat_efficient(assignment, max_steps, p=0.5):
        current_unsat = count_unsat_fast(assignment)
        if current_unsat == 0:
            return True, assignment
        
        for _ in range(max_steps):
            if current_unsat == 0:
                return True, assignment
            
            while True:
                ci = random.randint(0, len(clauses) - 1)
                if not is_satisfied(assignment, clauses[ci]):
                    break
            
            clause = clauses[ci]
            vars_in_clause = clause_vars[ci]
            
            min_delta = float('inf')
            best_vars = []
            
            for v in vars_in_clause:
                old_val = assignment[v]
                assignment[v] = not old_val
                new_unsat = count_unsat_fast(assignment)
                assignment[v] = old_val
                
                delta = new_unsat - current_unsat
                
                if delta < min_delta:
                    min_delta = delta
                    best_vars = [v]
                elif delta == min_delta:
                    best_vars.append(v)
            
            if random.random() < p:
                v_to_flip = random.choice(vars_in_clause)
            else:
                v_to_flip = random.choice(best_vars)
            
            assignment[v_to_flip] = not assignment[v_to_flip]
            current_unsat = count_unsat_fast(assignment)
            
        return False, assignment

    # Main loop with restarts
    start_time = time.time()
    max_time = 310.0
    
    restart = 0
    best_unsat_count = float('inf')
    best_assignment = None
    
    while time.time() - start_time < max_time:
        restart += 1
        steps = min(1000 * (2 ** (restart - 1)), 100000)
        
        # Choose initial assignment: mix of spectral and random
        if restart <= len(spectral_assignments):
            assignment = list(spectral_assignments[restart - 1])
        elif restart <= len(spectral_assignments) + 5:
            # Use perturbed spectral
            assignment = list(spectral_assignment)
            for i in range(n):
                if random.random() < 0.15:
                    assignment[i] = not assignment[i]
        else:
            # Pure random for diversity
            assignment = [random.random() < 0.5 for _ in range(n)]
        
        init_unsat = count_unsat_fast(assignment)
        if init_unsat < best_unsat_count:
            best_unsat_count = init_unsat
            best_assignment = list(assignment)
        
        solved, final_assignment = walksat_efficient(assignment, steps, p=0.5)
        
        if solved:
            return final_assignment
        
        final_unsat = count_unsat_fast(final_assignment)
        if final_unsat < best_unsat_count:
            best_unsat_count = final_unsat
            best_assignment = list(final_assignment)
            
    if best_assignment is not None:
        return best_assignment
    return assignment
# EVOLVE-BLOCK-END
