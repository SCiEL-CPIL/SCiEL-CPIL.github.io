"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def _phase_seed(n, clauses, rng):
    """Initialize assignment using phase information from clause polarities."""
    pos_count = [0] * n
    neg_count = [0] * n
    for clause in clauses:
        for lit in clause:
            if lit > 0:
                pos_count[lit - 1] += 1
            else:
                neg_count[-lit - 1] += 1
    assignment = []
    for i in range(n):
        if pos_count[i] > neg_count[i]:
            assignment.append(True)
        elif neg_count[i] > pos_count[i]:
            assignment.append(False)
        else:
            assignment.append(rng.random() < 0.5)
    return assignment


def _count_unsat(assignment, clauses):
    """Count number of unsatisfied clauses."""
    count = 0
    for clause in clauses:
        sat = False
        for lit in clause:
            if assignment[abs(lit) - 1] == (lit > 0):
                sat = True
                break
        if not sat:
            count += 1
    return count


def _is_satisfied(assignment, clause):
    """Check if a single clause is satisfied."""
    for lit in clause:
        if assignment[abs(lit) - 1] == (lit > 0):
            return True
    return False


def _walksat(assignment, clauses, n, rng, max_steps, deadline):
    """WalkSAT with random walk probability 0.5."""
    for _ in range(max_steps):
        if time.time() >= deadline:
            return assignment
        # Find unsatisfied clauses
        unsat = []
        for c in clauses:
            if not _is_satisfied(assignment, c):
                unsat.append(c)
        if not unsat:
            return assignment
        # Pick a random unsatisfied clause
        clause = rng.choice(unsat)
        # With probability 0.5, flip a random literal in the clause
        # Otherwise, flip the literal that satisfies the most other clauses
        if rng.random() < 0.5:
            lit = rng.choice(clause)
            var = abs(lit) - 1
            assignment[var] = not assignment[var]
        else:
            best_var = -1
            best_gain = -1
            for lit in clause:
                var = abs(lit) - 1
                # Count how many currently unsatisfied clauses would be satisfied
                # if we set this variable to satisfy this literal
                gain = 0
                target_val = (lit > 0)
                if assignment[var] != target_val:
                    for c in clauses:
                        if _is_satisfied(assignment, c):
                            continue
                        # Check if this clause would be satisfied by setting var to target_val
                        would_sat = False
                        for l2 in c:
                            v2 = abs(l2) - 1
                            if v2 == var:
                                if (l2 > 0) == target_val:
                                    would_sat = True
                                    break
                            else:
                                if assignment[v2] == (l2 > 0):
                                    would_sat = True
                                    break
                        if would_sat:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_var = var
            if best_var >= 0:
                assignment[best_var] = not assignment[best_var]
    return assignment


def _minconflicts(assignment, clauses, n, rng, max_steps, deadline):
    """MinConflicts: pick most violated clause, flip var that minimizes violations."""
    for _ in range(max_steps):
        if time.time() >= deadline:
            return assignment
        # Find all unsatisfied clauses
        unsat_clauses = []
        for c in clauses:
            if not _is_satisfied(assignment, c):
                unsat_clauses.append(c)
        if not unsat_clauses:
            return assignment
        # Pick a random unsatisfied clause
        clause = rng.choice(unsat_clauses)
        # For each literal in the clause, compute the cost of flipping that variable
        best_var = -1
        best_cost = float('inf')
        for lit in clause:
            var = abs(lit) - 1
            # Temporarily flip and count violations
            old_val = assignment[var]
            assignment[var] = not assignment[var]
            cost = 0
            for c in clauses:
                if not _is_satisfied(assignment, c):
                    cost += 1
            assignment[var] = old_val
            if cost < best_cost:
                best_cost = cost
                best_var = var
            elif cost == best_cost and rng.random() < 0.5:
                best_var = var
        if best_var >= 0:
            assignment[best_var] = not assignment[best_var]
    return assignment


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 280  # Leave 50s margin before 330s kill
    
    # Precompute clause structure for faster access
    # clauses is a list of lists of signed 1-based literals
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Strategy 1: Phase-seeded WalkSAT with multiple restarts
    for attempt in range(50):
        if time.time() >= deadline:
            break
        seed = attempt * 7919 + 12345
        rng = random.Random(seed)
        assignment = _phase_seed(n, clauses, rng)
        
        # Initial check
        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        # WalkSAT with increasing step budget
        steps = 1000 + attempt * 200
        assignment = _walksat(assignment, clauses, n, rng, steps, deadline)
        
        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
    
    # Strategy 2: MinConflicts from best assignment found so far
    if best_assignment is not None:
        rng = random.Random(99999)
        assignment = best_assignment[:]
        assignment = _minconflicts(assignment, clauses, n, rng, 5000, deadline)
        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
    
    # Strategy 3: Pure random WalkSAT from scratch with different seeds
    for attempt in range(20):
        if time.time() >= deadline:
            break
        seed = 100000 + attempt * 31337
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        assignment = _walksat(assignment, clauses, n, rng, 2000, deadline)
        
        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
    
    # Final: return best assignment found
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: all False
    return [False] * n
# EVOLVE-BLOCK-END
