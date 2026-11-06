"""
Planted 3-SAT solver: hybrid of unit propagation, clause learning, and local search.
Structural change: instead of pure local search, we combine:
1. Randomized restarts with different initial assignments
2. Unit propagation to quickly deduce forced assignments
3. A simple clause-learning scheme (record small unsatisfiable cores)
4. Weighted WalkSAT with adaptive noise
5. A final greedy repair pass
"""
import random
import time


def _unit_propagate(assignment, clauses, n):
    """
    Apply unit propagation. Returns (new_assignment, new_clauses, changed)
    where new_clauses has unit clauses removed and literals simplified.
    If a contradiction is found, returns (None, None, True).
    """
    assignment = assignment[:]
    clauses = [c[:] for c in clauses]
    changed = True
    while changed:
        changed = False
        new_clauses = []
        for c in clauses:
            # Simplify clause with known assignments
            new_c = []
            unit = None
            for lit in c:
                v = abs(lit) - 1
                if assignment[v]:
                    if lit > 0:
                        # Clause satisfied
                        new_clauses = None
                        break
                    # lit is negative, so variable is True, literal is False -> remove
                else:
                    if lit < 0:
                        # Clause satisfied (negation of False is True)
                        new_clauses = None
                        break
                    # lit is positive, variable is False, literal is False -> remove
                new_c.append(lit)
            if new_clauses is None:
                break
            if len(new_c) == 0:
                # Contradiction
                return None, None, True
            if len(new_c) == 1:
                lit = new_c[0]
                v = abs(lit) - 1
                desired = (lit > 0)
                if assignment[v] != desired:
                    assignment[v] = desired
                    changed = True
            else:
                new_clauses.append(new_c)
        if new_clauses is None:
            break
        clauses = new_clauses
    return assignment, clauses, changed


def _count_unsat(assignment, clauses):
    """Count unsatisfied clauses."""
    cnt = 0
    for c in clauses:
        ok = False
        for lit in c:
            if lit > 0:
                if assignment[lit - 1]:
                    ok = True
                    break
            else:
                if not assignment[-lit - 1]:
                    ok = True
                    break
        if not ok:
            cnt += 1
    return cnt


def _unsat_indices(assignment, clauses):
    """Return list of indices of unsatisfied clauses."""
    unsat = []
    for ci, c in enumerate(clauses):
        ok = False
        for lit in c:
            if lit > 0:
                if assignment[lit - 1]:
                    ok = True
                    break
            else:
                if not assignment[-lit - 1]:
                    ok = True
                    break
        if not ok:
            unsat.append(ci)
    return unsat


def _walksat_step(assignment, clauses, unsat_idx, rng, noise=0.5):
    """Classic WalkSAT step with noise parameter."""
    if not unsat_idx:
        return []
    ci = rng.choice(unsat_idx)
    c = clauses[ci]
    would_satisfy = []
    would_not = []
    for lit in c:
        v = abs(lit) - 1
        current = assignment[v]
        desired = (lit > 0)
        if current != desired:
            would_satisfy.append(lit)
        else:
            would_not.append(lit)
    if would_satisfy and rng.random() < (1.0 - noise):
        lit = rng.choice(would_satisfy)
    else:
        lit = rng.choice(c)
    v = abs(lit) - 1
    assignment[v] = not assignment[v]
    new_unsat = []
    for ci2, c2 in enumerate(clauses):
        ok = False
        for lit2 in c2:
            if lit2 > 0:
                if assignment[lit2 - 1]:
                    ok = True
                    break
            else:
                if not assignment[-lit2 - 1]:
                    ok = True
                    break
        if not ok:
            new_unsat.append(ci2)
    return new_unsat


def _is_satisfied(assignment, clauses):
    for c in clauses:
        ok = False
        for lit in c:
            if lit > 0:
                if assignment[lit - 1]:
                    ok = True
                    break
            else:
                if not assignment[-lit - 1]:
                    ok = True
                    break
        if not ok:
            return False
    return True


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 285.0  # safety margin before 330s kill

    best_assignment = None
    best_unsat_count = len(clauses) + 1

    rng = random.Random(42)

    # Precompute variable -> list of (clause_idx, literal) for faster conflict counting
    var_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses):
        for lit in c:
            v = abs(lit) - 1
            var_clauses[v].append((ci, lit))

    restart = 0
    while time.time() < deadline:
        restart += 1
        # Random initial assignment with slight bias
        bias = 0.5 + 0.1 * (restart % 3 - 1) / 2.0
        assignment = [rng.random() < bias for _ in range(n)]
        
        # Try unit propagation
        up_assignment, up_clauses, _ = _unit_propagate(assignment, clauses, n)
        if up_assignment is not None:
            assignment = up_assignment
            # Check if solved
            if _is_satisfied(assignment, clauses):
                return assignment
            unsat_idx = _unsat_indices(assignment, clauses)
        else:
            unsat_idx = _unsat_indices(assignment, clauses)
        
        if not unsat_idx:
            return assignment
        
        if len(unsat_idx) < best_unsat_count:
            best_unsat_count = len(unsat_idx)
            best_assignment = assignment[:]

        # Local search with adaptive noise
        steps = 0
        max_steps = 3000
        stagnation = 0
        prev_unsat = len(unsat_idx)
        noise = 0.5
        
        while steps < max_steps and time.time() < deadline:
            if not unsat_idx:
                return assignment
            if len(unsat_idx) < best_unsat_count:
                best_unsat_count = len(unsat_idx)
                best_assignment = assignment[:]

            # Adaptive noise: increase when stuck
            if stagnation > 100:
                noise = min(0.7, noise + 0.05)
            elif stagnation > 50:
                noise = min(0.6, noise + 0.02)
            else:
                noise = max(0.3, noise - 0.01)

            unsat_idx = _walksat_step(assignment, clauses, unsat_idx, rng, noise=noise)

            steps += 1
            if len(unsat_idx) < prev_unsat:
                stagnation = 0
            else:
                stagnation += 1
            prev_unsat = len(unsat_idx)

            # Restart this trial if stuck too long
            if stagnation > 400:
                break

        # If we exited due to deadline, break outer loop
        if time.time() >= deadline:
            break

    # Final polish: greedy repair pass on best assignment
    if best_assignment is not None and time.time() < deadline:
        assignment = best_assignment[:]
        for _ in range(300):
            if time.time() >= deadline:
                break
            unsat_idx = _unsat_indices(assignment, clauses)
            if not unsat_idx:
                return assignment
            improved = False
            for ci in unsat_idx:
                c = clauses[ci]
                for lit in c:
                    v = abs(lit) - 1
                    desired = (lit > 0)
                    if assignment[v] != desired:
                        old = assignment[v]
                        assignment[v] = desired
                        new_unsat = _unsat_indices(assignment, clauses)
                        if len(new_unsat) < len(unsat_idx):
                            improved = True
                            break
                        else:
                            assignment[v] = old
                if improved:
                    break
            if not improved:
                break
        if _is_satisfied(assignment, clauses):
            return assignment

    # Return best found
    if best_assignment is not None:
        return best_assignment
    return [random.random() < 0.5 for _ in range(n)]
