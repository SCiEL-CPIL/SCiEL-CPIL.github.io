"""Rewritten solver: multi-restart min-conflicts local search for planted 3-SAT."""
# EVOLVE-BLOCK-START
import random
import time


def _satisfied_count(assignment, clauses, c_pos, c_neg, n):
    """Return number of satisfied clauses using precomputed pos/neg literal index maps."""
    cnt = 0
    for c in clauses:
        # a clause is satisfied if any literal is true
        ok = False
        for lit in c:
            if lit > 0:
                if assignment[c_pos[lit] - 1]:
                    ok = True
                    break
            else:
                if not assignment[c_neg[-lit] - 1]:
                    ok = True
                    break
        if ok:
            cnt += 1
    return cnt


def _build_clause_index(clauses, n):
    """Precompute variable -> list of (clause_idx, literal) occurrences for fast counting."""
    var_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses):
        for lit in c:
            v = abs(lit)
            var_clauses[v - 1].append((ci, lit))
    return var_clauses


def _unsat_clause_indices(assignment, clauses, c_pos, c_neg):
    unsat = []
    for ci, c in enumerate(clauses):
        ok = False
        for lit in c:
            if lit > 0:
                if assignment[c_pos[lit] - 1]:
                    ok = True
                    break
            else:
                if not assignment[c_neg[-lit] - 1]:
                    ok = True
                    break
        if not ok:
            unsat.append(ci)
    return unsat


def _min_conflicts_step(assignment, clauses, var_clauses, unsat_idx, unsat_set, rng,
                        phase_bias=0.7):
    """
    One min-conflicts step:
      - Count how many unsat clauses each variable appears in (and how many of those
        it would satisfy if flipped to the 'good' value).
      - Pick a variable with high conflict count; with probability phase_bias flip it
        to the value that satisfies the most of its unsat clauses, else flip randomly.
    Returns (new_unsat_idx, new_unsat_set) or None if solved.
    """
    n = len(assignment)
    # Conflict counts per variable: number of unsat clauses containing the variable.
    conflict = [0] * n
    # For each variable, track the best flip value and how many unsat clauses it'd fix.
    best_fix = [0] * n   # count of unsat clauses satisfied if set to True
    best_fix_false = [0] * n  # count satisfied if set to False
    for ci in unsat_idx:
        c = clauses[ci]
        for lit in c:
            v = abs(lit) - 1
            conflict[v] += 1
            if lit > 0:
                # literal true when assignment[v] is True
                best_fix[v] += 1
            else:
                best_fix_false[v] += 1

    # Choose a variable among those with max conflict (or top few) to flip.
    max_conf = max(conflict) if conflict else 0
    if max_conf == 0:
        return None  # solved

    # Pick among variables with conflict == max_conf (or >= max_conf - 1 for diversity)
    threshold = max_conf
    candidates = [v for v in range(n) if conflict[v] >= threshold]
    if not candidates:
        candidates = [v for v in range(n) if conflict[v] > 0]
    v = rng.choice(candidates)

    # Decide new value
    if rng.random() < phase_bias:
        # choose the value that fixes more unsat clauses
        if best_fix[v] >= best_fix_false[v]:
            assignment[v] = True
        else:
            assignment[v] = False
    else:
        assignment[v] = not assignment[v]

    # Recompute unsat set (incremental is possible but keep simple; n small)
    new_unsat = []
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
            new_unsat.append(ci)
    return new_unsat


def _walksat_step(assignment, clauses, unsat_idx, rng, noise=0.5):
    """Classic WalkSAT step: pick a random unsat clause, flip a literal (biased to satisfying)."""
    if not unsat_idx:
        return []
    ci = rng.choice(unsat_idx)
    c = clauses[ci]
    # Split literals into those that would satisfy if flipped, and those that wouldn't
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
    # Recompute unsat
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
    deadline = time.time() + 290.0  # safety margin before 330s kill

    # Precompute literal -> variable index maps
    c_pos = {lit: lit for lit in range(1, n + 1)}
    c_neg = {lit: lit for lit in range(1, n + 1)}

    var_clauses = _build_clause_index(clauses, n)

    best_assignment = None
    best_unsat_count = len(clauses) + 1

    rng = random.Random(12345)

    # Multi-restart loop
    restart = 0
    while time.time() < deadline:
        restart += 1
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        unsat_idx = _unsat_clause_indices(assignment, clauses, c_pos, c_neg)
        if not unsat_idx:
            return assignment
        if len(unsat_idx) < best_unsat_count:
            best_unsat_count = len(unsat_idx)
            best_assignment = assignment[:]

        # Local search: alternate between min-conflicts and walksat phases
        steps = 0
        max_steps = 2000
        stagnation = 0
        prev_unsat = len(unsat_idx)
        while steps < max_steps and time.time() < deadline:
            if not unsat_idx:
                return assignment
            if len(unsat_idx) < best_unsat_count:
                best_unsat_count = len(unsat_idx)
                best_assignment = assignment[:]

            # Phase selection: use min-conflicts when many unsat, walksat when few
            if len(unsat_idx) > n * 0.3:
                result = _min_conflicts_step(
                    assignment, clauses, var_clauses, unsat_idx, None, rng,
                    phase_bias=0.7
                )
                if result is None:
                    return assignment
                unsat_idx = result
            else:
                unsat_idx = _walksat_step(assignment, clauses, unsat_idx, rng, noise=0.45)

            steps += 1
            if len(unsat_idx) < prev_unsat:
                stagnation = 0
            else:
                stagnation += 1
            prev_unsat = len(unsat_idx)

            # Restart this trial if stuck too long
            if stagnation > 300:
                break

        # If we exited due to deadline, break outer loop
        if time.time() >= deadline:
            break

    # Final polish: try to fix remaining unsat clauses with a greedy pass
    if best_assignment is not None and time.time() < deadline:
        assignment = best_assignment[:]
        for _ in range(500):
            if time.time() >= deadline:
                break
            unsat_idx = _unsat_clause_indices(assignment, clauses, c_pos, c_neg)
            if not unsat_idx:
                return assignment
            # For each unsat clause, try flipping a literal to satisfy it
            improved = False
            for ci in unsat_idx:
                c = clauses[ci]
                # Try each literal flip that would satisfy this clause
                for lit in c:
                    v = abs(lit) - 1
                    desired = (lit > 0)
                    if assignment[v] != desired:
                        # Flipping v to desired satisfies this clause; check net effect
                        old = assignment[v]
                        assignment[v] = desired
                        new_unsat = _unsat_clause_indices(assignment, clauses, c_pos, c_neg)
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
    # Fallback: random
    return [random.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
