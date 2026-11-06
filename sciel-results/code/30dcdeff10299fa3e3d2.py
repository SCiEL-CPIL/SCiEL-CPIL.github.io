"""DPLL-based planted 3-SAT solver with WalkSAT fallback."""
import time
import random


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    deadline = time.time() + 280.0  # safety margin before 330s kill

    # Convert to 0-based signed literals
    cnf = [[l - 1 if l > 0 else l for l in c] for c in clauses]

    result = _dpll(n, cnf, deadline)
    if result is not None:
        return result

    # Fallback: WalkSAT with multiple restarts
    return _walksat(n, clauses, deadline)


def _dpll(n, cnf, deadline):
    """Deterministic DPLL with unit propagation and pure literal elimination.
    Returns a satisfying assignment (list of booleans) or None."""
    # assignment: 0 = unset, 1 = True, -1 = False (per variable, 0-based)
    assignment = [0] * n

    def unit_propagate(clauses, assignment):
        """Propagate unit clauses. Returns (new_clauses, assignment, ok)."""
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for c in clauses:
                # Evaluate clause
                sat = False
                unassigned = []
                for l in c:
                    v = assignment[abs(l)]
                    if v != 0:
                        if (v == 1 and l > 0) or (v == -1 and l < 0):
                            sat = True
                            break
                    else:
                        unassigned.append(l)
                if sat:
                    continue  # clause satisfied, drop it
                if len(unassigned) == 0:
                    return None, None, False  # empty clause -> conflict
                if len(unassigned) == 1:
                    l = unassigned[0]
                    var = abs(l)
                    val = 1 if l > 0 else -1
                    if assignment[var] == 0:
                        assignment[var] = val
                        changed = True
                    elif assignment[var] != val:
                        return None, None, False  # conflict
                else:
                    new_clauses.append(unassigned)
            clauses = new_clauses
        return clauses, assignment, True

    def pure_literal_eliminate(clauses, assignment):
        """Set pure literals. Returns (new_clauses, assignment, changed)."""
        # Collect which polarity each variable appears in among unassigned
        pos = set()
        neg = set()
        for c in clauses:
            for l in c:
                if assignment[abs(l)] == 0:
                    if l > 0:
                        pos.add(abs(l))
                    else:
                        neg.add(abs(l))
        pure = pos - neg  # only positive
        pure |= neg - pos  # only negative
        if not pure:
            return clauses, assignment, False
        for var in pure:
            if var in (pos - neg):
                assignment[var] = 1
            else:
                assignment[var] = -1
        # Re-filter
        new_clauses = []
        for c in clauses:
            sat = False
            unassigned = []
            for l in c:
                v = assignment[abs(l)]
                if v != 0:
                    if (v == 1 and l > 0) or (v == -1 and l < 0):
                        sat = True
                        break
                else:
                    unassigned.append(l)
            if sat:
                continue
            if len(unassigned) == 0:
                return None, None, True  # conflict
            new_clauses.append(unassigned)
        return new_clauses, assignment, True

    def choose_var(clauses):
        """Pick variable with highest occurrence count."""
        counts = {}
        for c in clauses:
            seen = set()
            for l in c:
                if abs(l) not in seen:
                    counts[abs(l)] = counts.get(abs(l), 0) + 1
                    seen.add(abs(l))
        if not counts:
            return None
        return max(counts, key=lambda v: (counts[v], -v))

    def dpll(clauses, assignment):
        # Time guard
        if time.time() > deadline:
            return None

        # Unit propagation
        clauses, assignment, ok = unit_propagate(clauses, assignment)
        if not ok:
            return None
        if not clauses:
            # All satisfied
            return [assignment[i] == 1 for i in range(n)]

        # Pure literal elimination (loop a few times)
        for _ in range(3):
            clauses, assignment, changed = pure_literal_eliminate(clauses, assignment)
            if clauses is None:
                return None
            if not changed:
                break
            if not clauses:
                return [assignment[i] == 1 for i in range(n)]
            # Re-propagate after pure literals
            clauses, assignment, ok = unit_propagate(clauses, assignment)
            if not ok:
                return None
            if not clauses:
                return [assignment[i] == 1 for i in range(n)]

        # Choose a variable
        var = choose_var(clauses)
        if var is None:
            return [assignment[i] == 1 for i in range(n)]

        # Try True first (planted instances often biased), then False
        for val in (1, -1):
            new_assignment = assignment[:]
            new_assignment[var] = val
            # Reduce clauses
            new_clauses = []
            conflict = False
            for c in clauses:
                sat = False
                unassigned = []
                for l in c:
                    if abs(l) == var:
                        if (val == 1 and l > 0) or (val == -1 and l < 0):
                            sat = True
                            break
                    else:
                        unassigned.append(l)
                if sat:
                    continue
                if not unassigned:
                    conflict = True
                    break
                new_clauses.append(unassigned)
            if conflict:
                continue
            res = dpll(new_clauses, new_assignment)
            if res is not None:
                return res
        return None

    return dpll(cnf, assignment)


def _walksat(n, clauses, deadline):
    """WalkSAT with random restarts until deadline."""
    rng = random.Random(12345)
    best_assignment = None
    best_unsat = None
    best_count = None

    while time.time() < deadline - 5.0:
        # Fresh random restart
        assignment = [rng.random() < 0.5 for _ in range(n)]
        # Count unsat
        def count_unsat(assignment):
            cnt = 0
            for c in clauses:
                if not any(assignment[abs(l) - 1] == (l > 0) for l in c):
                    cnt += 1
            return cnt

        cur = count_unsat(assignment)
        if cur < (best_count if best_count is not None else float('inf')):
            best_count = cur
            best_assignment = assignment[:]
        if cur == 0:
            return assignment

        # Walk
        for step in range(10000):
            if time.time() > deadline - 5.0:
                break
            unsat = [c for c in clauses if not any(assignment[abs(l) - 1] == (l > 0) for l in c)]
            if not unsat:
                return assignment
            clause = rng.choice(unsat)
            # Flip: with prob 0.5 pick a literal that would satisfy, else random
            if rng.random() < 0.5:
                literal = rng.choice(clause)
            else:
                # pick random literal, but prefer unsatisfied
                literal = rng.choice(clause)
            assignment[abs(literal) - 1] = not assignment[abs(literal) - 1]
            cur = count_unsat(assignment)
            if cur < (best_count if best_count is not None else float('inf')):
                best_count = cur
                best_assignment = assignment[:]
            if cur == 0:
                return assignment

    # Return best found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
