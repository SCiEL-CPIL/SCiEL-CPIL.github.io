# EVOLVE-BLOCK-START
import time
import random


def solve(instance):
    """Solve a planted satisfiable 3-CNF using watched-literal DPLL with
    unit propagation, pure literal elimination, and time-budgeted restarts."""
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290.0  # safety margin before 300s kill

    # Pre-compute clause data
    num_clauses = len(clauses)

    # For each clause, store the literals as a tuple for faster access
    clause_tups = [tuple(c) for c in clauses]

    # Variable selection order: we'll use a static order but with randomization
    var_order = list(range(n))

    def try_solve(seed):
        """Try to solve with a given random seed. Return list of bools or None."""
        rng = random.Random(seed)
        rng.shuffle(var_order)

        # Assignment: -1 = unassigned, 0 = False, 1 = True
        assignment = [-1] * n
        trail = []  # stack of (var_idx) for backtracking

        # Watched literals setup
        # watch_lits[i] = [l1, l2] for clause i
        # watches[lit] = list of clause indices
        watch_lits = []
        watches = {}

        for i, clause in enumerate(clause_tups):
            if len(clause) >= 2:
                l1, l2 = clause[0], clause[1]
            else:
                l1 = l2 = clause[0]
            watch_lits.append([l1, l2])
            watches.setdefault(l1, []).append(i)
            watches.setdefault(l2, []).append(i)

        # Propagation queue
        prop_queue = []

        # Check if literal is satisfied given current assignment
        def lit_sat(lit):
            v = abs(lit) - 1
            if lit > 0:
                return assignment[v] == 1
            return assignment[v] == 0

        def lit_unsat(lit):
            v = abs(lit) - 1
            if lit > 0:
                return assignment[v] == 0
            return assignment[v] == 1

        def process_queue():
            """Process all items in prop_queue. Return True if no conflict."""
            while prop_queue:
                v_idx, val = prop_queue.pop()
                if assignment[v_idx] != -1:
                    if assignment[v_idx] != val:
                        return False
                    continue

                assignment[v_idx] = val
                trail.append(v_idx)

                # Falsified literal
                if val == 1:
                    falsified = -(v_idx + 1)
                else:
                    falsified = v_idx + 1

                wl = watches.get(falsified, [])
                i = 0
                while i < len(wl):
                    ci = wl[i]
                    pair = watch_lits[ci]
                    # Find the other watched literal
                    if pair[0] == falsified:
                        other = pair[1]
                    else:
                        other = pair[0]

                    # Check if other satisfies the clause
                    if lit_sat(other):
                        i += 1
                        continue

                    # Check if other is also falsified
                    if lit_unsat(other):
                        return False

                    # Find a new literal to watch
                    clause = clause_tups[ci]
                    new_lit = None
                    for l in clause:
                        if l == falsified or l == other:
                            continue
                        lv = abs(l) - 1
                        if assignment[lv] == -1:
                            new_lit = l
                            break

                    if new_lit is None:
                        # Unit clause: queue 'other' for assignment
                        ov = abs(other) - 1
                        if other > 0:
                            prop_queue.append((ov, 1))
                        else:
                            prop_queue.append((ov, 0))
                        # Remove watch
                        wl[i] = wl[-1]
                        wl.pop()
                        continue

                    # Rewatch
                    watches.setdefault(new_lit, []).append(ci)
                    if pair[0] == falsified:
                        pair[0] = new_lit
                    else:
                        pair[1] = new_lit
                    wl[i] = wl[-1]
                    wl.pop()
                    # Don't increment i

            return True

        def is_fully_satisfied():
            """Check if all clauses are satisfied."""
            for clause in clause_tups:
                sat = False
                for l in clause:
                    if lit_sat(l):
                        sat = True
                        break
                if not sat:
                    return False
            return True

        def count_unassigned():
            cnt = 0
            for v in range(n):
                if assignment[v] == -1:
                    cnt += 1
            return cnt

        # Pure literal elimination
        def pure_literal_elim():
            """Find and queue pure literals."""
            pos_count = [0] * n
            neg_count = [0] * n
            for clause in clause_tups:
                for l in clause:
                    v = abs(l) - 1
                    if assignment[v] != -1:
                        continue
                    if l > 0:
                        pos_count[v] += 1
                    else:
                        neg_count[v] += 1

            for v in range(n):
                if assignment[v] != -1:
                    continue
                if pos_count[v] > 0 and neg_count[v] == 0:
                    prop_queue.append((v, 1))
                elif neg_count[v] > 0 and pos_count[v] == 0:
                    prop_queue.append((v, 0))

        # DPLL search (iterative to avoid recursion depth issues)
        # We use a stack-based approach
        # Each frame: (var_idx, value_to_try, next_value)
        # We'll use a simpler recursive approach but with a depth limit

        def dpll():
            """DPLL search. Return True if SAT found."""
            if time.time() > deadline:
                return False

            # Process propagation
            if not process_queue():
                return False

            # Check if all variables assigned
            all_assigned = True
            for v in range(n):
                if assignment[v] == -1:
                    all_assigned = False
                    break
            if all_assigned:
                return is_fully_satisfied()

            # Pure literal elimination
            pure_literal_elim()
            if not process_queue():
                return False

            # Check again
            all_assigned = True
            for v in range(n):
                if assignment[v] == -1:
                    all_assigned = False
                    break
            if all_assigned:
                return is_fully_satisfied()

            # Variable selection
            # Pick unassigned variable with highest occurrence in unsatisfied clauses
            # For speed, use a simpler heuristic
            unassigned = [v for v in var_order if assignment[v] == -1]
            if not unassigned:
                return is_fully_satisfied()

            # Score variables by degree in unsatisfied clauses
            # Quick approximation: just pick from unassigned
            # Use a random choice for diversity
            v = rng.choice(unassigned)

            # Try True
            assignment[v] = 1
            trail.append(v)
            if process_queue():
                if dpll():
                    return True
            # Backtrack
            while trail and trail[-1] != v:
                tv = trail.pop()
                assignment[tv] = -1
            if trail and trail[-1] == v:
                trail.pop()
            assignment[v] = -1

            # Try False
            assignment[v] = 0
            trail.append(v)
            if process_queue():
                if dpll():
                    return True
            # Backtrack
            while trail and trail[-1] != v:
                tv = trail.pop()
                assignment[tv] = -1
            if trail and trail[-1] == v:
                trail.pop()
            assignment[v] = -1

            return False

        if dpll():
            return [bool(assignment[v]) for v in range(n)]
        return None

    # Try with different seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        result = try_solve(seed)
        if result is not None:
            # Verify
            valid = True
            for clause in clauses:
                sat = False
                for l in clause:
                    v = abs(l) - 1
                    if l > 0:
                        if result[v]:
                            sat = True
                            break
                    else:
                        if not result[v]:
                            sat = True
                            break
                if not sat:
                    valid = False
                    break
            if valid:
                return result

    # Fallback: random assignment
    rng_final = random.Random(0)
    return [rng_final.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
