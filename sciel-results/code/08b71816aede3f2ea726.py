"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    """Improved WalkSAT with incremental clause tracking and proper flip scoring."""
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280.0  # 50s safety margin before 330s kill

    # Pre-extract clause literals as (var_idx, sign) for speed
    clause_data = []
    for clause in clauses:
        clause_data.append([(abs(l) - 1, 1 if l > 0 else -1) for l in clause])

    num_clauses = len(clause_data)

    # Build index: var_idx -> list of (clause_idx, sign_in_that_clause)
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clause_data):
        for var_idx, sign in clause:
            var_to_clauses[var_idx].append((ci, sign))

    def clause_is_satisfied(ci, assignment):
        """Check if clause ci is satisfied under assignment."""
        for var_idx, sign in clause_data[ci]:
            if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                return True
        return False

    def build_unsat_set(assignment):
        """Build set of unsatisfied clause indices."""
        unsat = set()
        for ci in range(num_clauses):
            if not clause_is_satisfied(ci, assignment):
                unsat.add(ci)
        return unsat

    def walk_one(assignment, rng, max_steps):
        """Run WalkSAT with incremental unsat tracking."""
        # Build initial unsatisfied clause set
        unsat_clauses = build_unsat_set(assignment)

        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, False

            if not unsat_clauses:
                return assignment, True

            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_clauses))
            clause = clause_data[ci]

            # For each literal in the clause, compute the net change in unsat count
            # if we flip that variable.
            # Net change = (clauses that become unsat) - (clauses that become sat)
            # We want to minimize this (most negative is best).

            best_var = None
            best_score = float('inf')
            scores = []

            for var_idx, sign in clause:
                # Simulate flipping var_idx
                # Clauses affected: all clauses containing var_idx
                gain = 0  # clauses that go from unsat -> sat
                loss = 0  # clauses that go from sat -> unsat

                for (uci, u_sign) in var_to_clauses[var_idx]:
                    if uci == ci:
                        # This is the chosen clause; it's currently unsat.
                        # After flipping var_idx, does it become sat?
                        # The literal (var_idx, sign) will now be true, so yes.
                        gain += 1
                        continue

                    # Check if this other clause is currently unsat
                    is_unsat = uci in unsat_clauses

                    if is_unsat:
                        # Currently unsat. After flipping var_idx, will it be sat?
                        # It becomes sat if the literal (var_idx, u_sign) becomes true
                        # AND no other literal was already making it sat (but it's unsat,
                        # so no other literal is true).
                        # So it becomes sat iff (var_idx, u_sign) becomes true after flip.
                        # Currently, (var_idx, u_sign) is false (since clause is unsat).
                        # After flip, it becomes true.
                        # But wait: we need to check if flipping makes this literal true.
                        # Currently assignment[var_idx] has some value. After flip it's negated.
                        # The literal (var_idx, u_sign) is true iff
                        #   (assignment[var_idx] and u_sign==1) or (not assignment[var_idx] and u_sign==-1)
                        # After flip, assignment[var_idx] becomes not assignment[var_idx].
                        # So the literal becomes true iff
                        #   (not assignment[var_idx] and u_sign==1) or (assignment[var_idx] and u_sign==-1)
                        # Which is the opposite of the current truth value.
                        # Since the clause is currently unsat, all literals are false,
                        # so this literal is currently false, and after flip it becomes true.
                        # Therefore the clause becomes sat.
                        gain += 1
                    else:
                        # Currently sat. After flipping var_idx, will it become unsat?
                        # It becomes unsat only if the literal (var_idx, u_sign) was the
                        # ONLY true literal in the clause.
                        # Check: is (var_idx, u_sign) currently true?
                        lit_true = (assignment[var_idx] and u_sign == 1) or (not assignment[var_idx] and u_sign == -1)
                        if lit_true:
                            # Check if any other literal in this clause is true
                            other_true = False
                            for (v2, s2) in clause_data[uci]:
                                if v2 == var_idx:
                                    continue
                                if (assignment[v2] and s2 == 1) or (not assignment[v2] and s2 == -1):
                                    other_true = True
                                    break
                            if not other_true:
                                loss += 1

                score = loss - gain  # We want to minimize this
                scores.append(score)

                if score < best_score:
                    best_score = score
                    best_var = var_idx

            # WalkSAT: with probability p, pick randomly; otherwise pick the best
            p = 0.5
            if rng.random() < p:
                chosen_idx = rng.randrange(len(clause))
            else:
                # Pick the variable with the best score
                min_score = min(scores)
                # Among ties, pick randomly
                candidates = [i for i, s in enumerate(scores) if s == min_score]
                chosen_idx = rng.choice(candidates)

            var_idx, sign = clause[chosen_idx]
            old_val = assignment[var_idx]
            assignment[var_idx] = not old_val

            # Update unsat_clauses incrementally
            # Only clauses containing var_idx can change status
            for (uci, u_sign) in var_to_clauses[var_idx]:
                was_unsat = uci in unsat_clauses
                is_now_satisfied = clause_is_satisfied(uci, assignment)
                if was_unsat and is_now_satisfied:
                    unsat_clauses.discard(uci)
                elif not was_unsat and not is_now_satisfied:
                    unsat_clauses.add(uci)

        return assignment, False

    # Multi-restart strategy
    rng = random.Random(0)
    best_assignment = None
    best_unsat_count = float('inf')

    # Parameters
    max_steps_per_restart = 20000
    num_restarts = 200  # max restarts, will stop at deadline

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Try to solve
        assignment, solved = walk_one(assignment, rng, max_steps_per_restart)

        if solved:
            return assignment

        # Track best
        uc = len(build_unsat_set(assignment))
        if uc < best_unsat_count:
            best_unsat_count = uc
            best_assignment = assignment[:]

    # Return best assignment found
    if best_assignment is not None:
        return best_assignment

    # Fallback: all False
    return [False] * n

# EVOLVE-BLOCK-END
