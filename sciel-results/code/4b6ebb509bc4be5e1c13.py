"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 280  # 300s limit, 20s safety margin

    # Precompute: for each variable, list of (clause_idx, literal_value) where
    # literal_value is True if the clause contains +var, False if -var.
    var_occurrences = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            var_occurrences[var].append((ci, lit > 0))

    def random_walksat(seed):
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # sat_count[c] = number of satisfied literals in clause c
        sat_count = [0] * m
        for ci, clause in enumerate(clauses):
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    sat_count[ci] += 1

        unsat_set = set(i for i in range(m) if sat_count[i] == 0)

        # If already satisfied
        if not unsat_set:
            return assignment

        # WalkSAT with clause-count tracking
        # Parameters: noise probability, max steps
        noise = 0.3
        max_steps = 500000

        for step in range(max_steps):
            if time.time() > deadline:
                break
            if not unsat_set:
                break

            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_set))
            clause = clauses[ci]

            # Score each literal in the clause:
            # gain = number of other unsat clauses that would become satisfied
            # loss = number of currently-sat clauses that would become unsat
            best_var = -1
            best_score = -1
            best_is_noise = False

            # Evaluate each literal in the clause
            candidates = []
            for lit in clause:
                var = abs(lit) - 1
                want_val = lit > 0
                # If we set assignment[var] = want_val:
                # For each other clause containing this var:
                #   if currently sat_count == 1 and the only sat literal is this var with current value:
                #     flipping makes it unsat (loss)
                #   if currently sat_count == 0:
                #     if this literal would be satisfied after flip: gain
                gain = 0
                loss = 0
                for (oc, lit_val) in var_occurrences[var]:
                    if oc == ci:
                        continue
                    if lit_val == want_val:
                        # This literal would be satisfied after flip
                        if sat_count[oc] == 0:
                            gain += 1
                    else:
                        # This literal would be unsatisfied after flip
                        if sat_count[oc] == 1:
                            loss += 1
                score = gain - loss
                candidates.append((score, var, want_val))

            # With probability noise, pick random; else pick best
            if rng.random() < noise:
                _, var, want_val = rng.choice(candidates)
            else:
                best = max(candidates, key=lambda x: x[0])
                var, want_val = best[1], best[2]

            # Apply flip
            old_val = assignment[var]
            if old_val == want_val:
                # No change needed
                continue
            assignment[var] = want_val

            # Update sat_counts for all clauses containing var
            for (oc, lit_val) in var_occurrences[var]:
                if lit_val == want_val:
                    # This literal is now satisfied
                    sat_count[oc] += 1
                    if sat_count[oc] == 1:
                        unsat_set.discard(oc)
                else:
                    # This literal is now unsatisfied
                    sat_count[oc] -= 1
                    if sat_count[oc] == 0:
                        unsat_set.add(oc)

            if not unsat_set:
                break

        return assignment

    # Try multiple seeds/restarts with different noise levels
    best_assignment = None
    best_unsat = float('inf')

    # First try with default params
    result = random_walksat(0)
    unsat_count = sum(1 for c in clauses if not any(
        result[abs(l)-1] == (l > 0) for l in c))
    if unsat_count == 0:
        return result
    best_assignment = result
    best_unsat = unsat_count

    # Try more seeds with varying noise if time permits
    for seed in range(1, 50):
        if time.time() > deadline - 5:
            break
        result = random_walksat(seed)
        unsat_count = sum(1 for c in clauses if not any(
            result[abs(l)-1] == (l > 0) for l in c))
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    return best_assignment
# EVOLVE-BLOCK-END
