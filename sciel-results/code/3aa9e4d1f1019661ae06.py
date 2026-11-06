"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)

    # Preprocess: convert to 0-based variable indices and sign arrays
    cvars = []
    csigns = []
    for cl in clauses:
        vs = [abs(l) - 1 for l in cl]
        ss = [l > 0 for l in cl]
        cvars.append(vs)
        csigns.append(ss)

    # Build variable -> clause index lists
    var_clauses = [[] for _ in range(n)]
    for ci, vs in enumerate(cvars):
        for v in vs:
            var_clauses[v].append(ci)

    # Track unsat count per clause (0 = satisfied, 1+ = number of false literals)
    # For 3-SAT: unsat_count[ci] = 3 means all 3 literals false
    # We'll track which literals are false to know when a clause becomes satisfied

    def init_unsat(assignment):
        """Initialize unsat counts and false literal tracking."""
        unsat = [0] * m
        false_lits = [[False] * 3 for _ in range(m)]  # false_lits[ci][j] = literal j is false
        for ci in range(m):
            vs = cvars[ci]
            ss = csigns[ci]
            cnt = 0
            for j in range(3):
                v = vs[j]
                lit_true = assignment[v] == ss[j]
                if not lit_true:
                    cnt += 1
                    false_lits[ci][j] = True
            unsat[ci] = cnt
        return unsat, false_lits

    def is_satisfied_all(unsat):
        return all(u == 0 for u in unsat)

    def count_unsat(unsat):
        return sum(1 for u in unsat if u > 0)

    def flip_var(v, assignment, unsat, false_lits):
        """Flip variable v and update unsat counts. Returns old value."""
        old_val = assignment[v]
        assignment[v] = not old_val
        for ci in var_clauses[v]:
            vs = cvars[ci]
            ss = csigns[ci]
            # Find which position j has variable v
            for j in range(3):
                if vs[j] == v:
                    was_false = false_lits[ci][j]
                    # After flip, the literal truth value inverts
                    # If it was false, now true (and vice versa)
                    false_lits[ci][j] = not was_false
                    if was_false:
                        unsat[ci] -= 1
                    else:
                        unsat[ci] += 1
                    break
        return old_val

    rng = random.Random(0)
    deadline = time.time() + 310.0  # safety margin

    best_assignment = None
    best_unsat_count = m + 1

    # Multiple restarts with adaptive parameters
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        unsat, false_lits = init_unsat(assignment)
        cur_unsat = count_unsat(unsat)

        if cur_unsat == 0:
            return assignment

        if cur_unsat < best_unsat_count:
            best_unsat_count = cur_unsat
            best_assignment = assignment[:]

        # WalkSAT with clause-weighted variable selection
        # Use a noise parameter that increases as we get stuck
        max_steps = 5000 + restart * 2000
        noise_start = 0.1
        noise_end = 0.5

        for step in range(max_steps):
            if time.time() > deadline:
                break

            if cur_unsat == 0:
                return assignment

            if cur_unsat < best_unsat_count:
                best_unsat_count = cur_unsat
                best_assignment = assignment[:]

            # Pick a random unsatisfied clause
            # Find an unsat clause
            while True:
                ci = rng.randrange(m)
                if unsat[ci] > 0:
                    break

            # Choose a variable to flip from this clause
            # With probability noise, pick randomly; otherwise pick the one that minimizes unsat count
            noise = noise_start + (noise_end - noise_start) * (step / max_steps)

            if rng.random() < noise:
                # Random flip
                j = rng.randrange(3)
                v = cvars[ci][j]
            else:
                # Greedy: pick variable whose flip reduces unsat count the most
                best_delta = 999
                best_v = cvars[ci][0]
                for j in range(3):
                    v = cvars[ci][j]
                    # Compute delta: how many clauses change from unsat to sat minus sat to unsat
                    delta = 0
                    for ci2 in var_clauses[v]:
                        if unsat[ci2] == 0:
                            # Currently satisfied; if we flip, it might become unsat
                            # Check if this clause has another true literal besides v
                            vs2 = cvars[ci2]
                            ss2 = csigns[ci2]
                            other_true = False
                            for k in range(3):
                                if vs2[k] != v:
                                    if assignment[vs2[k]] == ss2[k]:
                                        other_true = True
                                        break
                            if not other_true:
                                delta += 1  # this clause will become unsat
                        else:
                            # Currently unsat; if we flip v, it might become sat
                            # Check if all other literals are false
                            vs2 = cvars[ci2]
                            ss2 = csigns[ci2]
                            all_other_false = True
                            for k in range(3):
                                if vs2[k] != v:
                                    if assignment[vs2[k]] == ss2[k]:
                                        all_other_false = False
                                        break
                            if all_other_false:
                                delta -= 1  # this clause will become sat
                    if delta < best_delta:
                        best_delta = delta
                        best_v = v
                v = best_v

            flip_var(v, assignment, unsat, false_lits)
            # Recompute cur_unsat (could optimize with delta tracking)
            cur_unsat = count_unsat(unsat)

        # If we're close to solution, try more steps with higher noise
        if cur_unsat < m * 0.3:
            for step2 in range(5000):
                if time.time() > deadline:
                    break
                if cur_unsat == 0:
                    return assignment
                if cur_unsat < best_unsat_count:
                    best_unsat_count = cur_unsat
                    best_assignment = assignment[:]

                while True:
                    ci = rng.randrange(m)
                    if unsat[ci] > 0:
                        break

                # Higher noise for escape
                if rng.random() < 0.3:
                    j = rng.randrange(3)
                    v = cvars[ci][j]
                else:
                    best_delta = 999
                    best_v = cvars[ci][0]
                    for j in range(3):
                        v = cvars[ci][j]
                        delta = 0
                        for ci2 in var_clauses[v]:
                            if unsat[ci2] == 0:
                                vs2 = cvars[ci2]
                                ss2 = csigns[ci2]
                                other_true = False
                                for k in range(3):
                                    if vs2[k] != v:
                                        if assignment[vs2[k]] == ss2[k]:
                                            other_true = True
                                            break
                                if not other_true:
                                    delta += 1
                            else:
                                vs2 = cvars[ci2]
                                ss2 = csigns[ci2]
                                all_other_false = True
                                for k in range(3):
                                    if vs2[k] != v:
                                        if assignment[vs2[k]] == ss2[k]:
                                            all_other_false = False
                                            break
                                if all_other_false:
                                    delta -= 1
                        if delta < best_delta:
                            best_delta = delta
                            best_v = v
                    v = best_v

                flip_var(v, assignment, unsat, false_lits)
                cur_unsat = count_unsat(unsat)

    if best_assignment is not None:
        return best_assignment
    return assignment
# EVOLVE-BLOCK-END
