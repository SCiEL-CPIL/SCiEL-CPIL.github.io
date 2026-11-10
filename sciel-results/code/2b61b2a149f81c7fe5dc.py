"""Synthetic planted 3-SAT solver (150 variables)."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 290  # safety margin before 300s limit

    # Precompute variable occurrences: var_occ[v] = list of (clause_idx, sign)
    # where sign is +1 if literal is positive, -1 if negative.
    var_occ = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_occ[v].append((ci, sign))

    # Build a literal-level co-occurrence / correlation matrix for spectral init.
    # We use a lightweight proxy: for each variable, accumulate a "polarity bias"
    # from how literals co-occur. Positive co-occurrence bias -> literal likely true.
    # This is cheap (no full NxN eigen) and gives a better-than-random start.
    def spectral_init():
        # polarity[v] in [-1, 1]; >0 suggests variable True, <0 suggests False.
        # Use clause-internal literal pairing: in a planted solution, literals in
        # the same clause are correlated toward being simultaneously satisfiable.
        # We approximate by summing, per variable, the sign-weighted frequency of
        # its literal across clauses, then normalize.
        polarity = np.zeros(n, dtype=float)
        freq = np.zeros(n, dtype=float)
        for clause in clauses:
            for lit in clause:
                v = abs(lit) - 1
                s = 1.0 if lit > 0 else -1.0
                polarity[v] += s
                freq[v] += 1.0
        # Normalize polarity by frequency to get a per-variable bias in [-1,1].
        bias = np.where(freq > 0, polarity / np.maximum(freq, 1.0), 0.0)
        # Assign True where bias is positive (with a small threshold), else False.
        # Add a tiny deterministic tie-break so it is reproducible.
        assignment = [b > 0.0 for b in bias]
        return assignment

    def random_assignment(rng):
        return [rng.random() < 0.5 for _ in range(n)]

    def compute_clause_counts(assignment):
        counts = [0] * m
        for ci, clause in enumerate(clauses):
            c = 0
            for lit in clause:
                if assignment[abs(lit) - 1] == (lit > 0):
                    c += 1
            counts[ci] = c
        return counts

    def flip_variable(v, assignment, counts):
        new_val = not assignment[v]
        delta_unsat = 0
        for ci, sign in var_occ[v]:
            old_sat = (assignment[v] == (sign == 1))
            new_sat = (new_val == (sign == 1))
            if old_sat and not new_sat:
                counts[ci] -= 1
                if counts[ci] == 0:
                    delta_unsat += 1
            elif not old_sat and new_sat:
                counts[ci] += 1
                if counts[ci] == 1:
                    delta_unsat -= 1
        assignment[v] = new_val
        return delta_unsat

    def _break_keep(v, ci, assignment, counts):
        """Return (break_count, keep_count) if we flip variable v, where ci is the
        unsat clause we are trying to fix."""
        break_count = 0
        keep_count = 0
        for oci, osign in var_occ[v]:
            if oci == ci:
                # Flipping v satisfies the literal in ci (since ci is unsat and
                # contains v, the literal is currently false; flipping makes it true).
                keep_count += 1
                continue
            if counts[oci] == 1 and assignment[v] == (osign == 1):
                # This clause has exactly one satisfied literal and it is v's;
                # flipping v would break it.
                break_count += 1
        return break_count, keep_count

    def walksat(assignment, counts, rng, steps, soft_beta=1.0):
        """WalkSAT with a smooth (soft-min) break-count scoring heuristic.
        Instead of a hard argmin, we sample among the clause literals with a
        probability proportional to exp(-beta * break_count). This gives a
        smooth surrogate for the nonsmooth min objective and avoids the
        sharp local-minima traps of a pure argmin."""
        for _ in range(steps):
            if time.time() > deadline:
                return False
            unsat = [ci for ci in range(m) if counts[ci] == 0]
            if not unsat:
                return True
            ci = rng.choice(unsat)
            clause = clauses[ci]

            # Compute soft scores for each literal in the clause.
            scores = []
            for lit in clause:
                v = abs(lit) - 1
                b, k = _break_keep(v, ci, assignment, counts)
                # Lower break is better; add small keep bonus.
                scores.append(-soft_beta * b + 0.3 * k)
            # Soft-max sampling (log-sum-exp style).
            max_s = max(scores)
            exps = [math.exp(s - max_s) for s in scores]
            total = sum(exps)
            r = rng.random() * total
            acc = 0.0
            chosen = 0
            for idx, e in enumerate(exps):
                acc += e
                if r <= acc:
                    chosen = idx
                    break
            v = abs(clause[chosen]) - 1
            flip_variable(v, assignment, counts)

            # Occasional random kick to escape local minima.
            if rng.random() < 0.04:
                v = rng.randrange(n)
                flip_variable(v, assignment, counts)

        return all(counts[ci] > 0 for ci in range(m))

    def min_conflicts(assignment, counts, rng, steps):
        """Min-conflicts: pick the literal in a random unsat clause that minimizes
        the number of clauses it would newly break (hard argmin)."""
        for _ in range(steps):
            if time.time() > deadline:
                return False
            unsat = [ci for ci in range(m) if counts[ci] == 0]
            if not unsat:
                return True
            ci = rng.choice(unsat)
            clause = clauses[ci]
            best_var = -1
            best_conflict = float('inf')
            for lit in clause:
                v = abs(lit) - 1
                b, _ = _break_keep(v, ci, assignment, counts)
                if b < best_conflict:
                    best_conflict = b
                    best_var = v
            if best_var >= 0:
                flip_variable(best_var, assignment, counts)
        return all(counts[ci] > 0 for ci in range(m))

    # Main search loop with multiple restarts and alternating strategies.
    best_assignment = None
    best_unsat = m + 1
    seed = 0
    restart_count = 0
    steps_per_restart = 5000

    while time.time() < deadline:
        rng = random.Random(seed)

        # Alternate initialization: spectral (structured) vs random.
        # Spectral init is reused as the structured prior; random for diversity.
        if restart_count % 2 == 0:
            assignment = spectral_init()
        else:
            assignment = random_assignment(rng)

        counts = compute_clause_counts(assignment)
        cur_unsat = sum(1 for c in counts if c == 0)

        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]

        if cur_unsat == 0:
            return assignment

        # Choose a strategy. Soft-min WalkSAT is primary; min-conflicts as backup.
        strategy = restart_count % 2
        if strategy == 0:
            solved = walksat(assignment, counts, rng, steps_per_restart,
                             soft_beta=1.0)
        else:
            solved = min_conflicts(assignment, counts, rng, steps_per_restart)

        if solved:
            return assignment

        cur_unsat = sum(1 for c in counts if c == 0)
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]

        restart_count += 1
        seed += 1

        # Gradually deepen search.
        if restart_count % 10 == 0:
            steps_per_restart = min(steps_per_restart * 2, 50000)

    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
