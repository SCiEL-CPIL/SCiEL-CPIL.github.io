"""Stronger 3-SAT solver: portfolio of WalkSAT variants + unit propagation."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def _walksat_numpy(n, clauses, rng, max_flips, deadline, init=None):
    """Vectorized WalkSAT. Returns (assignment, solved)."""
    # Precompute clause literal arrays (1-based signed)
    # clauses: list of lists of ints
    if init is None:
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=np.int8)
    else:
        assignment = np.array(init, dtype=np.int8)  # 0/1

    # Convert clauses to numpy arrays for speed
    clause_arrs = [np.array(c, dtype=np.int64) for c in clauses]

    # Precompute: for each clause, abs values and signs
    # We'll compute satisfaction on the fly per step

    for step in range(max_flips):
        if time.time() > deadline:
            return assignment, False

        # Check satisfaction: for each clause, count satisfied literals
        # A literal l is satisfied if assignment[abs(l)-1] == (l > 0)
        # i.e., (l > 0) and assignment[abs(l)-1]==1, or (l<0) and assignment[abs(l)-1]==0
        # Which is: assignment[abs(l)-1] == (1 if l>0 else 0)

        all_sat = True
        unsat_clauses = []
        for i, c in enumerate(clause_arrs):
            idx = np.abs(c) - 1
            vals = assignment[idx]
            targets = (c > 0).astype(np.int8)
            sat_mask = vals == targets
            if not sat_mask.any():
                all_sat = False
                unsat_clauses.append(i)
            elif not sat_mask.all():
                # partially satisfied, fine
                pass

        if all_sat:
            return assignment, True

        # Pick a random unsatisfied clause
        ci = unsat_clauses[rng.randrange(len(unsat_clauses))]
        c = clause_arrs[ci]

        # Flip: with prob p, flip a literal in the clause; else flip random var
        if rng.random() < 0.5:
            li = rng.randrange(len(c))
            l = c[li]
        else:
            l = c[rng.randrange(len(c))]
            # Sometimes flip random variable
            if rng.random() < 0.3:
                l = (rng.randrange(n) + 1) * (1 if rng.random() < 0.5 else -1)

        idx = abs(l) - 1
        assignment[idx] = 1 - assignment[idx]

    return assignment, False


def _walksat_with_unit_prop(n, clauses, rng, max_flips, deadline, init=None):
    """WalkSAT enhanced with unit propagation on the fly."""
    if init is None:
        assignment = [rng.random() < 0.5 for _ in range(n)]
    else:
        assignment = list(init)

    # Work on a copy of clauses; unit propagation removes satisfied literals
    # and removes clauses with a unit literal by setting that variable
    # We'll maintain active clauses

    active_clauses = [list(c) for c in clauses]

    for step in range(max_flips):
        if time.time() > deadline:
            # Check if current assignment satisfies original clauses
            for c in clauses:
                sat = any(assignment[abs(l)-1] == (l > 0) for l in c)
                if not sat:
                    return assignment, False
            return assignment, True

        # Find unsatisfied clauses and apply unit propagation
        unsat_indices = []
        for i, c in enumerate(active_clauses):
            # Check if satisfied
            sat = False
            unit = None
            unsat_count = 0
            for l in c:
                if assignment[abs(l)-1] == (l > 0):
                    sat = True
                    break
                else:
                    unsat_count += 1
            if not sat:
                unsat_indices.append(i)
                # Check for unit
                if unsat_count == 1:
                    unit = c[0]

        if not unsat_indices:
            return assignment, True

        # Apply unit propagation: if any unsat clause is unit, flip it
        unit_applied = False
        for i in unsat_indices:
            c = active_clauses[i]
            unsat_count = sum(1 for l in c if assignment[abs(l)-1] != (l > 0))
            if unsat_count == 1:
                # Find the unsatisfied literal
                for l in c:
                    if assignment[abs(l)-1] != (l > 0):
                        assignment[abs(l)-1] = (l > 0)
                        unit_applied = True
                        break
                break

        if not unit_applied:
            # Pick random unsat clause, flip a literal
            ci = unsat_indices[rng.randrange(len(unsat_indices))]
            c = active_clauses[ci]
            # Flip a random literal in the clause
            l = c[rng.randrange(len(c))]
            assignment[abs(l)-1] = not assignment[abs(l)-1]

    # Final check
    for c in clauses:
        sat = any(assignment[abs(l)-1] == (l > 0) for l in c)
        if not sat:
            return assignment, False
    return assignment, True


def _local_search_min_conflicts(n, clauses, rng, max_flips, deadline):
    """Min-conflicts style: always flip a literal that reduces conflicts."""
    assignment = [rng.random() < 0.5 for _ in range(n)]

    # Precompute variable-clause incidence for conflict counting
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses):
        for l in c:
            var_to_clauses[abs(l)-1].append(ci)

    def count_conflicts():
        conflicts = 0
        for c in clauses:
            sat = False
            for l in c:
                if assignment[abs(l)-1] == (l > 0):
                    sat = True
                    break
            if not sat:
                conflicts += 1
        return conflicts

    for step in range(max_flips):
        if time.time() > deadline:
            return assignment, False

        cur_conflicts = count_conflicts()
        if cur_conflicts == 0:
            return assignment, True

        # Try flipping each variable in a random order, pick best
        # To keep it fast, sample a subset of variables
        num_samples = min(n, 50)
        candidates = rng.sample(range(n), num_samples)
        best_var = -1
        best_conflicts = cur_conflicts

        for v in candidates:
            old_val = assignment[v]
            assignment[v] = not old_val
            new_conflicts = count_conflicts()
            if new_conflicts < best_conflicts:
                best_conflicts = new_conflicts
                best_var = v
            assignment[v] = old_val

        if best_var >= 0:
            assignment[best_var] = not assignment[best_var]
        else:
            # No improvement, random flip
            v = rng.randrange(n)
            assignment[v] = not assignment[v]

    return assignment, False


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, 20s safety margin

    # Strategy 1: Quick WalkSAT with fixed seed
    rng = random.Random(42)
    assignment, solved = _walksat_numpy(n, clauses, rng, max_flips=50000,
                                         deadline=min(deadline, start_time + 50.0))
    if solved:
        return list(assignment)

    # Strategy 2: WalkSAT with unit propagation, different seed
    if time.time() < deadline - 20.0:
        rng2 = random.Random(123)
        assignment2, solved2 = _walksat_with_unit_prop(n, clauses, rng2,
                                                        max_flips=100000,
                                                        deadline=min(deadline, start_time + 150.0))
        if solved2:
            return assignment2

    # Strategy 3: Min-conflicts local search
    if time.time() < deadline - 10.0:
        rng3 = random.Random(777)
        assignment3, solved3 = _local_search_min_conflicts(n, clauses, rng3,
                                                            max_flips=50000,
                                                            deadline=min(deadline, start_time + 250.0))
        if solved3:
            return assignment3

    # Strategy 4: More WalkSAT restarts with different seeds
    if time.time() < deadline - 5.0:
        for seed in [1, 2, 3, 4, 5]:
            if time.time() > deadline - 5.0:
                break
            rng4 = random.Random(seed * 1000 + 42)
            a, s = _walksat_numpy(n, clauses, rng4, max_flips=30000,
                                   deadline=deadline - 5.0)
            if s:
                return list(a)

    # Return best effort: check which of our assignments is most satisfied
    candidates = []
    if solved:
        candidates.append(list(assignment))
    if solved2:
        candidates.append(assignment2)
    if solved3:
        candidates.append(assignment3)

    # Pick the one with most satisfied clauses
    best = None
    best_score = -1
    for cand in candidates:
        score = sum(1 for c in clauses if any(cand[abs(l)-1] == (l > 0) for l in c))
        if score > best_score:
            best_score = score
            best = cand

    if best is not None:
        return best

    # Fallback: random assignment
    rng_final = random.Random(0)
    return [rng_final.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
