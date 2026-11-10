"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import time


def solve(instance):
    """SA-flavored WalkSAT with incremental clause tracking and escape moves."""
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before the 330s kill

    # Pre-extract clause literals as (var_idx, sign)
    clause_data = []
    for clause in clauses:
        clause_data.append([(abs(l) - 1, 1 if l > 0 else -1) for l in clause])

    num_clauses = len(clause_data)

    # Index: var_idx -> list of (clause_idx, sign_in_that_clause)
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clause_data):
        for var_idx, sign in clause:
            var_to_clauses[var_idx].append((ci, sign))

    def clause_is_satisfied(ci, assignment):
        for var_idx, sign in clause_data[ci]:
            if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                return True
        return False

    def build_unsat_set(assignment):
        unsat = set()
        for ci in range(num_clauses):
            if not clause_is_satisfied(ci, assignment):
                unsat.add(ci)
        return unsat

    def walk_one(assignment, rng, max_steps, T0):
        """Run SA-flavored WalkSAT with incremental unsat tracking.

        Accepts improving/neutral moves always; accepts worsening moves with
        probability exp(-delta/T) where T cools geometrically. This lets the
        search escape deep local minima that trap greedy min-conflicts.
        """
        unsat_clauses = build_unsat_set(assignment)
        cur_uc = len(unsat_clauses)

        if cur_uc == 0:
            return assignment, True

        # Track the best (fewest unsat) assignment seen during this run
        best_uc = cur_uc
        best_assignment = assignment[:]

        T = T0
        cool = 0.99999

        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, False

            if not unsat_clauses:
                return assignment, True

            # Weighted clause selection: prefer clauses that are "closer" to
            # being satisfied (fewer currently-true... i.e. more literals false
            # is normal; weight by number of literals that are currently true is 0
            # for unsat). Instead weight by how many of the clause's variables
            # appear in many unsat clauses (higher leverage).
            # Use a cheap proxy: weight = number of distinct variables in the
            # clause that are in at least one other unsat clause.
            # For speed, use a simpler weight: random among unsat, but bias
            # toward clauses with higher "activity" via precomputed occurrence.
            # We do a light weighted pick using clause length-based + random.
            uc_list = list(unsat_clauses)
            k = len(uc_list)
            if k == 1:
                ci = uc_list[0]
            else:
                # Weight by (number of variables in this clause that occur in
                # other unsat clauses) approximated by clause length + noise.
                weights = []
                for c in uc_list:
                    # cheap: length is constant 3; add small random weight
                    weights.append(1.0 + rng.random())
                total = sum(weights)
                r = rng.random() * total
                cum = 0.0
                chosen = 0
                for idx, w in enumerate(weights):
                    cum += w
                    if r <= cum:
                        chosen = idx
                        break
                ci = uc_list[chosen]

            clause = clause_data[ci]

            # Compute net delta in unsat count for flipping each variable
            scores = []
            for var_idx, sign in clause:
                gain = 0
                loss = 0
                for (uci, u_sign) in var_to_clauses[var_idx]:
                    if uci == ci:
                        gain += 1
                        continue
                    if uci in unsat_clauses:
                        gain += 1
                    else:
                        lit_true = (assignment[var_idx] and u_sign == 1) or (not assignment[var_idx] and u_sign == -1)
                        if lit_true:
                            other_true = False
                            for (v2, s2) in clause_data[uci]:
                                if v2 == var_idx:
                                    continue
                                if (assignment[v2] and s2 == 1) or (not assignment[v2] and s2 == -1):
                                    other_true = True
                                    break
                            if not other_true:
                                loss += 1
                scores.append(loss - gain)

            min_score = min(scores)
            candidates = [i for i, s in enumerate(scores) if s == min_score]
            chosen_idx = rng.choice(candidates)
            var_idx, sign = clause[chosen_idx]
            delta = scores[chosen_idx]

            # Simulated annealing acceptance
            if delta <= 0:
                accept = True
            else:
                # probability of accepting a worsening move
                p = math.exp(-delta / max(T, 1e-9))
                accept = (rng.random() < p)

            if accept:
                old_val = assignment[var_idx]
                assignment[var_idx] = not old_val

                # Incrementally update unsat_clauses
                for (uci, u_sign) in var_to_clauses[var_idx]:
                    was_unsat = uci in unsat_clauses
                    is_now_satisfied = clause_is_satisfied(uci, assignment)
                    if was_unsat and is_now_satisfied:
                        unsat_clauses.discard(uci)
                    elif (not was_unsat) and (not is_now_satisfied):
                        unsat_clauses.add(uci)

                cur_uc = len(unsat_clauses)
                if cur_uc < best_uc:
                    best_uc = cur_uc
                    best_assignment = assignment[:]

            # Cool temperature
            T *= cool

        return best_assignment, (best_uc == 0)

    rng = random.Random(0)

    # Multi-restart with SA. Early restarts use higher temperature (more
    # exploration); later restarts use lower temperature (more exploitation).
    max_steps_per_restart = 25000
    num_restarts = 300

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Temperature schedule: start high, decrease across restarts
        T0 = 2.0 * (0.95 ** restart) + 0.1

        assignment, solved = walk_one(assignment, rng, max_steps_per_restart, T0)

        if solved:
            return assignment

    # Fallback: all False
    return [False] * n

# EVOLVE-BLOCK-END
