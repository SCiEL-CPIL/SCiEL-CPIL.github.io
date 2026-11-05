"""Optimized 3-SAT solver with incremental tracking and min-confl WalkSAT."""
# EVOLVE-BLOCK-START
import random
import time
from collections import defaultdict


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    deadline = time.time() + 290.0  # safety margin below 330s kill
    num_clauses = len(clauses)

    # Build variable-to-clauses index for fast incremental updates
    var_to_clauses = defaultdict(list)
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_to_clauses[abs(lit) - 1].append(ci)

    def is_clause_satisfied(ci, assignment):
        for lit in clauses[ci]:
            if assignment[abs(lit) - 1] == (lit > 0):
                return True
        return False

    def compute_unsat_set(assignment):
        unsat = set()
        for ci in range(num_clauses):
            if not is_clause_satisfied(ci, assignment):
                unsat.add(ci)
        return unsat

    def update_unsat_after_flip(var_idx, old_val, new_val, unsat_set, assignment):
        """Update unsat set after flipping variable var_idx from old_val to new_val."""
        for ci in var_to_clauses[var_idx]:
            was_unsat = ci in unsat_set
            # Check if clause is now satisfied
            now_sat = False
            for lit in clauses[ci]:
                if assignment[abs(lit) - 1] == (lit > 0):
                    now_sat = True
                    break
            if now_sat:
                if was_unsat:
                    unsat_set.discard(ci)
            else:
                if not was_unsat:
                    unsat_set.add(ci)

    def make_random_assignment(seed):
        rng = random.Random(seed)
        return [rng.random() < 0.5 for _ in range(n)]

    def walksat_step(assignment, unsat_set, rng, p_walk):
        """Perform one WalkSAT step with min-confl heuristic.
        Returns (new_unsat_count, flipped_var, old_val) or None if no unsat clauses."""
        if not unsat_set:
            return 0, -1, None

        # Pick a random unsatisfied clause
        ci = rng.choice(list(unsat_set))
        clause = clauses[ci]

        # Min-confl: for each literal in the clause, count how many OTHER unsat clauses it appears in
        # We want to flip the literal that is in the fewest other unsat clauses
        best_var = None
        best_score = float('inf')
        best_lit_val = None

        for lit in clause:
            var_idx = abs(lit) - 1
            # Count how many unsat clauses (other than ci) contain this variable
            # and would be broken if we flip (i.e., the literal is currently satisfied)
            conflict = 0
            for other_ci in var_to_clauses[var_idx]:
                if other_ci == ci:
                    continue
                if other_ci in unsat_set:
                    # This clause is already unsat; flipping might help or hurt
                    # Check if this variable's literal in other_ci is currently satisfied
                    for olit in clauses[other_ci]:
                        if abs(olit) - 1 == var_idx:
                            if assignment[var_idx] == (olit > 0):
                                # This literal is satisfied; flipping would break this clause
                                # But it's already unsat, so no additional harm
                                pass
                            else:
                                # This literal is unsatisfied; flipping might satisfy
                                # This clause is already unsat, so no change to unsat count
                                pass
                            break
                else:
                    # This clause is satisfied; check if flipping would break it
                    for olit in clauses[other_ci]:
                        if abs(olit) - 1 == var_idx:
                            if assignment[var_idx] == (olit > 0):
                                # This literal is currently satisfied; flipping would make it unsatisfied
                                # If no other literal in this clause is satisfied, it becomes unsat
                                # Quick check: is any other literal in other_ci satisfied?
                                other_sat = False
                                for ll in clauses[other_ci]:
                                    if ll != olit and assignment[abs(ll) - 1] == (ll > 0):
                                        other_sat = True
                                        break
                                if not other_sat:
                                    conflict += 1
                            break

            # Also consider: flipping satisfies ci, so net effect = -1 + conflict
            score = conflict
            if score < best_score:
                best_score = score
                best_var = var_idx
                best_lit_val = (lit > 0)

        if best_var is None:
            return len(unsat_set), -1, None

        # WalkSAT: with probability p_walk, pick random literal from clause
        if rng.random() < p_walk:
            lit = rng.choice(clause)
            best_var = abs(lit) - 1
            best_lit_val = (lit > 0)

        old_val = assignment[best_var]
        new_val = not old_val
        assignment[best_var] = new_val
        update_unsat_after_flip(best_var, old_val, new_val, unsat_set, assignment)

        return len(unsat_set), best_var, old_val

    best_assignment = None
    best_unsat = num_clauses + 1

    # Try multiple random restarts with WalkSAT + min-confl
    for attempt in range(5000):
        if time.time() > deadline:
            break

        assignment = make_random_assignment(attempt * 7919 + 13)
        unsat_set = compute_unsat_set(assignment)
        cur_unsat = len(unsat_set)

        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
        if cur_unsat == 0:
            return assignment

        # Local search: WalkSAT with dynamic weights
        rng = random.Random(attempt * 31337 + 42)
        steps = 0
        max_steps = 10000

        # Adaptive walk probability: start low, increase if stuck
        p_walk = 0.05
        stuck_counter = 0

        while cur_unsat > 0 and steps < max_steps:
            if time.time() > deadline:
                break
            steps += 1

            new_unsat, flipped_var, old_val = walksat_step(assignment, unsat_set, rng, p_walk)
            cur_unsat = new_unsat

            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]
            if cur_unsat == 0:
                return assignment

            # Track if we're stuck (no improvement in recent steps)
            # Simple heuristic: if unsat count hasn't decreased in 100 steps, increase p_walk
            if steps % 100 == 0:
                if cur_unsat >= best_unsat:
                    stuck_counter += 1
                    if stuck_counter > 5:
                        p_walk = min(0.5, p_walk * 1.5)
                        stuck_counter = 0
                else:
                    stuck_counter = 0
                    p_walk = max(0.02, p_walk * 0.8)

    return best_assignment if best_assignment else make_random_assignment(0)
# EVOLVE-BLOCK-END
