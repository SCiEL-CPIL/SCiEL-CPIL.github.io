"""Planted 3-SAT solver (50 vars) — robust WalkSAT with time guard, smart init, restarts, polish."""
import time
import random
import math

# EVOLVE-BLOCK-START


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 290.0  # safety margin under 300s per-instance budget

    # Precompute clause literal indices for speed
    clause_vars = [[abs(l) - 1 for l in c] for c in clauses]
    clause_signs = [[(1 if l > 0 else -1) for l in c] for c in clauses]

    # Build variable-to-clauses index for faster lookahead
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            var_to_clauses[clause_vars[i][j]].append(i)

    def unsat_count(assign):
        cnt = 0
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_signs[i]
            for j in range(3):
                if assign[cv[j]] == (cs[j] > 0):
                    cnt += 1
                    break
        return m - cnt

    def is_satisfied(assign):
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_signs[i]
            ok = False
            for j in range(3):
                if assign[cv[j]] == (cs[j] > 0):
                    ok = True
                    break
            if not ok:
                return False
        return True

    def get_unsat_clauses(assign):
        unsat = []
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_signs[i]
            ok = False
            for j in range(3):
                if assign[cv[j]] == (cs[j] > 0):
                    ok = True
                    break
            if not ok:
                unsat.append(i)
        return unsat

    def polish(assign, max_iters=200):
        """Deterministic sweep: flip any variable that strictly reduces unsat count."""
        for _ in range(max_iters):
            changed = False
            base = unsat_count(assign)
            if base == 0:
                break
            for v in range(n):
                assign[v] = not assign[v]
                c = unsat_count(assign)
                if c < base:
                    base = c
                    changed = True
                else:
                    assign[v] = not assign[v]
            if not changed:
                break
        return assign

    def unit_propagate(assign):
        """Placeholder for compatibility. Returns (assign, solved_bool, conflict_bool)."""
        solved = is_satisfied(assign)
        return assign, solved, False

    def walk_sat(assign, rng, flip_budget):
        """Run WalkSAT; return (assign, solved_bool)."""
        for _ in range(flip_budget):
            if time.time() > deadline - 5.0:
                return assign, False
            
            # Check if solved
            if is_satisfied(assign):
                return assign, True
            
            unsat_idx = get_unsat_clauses(assign)
            if not unsat_idx:
                return assign, True
            
            # Pick a random unsat clause
            ci = unsat_idx[rng.randrange(len(unsat_idx))]
            cv = clause_vars[ci]
            cs = clause_signs[ci]
            
            # For each literal in this clause, count how many other unsat clauses it would satisfy
            best_score = -1
            best_j = -1
            for j in range(3):
                v = cv[j]
                s = cs[j]
                new_val = (s > 0)
                score = 0
                for ui in unsat_idx:
                    if ui == ci:
                        continue
                    uc = clause_vars[ui]
                    us = clause_signs[ui]
                    for k in range(3):
                        if uc[k] == v:
                            if new_val == (us[k] > 0):
                                score += 1
                            break
                # Add small random noise
                score += rng.random() * 0.01
                if score > best_score:
                    best_score = score
                    best_j = j
            v = cv[best_j]
            assign[v] = (cs[best_j] > 0)
        return assign, is_satisfied(assign)

    # --- Initialization: try several seeds, keep best ---
    rng = random.Random(12345)
    best_assign = None
    best_unsat = m + 1

    # Generate candidate initial assignments
    candidates = []
    for _ in range(12):
        a = [rng.random() < 0.5 for _ in range(n)]
        candidates.append(a)
    candidates.append([True] * n)
    candidates.append([False] * n)
    candidates.append([i % 2 == 0 for i in range(n)])
    candidates.append([i % 3 == 0 for i in range(n)])

    for a in candidates:
        # Try unit propagation on each candidate
        a2 = list(a)
        a2, solved, conflict = unit_propagate(a2)
        if solved:
            return a2
        u = unsat_count(a2)
        if u < best_unsat:
            best_unsat = u
            best_assign = list(a2)

    if best_unsat == 0:
        return best_assign

    # --- Main search: WalkSAT with restarts ---
    restarts = 0
    max_restarts = 10000
    while time.time() < deadline - 10.0 and restarts < max_restarts:
        restarts += 1
        # Start from the best assignment found so far
        current = list(best_assign)
        
        # Perturb the current assignment to escape local minima
        num_perturb = 1 + rng.randrange(3)
        for _ in range(num_perturb):
            v = rng.randrange(n)
            current[v] = not current[v]
            
        flip_budget = 300 + rng.randrange(400)
        res, solved = walk_sat(current, rng, flip_budget)
        u = unsat_count(res)
        if u < best_unsat:
            best_unsat = u
            best_assign = list(res)
        if solved:
            return res
        # Note: We do not update current to res if it wasn't better.
        # The next iteration will restart from best_assign again.

    # Final polish on best found
    final = list(best_assign)
    polish(final)
    if is_satisfied(final):
        return final
    # Try a few more polish passes with random perturbations
    for _ in range(20):
        perturbed = list(final)
        for _ in range(rng.randrange(1, 4)):
            v = rng.randrange(n)
            perturbed[v] = not perturbed[v]
        polish(perturbed, max_iters=50)
        if is_satisfied(perturbed):
            return perturbed
        u = unsat_count(perturbed)
        if u < best_unsat:
            best_unsat = u
            final = list(perturbed)
    return final


# EVOLVE-BLOCK-END
