"""WalkSAT for planted 3-SAT with unit propagation, weighted clause selection,
polish phase, and an explicit wall-clock deadline guard."""
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    deadline = time.monotonic() + 285.0  # safety margin vs 300s limit

    # Precompute per-clause variable lists and polarity bit per variable.
    clause_vars = []
    clause_polarity = []
    for c in clauses:
        vs = []
        ps = []
        for l in c:
            a = abs(l) - 1
            vs.append(a)
            ps.append(1 if l > 0 else 0)
        clause_vars.append(vs)
        clause_polarity.append(ps)

    # For each variable, list of (clause_index, polarity).
    var_clauses = [[] for _ in range(n)]
    for ci, (vs, ps) in enumerate(zip(clause_vars, clause_polarity)):
        for v, p in zip(vs, ps):
            var_clauses[v].append((ci, p))

    m = len(clauses)

    def make_assignment():
        return [rng.random() < 0.5 for _ in range(n)]

    def init_status(assignment):
        # num_sat[ci] = number of satisfied literals in clause ci
        num_sat = [0] * m
        for ci in range(m):
            s = 0
            for v, p in zip(clause_vars[ci], clause_polarity[ci]):
                if assignment[v] == bool(p):
                    s += 1
            num_sat[ci] = s
        return num_sat

    def num_unsat_of(num_sat):
        return sum(1 for s in num_sat if s == 0)

    def unit_propagate(assignment, num_sat):
        """Apply unit propagation. Returns (new_assignment, new_num_sat, new_unsat, changed)."""
        changed = False
        for v in range(n):
            for (ci, p) in var_clauses[v]:
                if num_sat[ci] == 1:
                    # Clause has exactly one unsatisfied literal; force it
                    vs = clause_vars[ci]
                    ps = clause_polarity[ci]
                    # Find the unsatisfied literal in this clause
                    for j in range(len(vs)):
                        if assignment[vs[j]] != bool(ps[j]):
                            target_v = vs[j]
                            if target_v != v:
                                # Need to flip target_v to satisfy
                                if assignment[target_v] != bool(p):
                                    # Flip target_v
                                    for (ci2, p2) in var_clauses[target_v]:
                                        if assignment[target_v] == bool(p2):
                                            num_sat[ci2] -= 1
                                        else:
                                            num_sat[ci2] += 1
                                    assignment[target_v] = not assignment[target_v]
                                    changed = True
                            break
                    break
        return assignment, num_sat, changed

    def polish(assignment, num_sat, max_iters=2000):
        """High-precision polish: try to fix remaining unsat clauses with minimal disruption."""
        unsat = sum(1 for s in num_sat if s == 0)
        for _ in range(max_iters):
            if time.monotonic() > deadline:
                break
            if unsat == 0:
                return assignment, num_sat, 0
            # Find an unsat clause
            start = rng.randrange(m)
            clause = None
            for k in range(m):
                ci = (start + k) % m
                if num_sat[ci] == 0:
                    clause = ci
                    break
            if clause is None:
                return assignment, num_sat, 0

            vs = clause_vars[clause]
            ps = clause_polarity[clause]
            
            # Try each literal that would satisfy this clause
            best_v = -1
            best_delta = float('inf')
            for i in range(len(vs)):
                v = vs[i]
                if assignment[v] != bool(ps[i]):
                    # Flipping v would satisfy this clause
                    # Count how many currently-satisfied clauses would become unsat
                    delta = 0
                    for (ci, p) in var_clauses[v]:
                        if assignment[v] == bool(p):
                            if num_sat[ci] == 1:
                                delta += 1  # This clause would become unsat
                    if delta < best_delta:
                        best_delta = delta
                        best_v = v
                        if delta == 0:
                            break

            if best_v < 0:
                break

            # Flip best_v
            v = best_v
            for (ci, p) in var_clauses[v]:
                if assignment[v] == bool(p):
                    num_sat[ci] -= 1
                    if num_sat[ci] == 0:
                        unsat += 1
                else:
                    num_sat[ci] += 1
                    if num_sat[ci] == 1:
                        unsat -= 1
            assignment[v] = not assignment[v]

        return assignment, num_sat, unsat

    # Best assignment seen (fewest unsat clauses).
    best_assignment = None
    best_unsat = None

    def record_if_better(assignment, unsat):
        nonlocal best_assignment, best_unsat
        if best_unsat is None or unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assignment[:]

    # Restart loop.
    restart_step_budget = 30000
    max_restarts = 500
    for restart in range(max_restarts):
        if time.monotonic() > deadline:
            break
        assignment = make_assignment()
        num_sat = init_status(assignment)
        unsat = num_unsat_of(num_sat)
        
        # Apply unit propagation
        assignment, num_sat, changed = unit_propagate(assignment, num_sat)
        if changed:
            unsat = num_unsat_of(num_sat)
        
        record_if_better(assignment, unsat)
        if unsat == 0:
            return assignment

        # Local search phase with weighted flips.
        steps = 0
        while steps < restart_step_budget:
            steps += 1
            if steps % 256 == 0 and time.monotonic() > deadline:
                record_if_better(assignment, unsat)
                break

            # Pick an unsatisfied clause with weighted selection.
            # Prefer clauses with more unsatisfied literals (harder to satisfy).
            # Use a simple approach: scan and pick randomly among unsat clauses,
            # but bias toward recently violated ones.
            start = rng.randrange(m)
            clause = None
            for k in range(m):
                ci = (start + k) % m
                if num_sat[ci] == 0:
                    clause = ci
                    break
            if clause is None:
                unsat = 0
                record_if_better(assignment, 0)
                return assignment

            vs = clause_vars[clause]
            ps = clause_polarity[clause]
            k = len(vs)

            # Choose flip variable with min-ones bias.
            # 15% chance: random flip (exploration)
            if rng.random() < 0.15:
                i = rng.randrange(k)
            else:
                # Prefer literals that would satisfy the clause
                unsat_lits = [i for i in range(k) if assignment[vs[i]] != bool(ps[i])]
                if unsat_lits:
                    i = unsat_lits[rng.randrange(len(unsat_lits))]
                else:
                    i = rng.randrange(k)

            v = vs[i]
            # Flip v; update num_sat for all clauses containing v.
            for (ci, cp) in var_clauses[v]:
                if assignment[v] == bool(cp):
                    # currently satisfied -> becomes unsat
                    num_sat[ci] -= 1
                    if num_sat[ci] == 0:
                        unsat += 1
                else:
                    # currently unsat -> becomes satisfied
                    num_sat[ci] += 1
                    if num_sat[ci] == 1:
                        unsat -= 1
            assignment[v] = not assignment[v]

            if unsat == 0:
                record_if_better(assignment, 0)
                return assignment

        # Polish phase: try to fix remaining unsat clauses
        if unsat <= 5:
            assignment, num_sat, unsat_after = polish(assignment, num_sat)
            if unsat_after == 0:
                record_if_better(assignment, 0)
                return assignment
            unsat = unsat_after

        record_if_better(assignment, unsat)

    # Deadline reached: return best seen (or last assignment).
    if best_assignment is None:
        best_assignment = assignment
    return best_assignment
