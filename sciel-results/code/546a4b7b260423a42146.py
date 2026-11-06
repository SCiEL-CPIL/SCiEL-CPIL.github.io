"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [False] * n

    start_time = time.time()
    deadline = start_time + 310.0

    # Pre-process: build per-variable clause lists and per-clause literal data
    var_to_clauses = [[] for _ in range(n)]
    clause_vars = []
    clause_pol = []  # True if positive literal

    for i, c in enumerate(clauses):
        cv = []
        cp = []
        for lit in c:
            v = abs(lit) - 1
            p = lit > 0
            cv.append(v)
            cp.append(p)
            var_to_clauses[v].append(i)
        clause_vars.append(cv)
        clause_pol.append(cp)

    # State
    assignment = [False] * n
    sat_count = [0] * m  # number of satisfying literals per clause
    unsat = []  # list of unsatisfied clause indices

    def init_state(asm):
        unsat.clear()
        for i in range(m):
            cnt = 0
            cv = clause_vars[i]
            cp = clause_pol[i]
            for j in range(len(cv)):
                if asm[cv[j]] == cp[j]:
                    cnt += 1
            sat_count[i] = cnt
            if cnt == 0:
                unsat.append(i)

    def flip(v):
        assignment[v] = not assignment[v]
        for ci in var_to_clauses[v]:
            cv = clause_vars[ci]
            cp = clause_pol[ci]
            # Recompute sat count for this clause
            cnt = 0
            for j in range(len(cv)):
                if assignment[cv[j]] == cp[j]:
                    cnt += 1
            was_sat = sat_count[ci] > 0
            sat_count[ci] = cnt
            is_sat = cnt > 0
            if was_sat and not is_sat:
                unsat.append(ci)
            elif not was_sat and is_sat:
                try:
                    unsat.remove(ci)
                except ValueError:
                    pass

    def breakage(v):
        """Number of currently-sat clauses that become unsat if v flips."""
        b = 0
        for ci in var_to_clauses[v]:
            if sat_count[ci] > 0:
                # Check if v is the only satisfying literal
                cv = clause_vars[ci]
                cp = clause_pol[ci]
                v_satisfies = False
                for j in range(len(cv)):
                    if cv[j] == v:
                        if assignment[v] == cp[j]:
                            v_satisfies = True
                        break
                if v_satisfies and sat_count[ci] == 1:
                    b += 1
        return b

    def gain(v):
        """Number of currently-unsat clauses that become sat if v flips."""
        g = 0
        for ci in var_to_clauses[v]:
            if sat_count[ci] == 0:
                cv = clause_vars[ci]
                cp = clause_pol[ci]
                for j in range(len(cv)):
                    if cv[j] == v:
                        if (not assignment[v]) == cp[j]:
                            g += 1
                        break
        return g

    rng = random.Random(42)
    best_asm = None
    best_unsat = m + 1

    # Pre-compute variable positions within each clause for faster lookup
    var_pos_in_clause = [[] for _ in range(n)]  # var_pos_in_clause[v] = list of (clause_idx, pos_in_clause)
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            v = abs(lit) - 1
            var_pos_in_clause[v].append((i, j))

    def flip_fast(v):
        """Optimized flip using pre-computed positions."""
        assignment[v] = not assignment[v]
        for ci, pos in var_pos_in_clause[v]:
            cv = clause_vars[ci]
            cp = clause_pol[ci]
            # Recompute sat count for this clause
            cnt = 0
            for j in range(3):  # 3-SAT
                if assignment[cv[j]] == cp[j]:
                    cnt += 1
            was_sat = sat_count[ci] > 0
            sat_count[ci] = cnt
            is_sat = cnt > 0
            if was_sat and not is_sat:
                unsat.append(ci)
            elif not was_sat and is_sat:
                try:
                    unsat.remove(ci)
                except ValueError:
                    pass

    def breakage_fast(v):
        """Number of currently-sat clauses that become unsat if v flips."""
        b = 0
        for ci, pos in var_pos_in_clause[v]:
            if sat_count[ci] > 0 and sat_count[ci] == 1:
                cv = clause_vars[ci]
                cp = clause_pol[ci]
                if assignment[v] == cp[pos]:
                    b += 1
        return b

    def gain_fast(v):
        """Number of currently-unsat clauses that become sat if v flips."""
        g = 0
        for ci, pos in var_pos_in_clause[v]:
            if sat_count[ci] == 0:
                cv = clause_vars[ci]
                cp = clause_pol[ci]
                if (not assignment[v]) == cp[pos]:
                    g += 1
        return g

    def local_search(noise, max_steps, stuck_limit):
        nonlocal assignment, unsat
        cur = len(unsat)
        if cur == 0:
            return assignment[:], 0

        best_local = cur
        best_local_asm = assignment[:]
        stuck = 0
        steps = 0

        while steps < max_steps:
            if time.time() > deadline - 3.0:
                break
            if not unsat:
                return assignment[:], 0

            ci = rng.choice(unsat)
            cv = clause_vars[ci]

            if rng.random() < noise:
                v = rng.choice(cv)
            else:
                best_v = -1
                best_b = 10**9
                best_g = -10**9
                for v in cv:
                    b = breakage_fast(v)
                    if b < best_b:
                        best_b = b
                        best_g = gain_fast(v)
                        best_v = v
                    elif b == best_b:
                        g = gain_fast(v)
                        if g > best_g:
                            best_g = g
                            best_v = v
                v = best_v

            flip_fast(v)
            steps += 1
            cur = len(unsat)

            if cur == 0:
                return assignment[:], 0

            if cur < best_local:
                best_local = cur
                best_local_asm = assignment[:]
                stuck = 0
            else:
                stuck += 1

            if stuck > stuck_limit:
                break

        return best_local_asm, best_local

    max_restarts = 5000
    max_steps = 30000
    noise = 0.3

    for r in range(max_restarts):
        if time.time() > deadline - 10.0:
            break

        assignment = [rng.random() < 0.5 for _ in range(n)]
        init_state(assignment)
        cur = len(unsat)

        if cur == 0:
            return assignment[:]

        if cur < best_unsat:
            best_unsat = cur
            best_asm = assignment[:]

        # Adaptive noise: increase noise when stuck
        current_noise = noise
        if best_unsat > m // 10:
            current_noise = min(0.5, noise + 0.1 * (r % 10))
        
        assignment, cur = local_search(current_noise, max_steps, 5000)

        if cur == 0:
            return assignment[:]

        if cur < best_unsat:
            best_unsat = cur
            best_asm = assignment[:]

        # Polish phase when close to solution
        if best_unsat <= max(5, m // 40) and time.time() < deadline - 15.0:
            assignment = best_asm[:]
            init_state(assignment)
            cur = len(unsat)
            if cur == 0:
                return assignment[:]
            assignment, cur = local_search(0.05, 50000, 10000)
            if cur == 0:
                return assignment[:]
            if cur < best_unsat:
                best_unsat = cur
                best_asm = assignment[:]

    if best_asm:
        return best_asm
    return assignment

# EVOLVE-BLOCK-END
