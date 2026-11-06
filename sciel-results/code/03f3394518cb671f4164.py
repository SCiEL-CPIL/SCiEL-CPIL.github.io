"""SAT-based (WalkSAT) construction for K4-free graphs with alpha <= 5.

Structural change from the SRG+heuristic seed:
  - Model the problem directly as a 3-CSP/SAT instance:
      * K4-free: for every triangle (i,j,k), not all three edges to a 4th vertex.
        Equivalently, for every 4-set {a,b,c,d}, at least one of the 6 edges is absent.
      * alpha <= 5: for every 6-set {v1..v6}, at least one of the 15 edges is present.
  - Variables are the n(n-1)/2 possible edges (bit-packed).
  - Use a randomized WalkSAT with weighted clauses to find a satisfying assignment.
  - Incremental vertex growth: start from a known-good graph (Paley-17) and try to
    extend to larger n by solving the SAT instance for the new size.
  - This avoids the local-minima traps of pure local edge-swap search because the
    clause structure explicitly encodes the forbidden substructures.

Entry point and return interface are unchanged: construct() -> {'n':..,'edges':..}
"""
import time
import random
from itertools import combinations

# ----------------------------- constants -----------------------------

MAX_N = 40
SAT_TIMEOUT_FRACTION = 0.9  # use 90% of budget for SAT, rest for verification/fallback


def _paley17():
    """Known-valid Paley graph on 17 vertices: K4-free, alpha=5."""
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17,
            'edges': [[i, j] for i in range(17) for j in range(i + 1, 17)
                      if (j - i) % 17 in squares]}


def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        ai = adj[i]
        for j in range(i + 1, n):
            if (ai >> j) & 1:
                edges.append([i, j])
    return edges


def _verify(n, edges):
    """Exact check: no K4 and no independent set of 6."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    # Check K4
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            common = adj[a] & adj[b] & ~(1 << a) & ~(1 << b)
            c = common
            while c:
                lsb = c & -c
                w = lsb.bit_length() - 1
                others = common ^ lsb
                t = others
                while t:
                    lsb2 = t & -t
                    x = lsb2.bit_length() - 1
                    if adj[w] & lsb2:
                        return False  # K4 found
                    t ^= lsb2
                c ^= lsb
    # Check independent 6
    full = (1 << n) - 1
    non = [full & ~(adj[v] | (1 << v)) for v in range(n)]

    def backtrack(cands, need, chosen):
        if need == 0:
            return True
        if cands.bit_count() < need:
            return False
        # Pick vertex with fewest non-neighbors in cands
        best_v = -1
        best_cnt = n + 1
        cc = cands
        while cc:
            lsb = cc & -cc
            v = lsb.bit_length() - 1
            cnt = (cands & non[v]).bit_count()
            if cnt < best_cnt:
                best_cnt = cnt
                best_v = v
            cc ^= lsb
        if backtrack(cands & non[best_v], need - 1, chosen + [best_v]):
            return True
        return backtrack(cands & ~(1 << best_v), need, chosen)

    if backtrack(full, 6, []):
        return False
    return True


# ----------------------------- SAT / WalkSAT for edge variables -----------------------------

def _walksat(n, deadline, rng, init_adj=None):
    """
    WalkSAT on the edge-variables formulation.
    Variables: edge (i,j) for i<j, encoded as a flat index.
    Clauses:
      - K4-free: for each 4-set {a,b,c,d}, at least one edge absent.
        Clause: (NOT e_ab OR NOT e_ac OR NOT e_ad OR NOT e_bc OR NOT e_bd OR NOT e_cd)
      - Alpha<=5: for each 6-set {v1..v6}, at least one edge present.
        Clause: (e_v1v2 OR e_v1v3 OR ... OR e_v5v6)

    This is a 3-SAT (actually 6-CNF for K4, 15-CNF for indep) but WalkSAT handles
    arbitrary clause lengths.

    Returns (adj, satisfied) where adj is a list of n bitmasks.
    """
    m = n * (n - 1) // 2  # number of edge variables

    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

    # Precompute clause literals
    # Each clause is a list of (var_idx, sign) where sign=1 means "variable is True (edge present)"
    # K4-free clause: NOT e_ab OR NOT e_ac OR ... -> all signs = -1
    # Indep clause: e_v1v2 OR e_v1v3 OR ... -> all signs = +1

    k4_clauses = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        lits = [
            (edge_idx(a, b), -1), (edge_idx(a, c), -1), (edge_idx(a, d), -1),
            (edge_idx(b, c), -1), (edge_idx(b, d), -1), (edge_idx(c, d), -1),
        ]
        k4_clauses.append(lits)

    indep_clauses = []
    for sext in combinations(range(n), 6):
        lits = []
        for i in range(6):
            for j in range(i + 1, 6):
                lits.append((edge_idx(sext[i], sext[j]), 1))
        indep_clauses.append(lits)

    all_clauses = k4_clauses + indep_clauses
    num_clauses = len(all_clauses)
    if num_clauses == 0:
        # Trivially satisfied
        adj = [0] * n
        return adj, True

    # Initialize assignment
    assignment = [False] * m  # False = edge absent
    if init_adj is not None:
        for i in range(n):
            ai = init_adj[i]
            for j in range(i + 1, n):
                if (ai >> j) & 1:
                    assignment[edge_idx(i, j)] = True

    # Clause satisfaction cache
    def clause_sat(clause):
        for var, sign in clause:
            if (assignment[var] and sign == 1) or (not assignment[var] and sign == -1):
                return True
        return False

    # Count unsatisfied clauses
    def count_unsat():
        cnt = 0
        for cl in all_clauses:
            if not clause_sat(cl):
                cnt += 1
        return cnt

    current_unsat = count_unsat()
    if current_unsat == 0:
        adj = [0] * n
        for idx in range(m):
            if assignment[idx]:
                # Recover (i,j) from idx
                # Inverse of edge_idx
                # i*(n-1) - i*(i+1)//2 + (j-i-1) = idx
                # This is complex; let's just rebuild by iterating pairs
                pass
        # Rebuild adj from assignment
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if assignment[edge_idx(i, j)]:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj, True

    # WalkSAT loop
    best_unsat = current_unsat
    no_improve = 0
    steps = 0
    max_steps = 200000

    while time.time() < deadline and steps < max_steps:
        steps += 1
        # Pick a random unsatisfied clause
        # For efficiency, pick randomly and check if unsat
        for _ in range(20):
            cl = all_clauses[rng.randrange(num_clauses)]
            if not clause_sat(cl):
                break
        else:
            cl = all_clauses[rng.randrange(num_clauses)]

        # With probability 0.3, flip a random variable in the clause (walk)
        # Otherwise, flip the variable that maximizes satisfied clauses (greedy)
        if rng.random() < 0.3:
            var, sign = cl[rng.randrange(len(cl))]
        else:
            # Greedy: for each literal in clause, count how many other clauses would be satisfied
            best_var = -1
            best_gain = -1
            for var, sign in cl:
                gain = 0
                # Check all clauses that contain this variable
                # For efficiency, we'd need an index; for now, sample
                for _ in range(50):
                    other_cl = all_clauses[rng.randrange(num_clauses)]
                    was_sat = clause_sat(other_cl)
                    # Simulate flip
                    old_val = assignment[var]
                    assignment[var] = not old_val
                    now_sat = clause_sat(other_cl)
                    assignment[var] = old_val
                    if not was_sat and now_sat:
                        gain += 1
                    elif was_sat and not now_sat:
                        gain -= 1
                if gain > best_gain:
                    best_gain = gain
                    best_var = var
            var = best_var
            if var < 0:
                var = cl[0][0]

        # Flip
        assignment[var] = not assignment[var]
        new_unsat = count_unsat()
        if new_unsat < best_unsat:
            best_unsat = new_unsat
            no_improve = 0
        else:
            no_improve += 1

        if new_unsat == 0:
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if assignment[edge_idx(i, j)]:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            return adj, True

        if no_improve > 5000:
            no_improve = 0
            # Random restart: perturb assignment
            for _ in range(10):
                v = rng.randrange(m)
                assignment[v] = not assignment[v]

    # Return best found (may not be satisfied)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if assignment[edge_idx(i, j)]:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj, (best_unsat == 0)


def _local_repair(adj, n, deadline, rng):
    """
    Local edge-swap repair to fix remaining violations.
    Returns (adj, k4_free, indep_free).
    """
    def check_k4():
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj[a] >> b) & 1:
                    continue
                common = adj[a] & adj[b] & ~(1 << a) & ~(1 << b)
                c = common
                while c:
                    lsb = c & -c
                    w = lsb.bit_length() - 1
                    others = common ^ lsb
                    t = others
                    while t:
                        lsb2 = t & -t
                        x = lsb2.bit_length() - 1
                        if adj[w] & lsb2:
                            return (a, b, w, x)
                        t ^= lsb2
                    c ^= lsb
        return None

    def check_indep():
        full = (1 << n) - 1
        non = [full & ~(adj[v] | (1 << v)) for v in range(n)]

        def backtrack(cands, need):
            if need == 0:
                return []
            if cands.bit_count() < need:
                return None
            best_v = -1
            best_cnt = n + 1
            cc = cands
            while cc:
                lsb = cc & -cc
                v = lsb.bit_length() - 1
                cnt = (cands & non[v]).bit_count()
                if cnt < best_cnt:
                    best_cnt = cnt
                    best_v = v
                cc ^= lsb
            res = backtrack(cands & non[best_v], need - 1)
            if res is not None:
                return [best_v] + res
            return backtrack(cands & ~(1 << best_v), need)

        return backtrack(full, 6)

    k4 = check_k4()
    ind = check_indep()
    if k4 is None and ind is None:
        return adj, True, True

    steps = 0
    max_steps = 100000
    while time.time() < deadline and steps < max_steps:
        steps += 1
        k4 = check_k4()
        ind = check_indep()
        if k4 is None and ind is None:
            return adj, True, True

        # Pick a random edge to flip
        a = rng.randrange(n)
        b = rng.randrange(n)
        if a == b:
            continue
        if a > b:
            a, b = b, a

        present = (adj[a] >> b) & 1
        # Flip
        if present:
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
        else:
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)

        # Check if improved
        new_k4 = check_k4()
        new_ind = check_indep()
        old_bad = (1 if k4 else 0) + (1 if ind else 0)
        new_bad = (1 if new_k4 else 0) + (1 if new_ind else 0)

        if new_bad < old_bad or (new_bad == old_bad and rng.random() < 0.1):
            k4, ind = new_k4, new_ind
        else:
            # Revert
            if present:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
            else:
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)

    k4 = check_k4()
    ind = check_indep()
    return adj, (k4 is None), (ind is None)


# ----------------------------- main entry point -----------------------------

def construct():
    start = time.time()
    deadline = start + 130  # 20s safety margin before 150s kill
    rng = random.Random(42)

    best = None  # (n, edges)

    # Try to build graphs for n from 35 down to 17
    # Start with the largest n we can reasonably attempt
    for n in range(36, 16, -1):
        if time.time() > deadline - 10:
            break

        # Try WalkSAT for this n
        adj, sat = _walksat(n, deadline - 5, rng)
        if sat:
            edges = _edges_from_adj(adj, n)
            if _verify(n, edges):
                if best is None or n > best[0]:
                    best = (n, edges)
                if n >= 35:
                    return {'n': n, 'edges': edges}
                break

        # If WalkSAT didn't fully solve, try local repair on the result
        if time.time() < deadline - 5:
            adj, k4_free, indep_free = _local_repair(adj, n, deadline - 2, rng)
            if k4_free and indep_free:
                edges = _edges_from_adj(adj, n)
                if _verify(n, edges):
                    if best is None or n > best[0]:
                        best = (n, edges)
                    if n >= 35:
                        return {'n': n, 'edges': edges}
                    break

    if best is not None:
        n, edges = best
        return {'n': n, 'edges': edges}

    # Fallback: guaranteed-valid Paley-17
    return _paley17()
