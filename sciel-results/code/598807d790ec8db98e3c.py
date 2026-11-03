"""Difference triangle set (7,5): minimize the largest entry (scope).
Primary solver: CP-SAT (minimize scope, symmetric-breaking, time-bounded).
Fallback: fast greedy. Both always return a valid construction.
"""
import time

_T0 = time.time()
_DEADLINE = _T0 + 140.0  # hard safety margin before the 150 s kill


def _build_greedy(n_rows, k):
    """Fast greedy: each row takes the smallest next mark whose new
    differences are all unused. Deterministic, always valid."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = set()
            for a in row:
                d = x - a
                if d in used:
                    new = None
                    break
                new.add(d)
            if new is not None:
                row.append(x)
                used.update(new)
        rows.append(row)
    return rows


def _solve_cpsat(n_rows, k):
    """CP-SAT: minimize the largest entry subject to all within-row
    differences being distinct across all rows. Returns a row list or None."""
    try:
        from ortools.sat.python import cp_model
    except Exception:
        return None

    n = n_rows
    m = k + 1  # entries per row (index 0 is 0)
    total = n * m
    model = cp_model.CpModel()

    # entry variables
    X = [[model.NewIntVar(0, 300, "x_%d_%d" % (i, j)) for j in range(m)]
         for i in range(n)]
    for i in range(n):
        model.Add(X[i][0] == 0)
        for j in range(1, m):
            model.Add(X[i][j] > X[i][j - 1])

    # scope
    scope = model.NewIntVar(0, 300, "scope")
    for i in range(n):
        model.Add(X[i][m - 1] <= scope)

    # differences -> boolean "is this difference present"
    B = {}
    for i in range(n):
        for a in range(m):
            for b in range(a + 1, m):
                B[(i, a, b)] = model.NewBoolVar("b_%d_%d_%d" % (i, a, b))
                model.Add(B[(i, a, b)] == 1).OnlyEnforceIf(
                    model.NewBoolVar("t_%d_%d_%d" % (i, a, b)).Not()
                    .Then(model.NewBoolVar("t_%d_%d_%d" % (i, a, b)).Not()))
                # difference value
                dvar = model.NewIntVar(1, 300, "d_%d_%d_%d" % (i, a, b))
                model.Add(dvar == X[i][b] - X[i][a])
                model.Add(dvar == 1).OnlyEnforceIf(B[(i, a, b)])
                model.Add(dvar == 0).OnlyEnforceIf(B[(i, a, b)].Not())
                B[(i, a, b)] = (B[(i, a, b)], dvar)

    # per-difference at-most-one (d from 1..300)
    for d in range(1, 301):
        present = []
        for key, (bv, dvar) in B.items():
            present.append(bv)
            # enforce bv == 1 only if dvar == d, else 0
            model.Add(bv == 1).OnlyEnforceIf(
                model.NewBoolVar("eq_%d_%d" % (d, hash(key) % 100000)).Not())
        # (handled below more simply)

    # The above got tangled; build a clean version:
    return None  # placeholder to avoid shipping broken model


def _solve_cpsat_clean(n_rows, k):
    """Clean CP-SAT construction."""
    try:
        from ortools.sat.python import cp_model
    except Exception:
        return None

    n = n_rows
    m = k + 1
    model = cp_model.CpModel()

    X = [[model.NewIntVar(0, 300, "x_%d_%d" % (i, j)) for j in range(m)]
         for i in range(n)]
    for i in range(n):
        model.Add(X[i][0] == 0)
        for j in range(1, m):
            model.Add(X[i][j] > X[i][j - 1])

    scope = model.NewIntVar(0, 300, "scope")
    for i in range(n):
        model.Add(X[i][m - 1] <= scope)

    # For each pair (i,a,b) create a difference var and a bool that it
    # equals the difference value; then per difference value at-most-one.
    diff_vars = []
    for i in range(n):
        for a in range(m):
            for b in range(a + 1, m):
                dv = model.NewIntVar(1, 300, "dv_%d_%d_%d" % (i, a, b))
                model.Add(dv == X[i][b] - X[i][a])
                diff_vars.append(dv)

    # at-most-one per value 1..300 using AddAtMostOne over indicator bools
    for val in range(1, 301):
        inds = []
        for dv in diff_vars:
            ind = model.NewBoolVar("ind_%d_%d" % (val, id(dv) % 1000000))
            model.Add(dv == val).OnlyEnforceIf(ind)
            model.Add(dv != val).OnlyEnforceIf(ind.Not())
            inds.append(ind)
        model.AddAtMostOne(inds)

    model.Minimize(scope)

    # symmetric breaking: row 0 first, row1 > row0, ...
    for i in range(1, n):
        model.Add(X[i][m - 1] >= X[i - 1][m - 1])

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(1.0, _DEADLINE - time.time())
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    return [[int(solver.Value(X[i][j])) for j in range(m)] for i in range(n)]


def construct(n_rows, k):
    # Try CP-SAT (minimizes scope). Fallback to greedy on any failure.
    result = None
    if time.time() < _DEADLINE - 5.0:
        result = _solve_cpsat_clean(n_rows, k)
    if result is None:
        result = _build_greedy(n_rows, k)
    return result
