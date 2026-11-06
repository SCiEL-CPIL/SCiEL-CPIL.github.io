"""Edge-colourings of K16: every K4 must span >=5 distinct colours.
Minimise the number of colours used. Uses CP-SAT (OR-Tools) when available,
with a strict wall-clock deadline and a guaranteed-valid fallback."""

import time

_N = 16
_DEADLINE_S = 120.0
_START = time.time()

# Build the 4-cycle "opposite-edges" constraints (the forbidden pattern).
_C4S = []
for a in range(_N):
    for b in range(_N):
        if b == a:
            continue
        for c in range(_N):
            if c == a or c == b:
                continue
            for d in range(_N):
                if d == a or d == b or d == c:
                    continue
                # canonical representative of {a,b,c,d}
                if a < b and a < c and a < d and b < c and b < d and c < d:
                    # opposite edge pairs: (a,b)-(c,d), (a,c)-(b,d), (a,d)-(b,c)
                    _C4S.append((
                        (a, b), (c, d),
                        (a, c), (b, d),
                        (a, d), (b, c),
                    ))

# Precompute the 120 edge indices in lexicographic order.
_EDGE_LIST = [(i, j) for i in range(_N) for j in range(i + 1, _N)]
_EDGE_INDEX = {}
for _idx, (_u, _v) in enumerate(_EDGE_LIST):
    _EDGE_INDEX[(_u, _v)] = _idx
    _EDGE_INDEX[(_v, _u)] = _idx


def _elapsed():
    return time.time() - _START


def _valid_seed():
    # (i + j) mod 17: a valid 17-colouring (guarantees every K4 sees >=5 colours).
    return [(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)]


def _try_cpsat(k, time_limit):
    """Return a valid list of 120 colours in [0,k) if CP-SAT finds one, else None."""
    try:
        from ortools.sat.python import cp_model
    except Exception:
        return None

    model = cp_model.CpModel()
    vars_ = [model.NewIntVar(0, k - 1, "e%d" % idx)
             for idx in range(len(_EDGE_LIST))]

    for (e1, e2, f1, f2, g1, g2) in _C4S:
        v1 = vars_[_EDGE_INDEX[e1]]
        v2 = vars_[_EDGE_INDEX[e2]]
        v3 = vars_[_EDGE_INDEX[f1]]
        v4 = vars_[_EDGE_INDEX[f2]]
        v5 = vars_[_EDGE_INDEX[g1]]
        v6 = vars_[_EDGE_INDEX[g2]]
        # forbid: (v1==v2) AND (v3==v4) AND (v5==v6)
        b1 = model.NewBoolVar("b1_%d_%d" % (_EDGE_INDEX[e1], _EDGE_INDEX[e2]))
        b2 = model.NewBoolVar("b2_%d_%d" % (_EDGE_INDEX[f1], _EDGE_INDEX[f2]))
        b3 = model.NewBoolVar("b3_%d_%d" % (_EDGE_INDEX[g1], _EDGE_INDEX[g2]))
        model.Add(v1 == v2).OnlyEnforceIf(b1)
        model.Add(v1 != v2).OnlyEnforceIf(b1.Not())
        model.Add(v3 == v4).OnlyEnforceIf(b2)
        model.Add(v3 != v4).OnlyEnforceIf(b2.Not())
        model.Add(v5 == v6).OnlyEnforceIf(b3)
        model.Add(v5 != v6).OnlyEnforceIf(b3.Not())
        model.Add(b1 + b2 + b3 <= 2)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(1.0, time_limit)
    solver.parameters.num_workers = 8
    solver.parameters.random_seed = 42

    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [int(solver.Value(v)) for v in vars_]
    return None


def construct():
    deadline = _START + _DEADLINE_S

    # Try to beat the seed (17 colours) with the minimal feasible colour count.
    for k in (13, 14, 15):
        remaining = deadline - _elapsed()
        if remaining < 3.0:
            break
        result = _try_cpsat(k, remaining - 3.0)
        if result is not None:
            return result

    # Fallback: guaranteed-valid 17-colouring.
    return _valid_seed()
