"""
Rewritten program: direct LP over full report space with proper IC constraints.
The key insight is that for a finite type space with 2 goods and values in {0,1,2,3},
we have 16 types. The LP has 32 allocation variables + 16 payment variables = 48 vars.
Constraints: monotonicity (implied by IC but we can add for speed), IR, and full IC.
Full IC: for every pair (t, t'), U(t,t) >= U(t,t'). That's 16*15 = 240 constraints.
Plus 16 IR constraints. Plus bounds.

The objective is to maximize expected revenue = sum_t p[t] * payment[t].
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 75.0  # 15s safety margin before 90s kill

    # --- Parse inputs ---
    p_vals = None
    try:
        if isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
            if len(probs) == 4:
                p_vals = np.outer(probs, probs)
            elif len(probs) == 16:
                p_vals = probs.reshape(4, 4)
        elif isinstance(distribution_spec, dict):
            p_vals = np.zeros((4, 4))
            for k, v in distribution_spec.items():
                if isinstance(k, tuple) and len(k) == 2:
                    p_vals[int(k[0]), int(k[1])] = float(v)
            # Check if it's a marginal dict
            if all(isinstance(k, int) for k in distribution_spec.keys()):
                marg = np.array([distribution_spec.get(i, 0) for i in range(4)])
                if marg.sum() > 0:
                    p_vals = np.outer(marg, marg)
    except Exception:
        pass

    if p_vals is None:
        p_vals = np.ones((4, 4)) / 16.0

    p_vals = np.asarray(p_vals, dtype=float)
    if p_vals.size == 4:
        p_vals = np.outer(p_vals, p_vals)
    p_vals = p_vals.reshape(4, 4)
    p_vals = p_vals / p_vals.sum()

    # Type space: 16 types, indexed by (i, j) where i,j in {0,1,2,3}
    all_types = [(i, j) for i in range(4) for j in range(4)]
    type_index = {t: i for i, t in enumerate(all_types)}

    # Determine effective support for IC/IR
    if support is None:
        effective_support = set(all_types)
    else:
        if isinstance(support, (list, tuple, set, np.ndarray)):
            eff = []
            for r in support:
                try:
                    eff.append((int(r[0]), int(r[1])))
                except Exception:
                    pass
            effective_support = set(eff) if eff else set(all_types)
        else:
            effective_support = set(all_types)

    # Ensure all types in effective_support are valid
    valid_support = set()
    for t in effective_support:
        if t in type_index:
            valid_support.add(t)
    if not valid_support:
        valid_support = set(all_types)

    n_types = 16
    n_vars = 48  # 32 alloc + 16 pay

    # Objective: maximize expected revenue = sum p[t] * pay[t]
    # In linprog, minimize c^T x, so c = -p[t] for payment vars
    c = np.zeros(n_vars)
    for i in range(16):
        t = all_types[i]
        c[32 + i] = -p_vals[t[0], t[1]]

    A_ub = []
    b_ub = []

    # 1. Individual Rationality (IR): pay[t] <= x[t] dot t  for all t in support
    #    pay[t] - sum_g x[t,g]*t[g] <= 0
    for t in valid_support:
        ti = type_index[t]
        row = np.zeros(n_vars)
        row[32 + ti] = 1.0
        row[ti * 2 + 0] = -float(t[0])
        row[ti * 2 + 1] = -float(t[1])
        A_ub.append(row)
        b_ub.append(0.0)

    # 2. Full Incentive Compatibility (IC):
    #    For all t, t' in support: U(t,t) >= U(t,t')
    #    U(t,t') = t[0]*x[t',0] + t[1]*x[t',1] - pay[t']
    #    U(t,t) >= U(t,t')  =>  t[0]*x[t,0]+t[1]*x[t,1]-pay[t] >= t[0]*x[t',0]+t[1]*x[t',1]-pay[t']
    #    =>  pay[t] - pay[t'] <= (t[0]*x[t,0]+t[1]*x[t,1]) - (t[0]*x[t',0]+t[1]*x[t',1])
    #    =>  pay[t] - pay[t'] - sum_g t[g]*(x[t,g] - x[t',g]) <= 0
    #    =>  pay[t] - pay[t'] - t[0]*x[t,0] - t[1]*x[t,1] + t[0]*x[t',0] + t[1]*x[t',1] <= 0

    support_list = list(valid_support)
    for t in support_list:
        ti = type_index[t]
        for t2 in support_list:
            if t == t2:
                continue
            ti2 = type_index[t2]
            row = np.zeros(n_vars)
            row[32 + ti] = 1.0       # pay[t]
            row[32 + ti2] = -1.0     # -pay[t']
            row[ti * 2 + 0] = -float(t[0])   # -t[0]*x[t,0]
            row[ti * 2 + 1] = -float(t[1])   # -t[1]*x[t,1]
            row[ti2 * 2 + 0] = float(t[0])   # +t[0]*x[t',0]
            row[ti2 * 2 + 1] = float(t[1])   # +t[1]*x[t',1]
            A_ub.append(row)
            b_ub.append(0.0)

    # 3. Monotonicity constraints (helps LP solver converge faster)
    #    x[t+e, g] >= x[t, g] for adjacent types
    #    x[t,g] - x[t+e,g] <= 0
    for i in range(4):
        for j in range(4):
            t = (i, j)
            idx = type_index[t]
            if i < 3:
                t_next = (i + 1, j)
                idx_next = type_index[t_next]
                for g in range(2):
                    row = np.zeros(n_vars)
                    row[idx * 2 + g] = 1.0
                    row[idx_next * 2 + g] = -1.0
                    A_ub.append(row)
                    b_ub.append(0.0)
            if j < 3:
                t_next = (i, j + 1)
                idx_next = type_index[t_next]
                for g in range(2):
                    row = np.zeros(n_vars)
                    row[idx * 2 + g] = 1.0
                    row[idx_next * 2 + g] = -1.0
                    A_ub.append(row)
                    b_ub.append(0.0)

    A_ub = np.array(A_ub, dtype=float)
    b_ub = np.array(b_ub, dtype=float)

    # Bounds: allocations in [0,1], payments in [0, 6] (max value is 3+3=6)
    bounds = []
    for i in range(32):
        bounds.append((0.0, 1.0))
    for i in range(16):
        bounds.append((0.0, 6.0))

    best_alloc = None
    best_pay = None
    best_obj = -np.inf

    # Try solving with HiGHS
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and res.x is not None:
            alloc = np.clip(res.x[:32].reshape(16, 2), 0, 1)
            pay = np.clip(res.x[32:], 0, 6)
            obj = float(c.dot(res.x))
            if obj > best_obj:
                best_obj = obj
                best_alloc = alloc
                best_pay = pay
    except Exception:
        pass

    if time.time() > deadline:
        if best_alloc is not None:
            return {"allocations": best_alloc, "payments": best_pay}
        return _fallback()

    # Try solving with interior point as backup
    try:
        res2 = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='interior-point')
        if res2.success and res2.x is not None:
            alloc = np.clip(res2.x[:32].reshape(16, 2), 0, 1)
            pay = np.clip(res2.x[32:], 0, 6)
            obj = float(c.dot(res2.x))
            if obj > best_obj:
                best_obj = obj
                best_alloc = alloc
                best_pay = pay
    except Exception:
        pass

    if best_alloc is not None:
        return {"allocations": best_alloc, "payments": best_pay}

    return _fallback()


def _fallback():
    """Simple fallback: allocate good g if value >= 2, payment = 2 if allocated."""
    alloc = np.zeros((16, 2))
    pay = np.zeros(16)
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            if i >= 2:
                alloc[idx, 0] = 1.0
                pay[idx] += 2.0
            if j >= 2:
                alloc[idx, 1] = 1.0
                pay[idx] += 2.0
    return {"allocations": alloc, "payments": pay}

# EVOLVE-BLOCK-END
