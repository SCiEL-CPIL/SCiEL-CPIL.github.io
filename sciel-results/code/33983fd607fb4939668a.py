"""Open-shop dispatch rule: portfolio search over heuristic constants, with
a hard wall-clock deadline guard so dispatch always returns safely."""
import time
import random
import math

# ----------------------------- deadline guard -----------------------------
_START = time.time()
_DEADLINE = _START + 120.0          # safety margin well under the 150s limit


def _time_left():
    return _DEADLINE - time.time()


# ----------------------------- heuristic family ---------------------------
# c: constant chosen by search; op: ready operation dict; machine: candidate.
def _heur(c, op, machine):
    d = op["machine_options"][machine]
    rw = op["remaining_work"]
    ro = op["remaining_ops"]
    a, b, g, p, q = c
    return (
        a * d
        + b * rw
        + g * (ro if ro > 0 else 1)
        + p / (d + 1.0)
        + q * rw / (ro + 1)
    )


def _pick(c, ready, machine_load, rng):
    """Return (index, machine, value). Ties broken by rng."""
    best = None
    best_val = None
    for i, op in enumerate(ready):
        for m, d in op["machine_options"].items():
            v = _heur(c, op, m) + 1e-9 * machine_load[m]  # tie-break bias
            if best_val is None or v < best_val:
                best_val = v
                best = (i, m)
    return best[0], best[1], best_val


# ----------------------------- instance generator -------------------------
def _gen_instance(rng, njobs=10, nmach=5, pmin=1, pmax=9):
    ops = []
    for j in range(njobs):
        for k in range(nmach):
            dur = rng.randint(pmin, pmax)
            # ~25% of ops are flexible (extra machine choice); rest pinned
            if rng.random() < 0.25:
                m2 = (k + 1) % nmach
                mopts = {k: dur, m2: max(1, dur + rng.randint(-2, 2))}
            else:
                mopts = {k: dur}
            ops.append({"job": j, "op": k, "machine_options": mopts})
    return ops


# ----------------------------- simulator ----------------------------------
def _simulate2(ops, c, rng):
    njobs = max(o["job"] for o in ops) + 1
    nmach = max(m for o in ops for m in o["machine_options"]) + 1
    # job op ordering: process ops in op-index order per job (arbitrary but valid)
    job_ops = {j: [i for i in range(len(ops)) if ops[i]["job"] == j]
               for j in range(njobs)}
    for j in range(njobs):
        job_ops[j].sort(key=lambda i: ops[i]["op"])
    rem_ops = [len(job_ops[j]) for j in range(njobs)]
    rem_work = [sum(o["machine_options"].values()) for o in ops]
    machine_load = [0] * nmach
    job_free = [False] * njobs
    # ops ready = first op of each job
    ready = [job_ops[j][0] for j in range(njobs)]
    makespan = 0
    scheduled = 0
    while scheduled < len(ops):
        # choose among ready ops whose job is free
        cands = [i for i in ready if job_free[ops[i]["job"]]]
        if not cands:
            # should not happen; fallback pick any ready
            cands = ready
        best_i = best_m = None
        best_val = None
        for i in cands:
            op = ops[i]
            for m, d in op["machine_options"].items():
                v = _heur(c, op, m) + 1e-9 * machine_load[m]
                if best_val is None or v < best_val:
                    best_val = v
                    best_i, best_m = i, m
        op = ops[best_i]
        d = op["machine_options"][best_m]
        start = machine_load[best_m]
        end = start + d
        machine_load[best_m] = end
        makespan = max(makespan, end)
        j = op["job"]
        job_free[j] = False
        scheduled += 1
        rem_ops[j] -= 1
        rem_work[best_i] = 0
        ready.remove(best_i)
        if rem_ops[j] > 0:
            nxt = job_ops[j][len(job_ops[j]) - rem_ops[j]]
            ready.append(nxt)
        else:
            job_free[j] = True
    return makespan


# ----------------------------- search -------------------------------------
def _search_budget(base=200):
    # Use spare time for more search, but never exceed the deadline.
    tl = _time_left()
    if tl <= 0:
        return 0
    # each sim is ~ njobs*nmach*log dispatches; keep conservative per-eval cap.
    n = min(3000, max(base, int(base * tl)))
    return n


def _run_search():
    rng = random.Random(20240517)
    # initial candidate pool: seed-like, plus varied
    best_c = (0.0, 1.0, 0.0, 0.0, 0.0)      # seed: max remaining_work
    best_score = float("inf")
    pool = [
        (1.0, 0.0, 0.0, 0.0, 0.0),          # SPT
        (0.0, 1.0, 0.0, 0.0, 0.0),          # seed
        (0.5, 0.5, 0.0, 0.0, 0.0),
        (1.0, 0.0, 1.0, 0.0, 0.0),
        (0.0, 0.5, 1.0, 0.5, 0.0),
        (1.0, 0.0, 0.0, 1.0, 0.0),
        (0.5, 0.5, 0.5, 0.5, 0.5),
        (1.0, 1.0, 0.0, 0.0, 0.0),
    ]
    budget = _search_budget()
    it = 0
    # evaluate pool first
    for c in list(pool):
        if it >= budget or _time_left() <= 2.0:
            break
        insts = [_gen_instance(rng) for _ in range(5)]
        mean = sum(_simulate2(ops, c, rng) for ops in insts) / 5.0
        if mean < best_score:
            best_score, best_c = mean, c
        it += 1
    # local search around best_c
    for _ in range(budget):
        if _time_left() <= 2.0:
            break
        base = best_c
        noise = [rng.gauss(0, 0.3) for _ in range(5)]
        cand = [max(0.0, base[k] + noise[k]) for k in range(5)]
        insts = [_gen_instance(rng) for _ in range(5)]
        mean = sum(_simulate2(ops, cand, rng) for ops in insts) / 5.0
        if mean < best_score - 1e-9:
            best_score, best_c = mean, cand
    return best_c


# ----------------------------- dispatch -----------------------------------
def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Lazy, one-time search with deadline guard.
    try:
        if not hasattr(dispatch, "_best_c"):
            if _time_left() > 3.0:
                dispatch._best_c = _run_search()
            else:
                dispatch._best_c = (0.0, 1.0, 0.0, 0.0, 0.0)  # seed fallback
    except Exception:
        dispatch._best_c = (0.0, 1.0, 0.0, 0.0, 0.0)
    c = dispatch._best_c

    try:
        # machine load proxy: count options popularity is not known; use 0 (pure heuristic)
        machine_load = {}
        for op in ready_operations:
            for m in op["machine_options"]:
                machine_load[m] = machine_load.get(m, 0) + 0.0
        idx, m, _v = _pick(c, ready_operations, machine_load, random)
        return (int(idx), int(m))
    except Exception:
        # safe fallback: max remaining_work
        idx = 0
        best = ready_operations[0]["remaining_work"]
        for i, op in enumerate(ready_operations[1:], 1):
            if op["remaining_work"] > best:
                best = op["remaining_work"]
                idx = i
        return int(idx)
