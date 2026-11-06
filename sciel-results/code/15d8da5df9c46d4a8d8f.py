"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

_START = _time.time()
_DEADLINE = 150.0 - 5.0  # safety margin


def _elapsed_ok():
    return (_time.time() - _START) < _DEADLINE


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        best_m = min(opts, key=lambda m: (opts[m], m))
        return (0, best_m)

    # Precompute per-op info
    min_durs = []
    best_machines = []
    flexibilities = []
    remaining_works = []
    remaining_ops = []
    jobs = []
    avg_durs = []
    max_durs = []
    
    for op in ready_operations:
        opts = op["machine_options"]
        best_m = min(opts, key=lambda m: (opts[m], m))
        best_machines.append(best_m)
        min_durs.append(opts[best_m])
        flexibilities.append(len(opts))
        remaining_works.append(op["remaining_work"])
        remaining_ops.append(op["remaining_ops"])
        jobs.append(op["job"])
        
        # Compute average and max duration across all machine options
        vals = list(opts.values())
        avg_durs.append(sum(vals) / len(vals))
        max_durs.append(max(vals))

    # Deadline guard: fall back to a simple deterministic choice.
    if not _elapsed_ok():
        idx = max(range(n), key=lambda i: ready_operations[i]["remaining_work"])
        return (idx, best_machines[idx])

    # Local normalization helper
    def normalize(vals, reverse=False):
        min_v = min(vals)
        max_v = max(vals)
        if max_v == min_v:
            return [1.0] * len(vals)
        if reverse:
            return [(max_v - v) / (max_v - min_v) for v in vals]
        else:
            return [(v - min_v) / (max_v - min_v) for v in vals]

    # We want to maximize the score.
    # 1. Remaining Work: Higher is better (0.45)
    # 2. Min Duration: Lower is better (0.20) -> Invert
    # 3. Flexibility: Lower is better (0.20) -> Invert
    # 4. Remaining Ops: Lower is better (0.10) -> Invert
    # 5. Average Duration: Lower is better (0.05) -> Invert (tie-breaker)
    
    norm_rw = normalize(remaining_works)
    norm_md = normalize(min_durs, reverse=True)
    norm_fl = normalize(flexibilities, reverse=True)
    norm_ro = normalize(remaining_ops, reverse=True)
    norm_ad = normalize(avg_durs, reverse=True)
    
    # Weights
    w_rw = 0.45
    w_md = 0.20
    w_fl = 0.20
    w_ro = 0.10
    w_ad = 0.05
    
    scores = []
    for i in range(n):
        score = (w_rw * norm_rw[i] + 
                 w_md * norm_md[i] + 
                 w_fl * norm_fl[i] + 
                 w_ro * norm_ro[i] +
                 w_ad * norm_ad[i])
        # Add a tiny epsilon based on job id for determinism in case of exact ties
        score += (jobs[i] * 1e-9)
        scores.append(score)
        
    # Find the index with the highest score
    max_score = -1
    best_idx = 0
    for i in range(n):
        if scores[i] > max_score:
            max_score = scores[i]
            best_idx = i
            
    return (best_idx, best_machines[best_idx])
# EVOLVE-BLOCK-END
