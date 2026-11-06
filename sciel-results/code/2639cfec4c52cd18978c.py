"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

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
        # pick machine with smallest duration
        best_m = min(opts, key=lambda m: (opts[m], m))
        return (0, best_m)

    # Pre-compute best machine and min duration for each operation
    # This is the "Model" input
    best_machines = []
    min_durs = []
    rem_works = []
    rem_ops = []
    machine_options_list = []
    
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        best_m = min(opts, key=lambda m: (opts[m], m))
        best_machines.append(best_m)
        min_durs.append(opts[best_m])
        rem_works.append(op["remaining_work"])
        rem_ops.append(op["remaining_ops"])
        machine_options_list.append(opts)

    # Deadline guard: fall back to simple deterministic choice
    if not _elapsed_ok():
        idx = int(np.argmax(rem_works))
        return (idx, best_machines[idx])

    # Load-Balancing Water-Filling Heuristic:
    # Estimate machine load by counting how many ready operations can use each machine
    # Prioritize operations that can be assigned to the least-loaded compatible machine
    # This minimizes the increase in max machine load (L-infinity norm)
    
    # Count machine availability (fewer ops using a machine = less loaded)
    machine_load_proxy = {}
    for opts in machine_options_list:
        for m in opts:
            machine_load_proxy[m] = machine_load_proxy.get(m, 0) + 1
    
    # For each operation, find the least-loaded machine among its options
    # and score by how well it balances loads
    scores = []
    for i in range(n):
        opts = machine_options_list[i]
        # Find the machine with minimum load proxy among options
        # Tie-break by duration (shorter is better to free machine faster)
        best_m_for_balance = min(opts, key=lambda m: (machine_load_proxy.get(m, 0), opts[m], m))
        load_score = machine_load_proxy.get(best_m_for_balance, 0)
        dur = opts[best_m_for_balance]
        
        # Composite score:
        # Primary: lower load on chosen machine (water-filling)
        # Secondary: fewer remaining ops (closer to job completion)
        # Tertiary: larger remaining work (LPT to avoid long jobs waiting)
        # Quaternary: shorter duration (SPT to free machine)
        scores.append((
            load_score,
            rem_ops[i],
            -rem_works[i],
            dur,
            i
        ))
    
    scores.sort()
    idx = scores[0][4]
    # Use the balanced machine choice for this operation
    opts = machine_options_list[idx]
    best_m_for_balance = min(opts, key=lambda m: (machine_load_proxy.get(m, 0), opts[m], m))
    return (idx, best_m_for_balance)
# EVOLVE-BLOCK-END
