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

    # Pre-calculate metrics for each ready operation
    min_durs = []
    best_machines = []
    is_last_ops = []
    remaining_works = []

    for op in ready_operations:
        opts = op["machine_options"]
        # Find the fastest machine for this operation (SPT)
        best_m = min(opts, key=lambda m: (opts[m], m))
        best_machines.append(best_m)
        min_durs.append(opts[best_m])
        is_last_ops.append(op["remaining_ops"] == 1)
        remaining_works.append(op["remaining_work"])

    # Deadline guard: fall back to a simple deterministic choice.
    if not _elapsed_ok():
        idx = int(np.argmax(remaining_works))
        return (idx, best_machines[idx])

    # Load-Balancing Water-Filling heuristic (refined):
    # 1. Better machine-load estimation via iterative greedy assignment
    # 2. Smooth soft-max surrogate for the L-infinity objective
    # 3. Auxiliary variable t (epigraph): target the minimum achievable peak
    
    # Collect all machines
    all_machines = set()
    for op in ready_operations:
        all_machines.update(op["machine_options"].keys())
    
    if not all_machines:
        idx = 0
        return (idx, best_machines[idx])
    
    # --- Pass 1: greedy assignment to build initial machine loads ---
    # Assign each op to its least-loaded compatible machine (greedy water-filling)
    machine_loads = {m: 0.0 for m in all_machines}
    # Sort ops by min duration (SPT) for more stable greedy assignment
    op_order = sorted(range(n), key=lambda i: (min_durs[i], -remaining_works[i]))
    for i in op_order:
        opts = ready_operations[i]["machine_options"]
        # Assign to least-loaded compatible machine
        best_m_for_load = min(opts.keys(), key=lambda m: (machine_loads.get(m, 0.0), m))
        machine_loads[best_m_for_load] += opts[best_m_for_load]
    
    # --- Pass 2: re-estimate with weighted contribution ---
    # Blend greedy assignment with weighted contribution for robustness
    machine_loads_v2 = {m: 0.0 for m in all_machines}
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        best_m = best_machines[i]
        for m, dur in opts.items():
            weight = 1.0 if m == best_m else 0.3
            machine_loads_v2[m] += dur * weight
    
    # Blend: 60% greedy, 40% weighted
    for m in all_machines:
        machine_loads[m] = 0.6 * machine_loads[m] + 0.4 * machine_loads_v2[m]
    
    # Compute baseline statistics
    load_values = list(machine_loads.values())
    current_max_load = max(load_values) if load_values else 0.0
    current_mean_load = sum(load_values) / len(load_values) if load_values else 0.0
    
    # Soft-max temperature: smaller = sharper (closer to true max)
    # Scale relative to typical load magnitude for numerical stability
    typical_load = current_max_load if current_max_load > 0 else 1.0
    beta = 5.0 / max(typical_load, 1.0)
    
    # --- Score each candidate using soft-max surrogate + auxiliary t ---
    scores = []
    for i in range(n):
        op = ready_operations[i]
        opts = op["machine_options"]
        
        # Find the least-loaded compatible machine (water-filling choice)
        min_load_machine = min(opts.keys(), key=lambda m: (machine_loads.get(m, 0.0), m))
        dur_on_machine = opts[min_load_machine]
        
        # Simulate: new load vector with this op added
        # For soft-max, compute log-sum-exp of (loads + assignment) / beta
        # Only the assigned machine changes
        new_load_assigned = machine_loads.get(min_load_machine, 0.0) + dur_on_machine
        
        # Soft-max: approximates max(load_values) but smooth
        # Use log-sum-exp for numerical stability
        # soft_max = (1/beta) * log(sum(exp(beta * load_m)))
        # After assignment: one load changes from L to L+d
        # We compute the ratio to avoid overflow
        # soft_max_new / soft_max_old = exp(beta*d) * (exp(beta*L) + exp(beta*(L+d)) - exp(beta*L)) / sum
        # Simpler: just compute the new soft-max directly with a cap
        
        # Compute soft-max of new loads (only one changes)
        # For efficiency, precompute base soft-max components
        # But n is small (ready ops), so just compute directly
        
        # New loads: all same except min_load_machine gets +dur
        # soft_max_new = (1/beta) * log( sum_{m != assigned} exp(beta*load_m) + exp(beta*(load_assigned + dur)) )
        # = (1/beta) * log( sum_all exp(beta*load_m) - exp(beta*load_assigned) + exp(beta*(load_assigned+dur)) )
        
        # Precompute: we'll just compute the change in soft-max
        # delta_soft_max ≈ (1/beta) * log( 1 + exp(beta*dur) - 1 ) * exp(beta*load_assigned) / sum_exp
        # Simplify: for small beta*dur, delta ≈ dur * exp(beta*load_assigned) / sum_exp(beta*load_m)
        
        # Compute sum_exp and exp_assigned
        exp_assigned = np.exp(beta * machine_loads.get(min_load_machine, 0.0))
        # sum_exp = sum of exp(beta * load_m) for all m
        # We can approximate: sum_exp ≈ exp_assigned + (len(all_machines)-1) * mean_exp
        # But let's just compute it since all_machines is small
        sum_exp = 0.0
        for m in all_machines:
            sum_exp += np.exp(beta * machine_loads.get(m, 0.0))
        
        if sum_exp < 1e-300:
            sum_exp = 1.0
        
        # Fraction of "mass" on the assigned machine
        frac_assigned = exp_assigned / sum_exp
        
        # Change in soft-max when adding dur to assigned machine:
        # soft_max_new = (1/beta)*log(sum_exp - exp_assigned + exp_assigned*exp(beta*dur))
        # = (1/beta)*log(sum_exp * (1 - frac_assigned + frac_assigned*exp(beta*dur)))
        # = (1/beta)*log(sum_exp) + (1/beta)*log(1 - frac_assigned + frac_assigned*exp(beta*dur))
        # delta_soft_max = (1/beta)*log(1 - frac_assigned + frac_assigned*exp(beta*dur))
        
        # For numerical stability when beta*dur is large:
        bd = beta * dur_on_machine
        if bd > 50:
            # exp(bd) dominates
            delta_soft_max = (1.0 / beta) * bd  # ≈ dur_on_machine
        elif bd > 20:
            # Use log1p for accuracy
            # 1 - frac + frac*exp(bd) = 1 + frac*(exp(bd) - 1)
            inner = 1.0 + frac_assigned * (np.exp(bd) - 1.0)
            delta_soft_max = (1.0 / beta) * np.log(inner)
        else:
            inner = 1.0 + frac_assigned * (np.exp(bd) - 1.0)
            delta_soft_max = (1.0 / beta) * np.log(inner)
        
        # Auxiliary variable t: the new estimated peak load
        # t = max(current_max_load, new_load_assigned) — but use soft version
        # t_soft = current soft-max + delta_soft_max
        t_soft = (1.0 / beta) * np.log(sum_exp) + delta_soft_max
        
        # Hard max for tie-breaking and secondary criteria
        new_load_assigned = machine_loads.get(min_load_machine, 0.0) + dur_on_machine
        t_hard = max(current_max_load, new_load_assigned)
        
        # Deviation from mean after assignment
        deviation = abs(new_load_assigned - current_mean_load)
        
        # Load balance spread: how uneven are the loads after assignment?
        # Approximate by std of loads (only one changes)
        # New mean ≈ current_mean + dur_on_machine / len(all_machines)
        new_mean = current_mean_load + dur_on_machine / len(all_machines)
        
        # Score: primarily minimize soft-max increase (smooth surrogate for L∞)
        # Secondarily minimize hard peak, then deviation, then tie-breakers
        score = (
            delta_soft_max,          # Smooth surrogate: how much does the soft-peak increase
            t_hard,                  # Hard peak load after assignment
            deviation,               # Deviation from mean
            0 if is_last_ops[i] else 1,  # Prefer last ops
            min_durs[i],             # SPT
            -remaining_works[i]      # LPT tie-break
        )
        scores.append(score)
    
    # Pick the operation with the best (lowest) score
    idx = min(range(n), key=lambda i: scores[i])
    best_m = best_machines[idx]
    return (idx, best_m)
# EVOLVE-BLOCK-END
