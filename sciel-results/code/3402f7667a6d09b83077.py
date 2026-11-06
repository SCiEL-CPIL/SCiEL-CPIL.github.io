"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

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

    # Precompute per-op info
    min_durs = []
    best_machines = []
    flexibilities = []
    remaining_works = []
    remaining_ops = []
    jobs = []
    urgencies = []
    
    # Count how many ready ops share each machine as their best choice
    machine_load = {}
    
    for op in ready_operations:
        opts = op["machine_options"]
        best_m = min(opts, key=lambda m: (opts[m], m))
        best_machines.append(best_m)
        min_durs.append(opts[best_m])
        flexibilities.append(len(opts))
        remaining_works.append(op["remaining_work"])
        remaining_ops.append(op["remaining_ops"])
        jobs.append(op["job"])
        urgency = op["remaining_work"] / max(1, op["remaining_ops"])
        urgencies.append(urgency)
        machine_load[best_m] = machine_load.get(best_m, 0) + 1
    
    # Contention: how many ops are competing for the same best machine
    contentions = []
    for bm in best_machines:
        contentions.append(machine_load[bm])

    # Deadline guard: fall back to a simple deterministic choice.
    if not _elapsed_ok():
        idx = max(range(n), key=lambda i: ready_operations[i]["remaining_work"])
        return (idx, best_machines[idx])

    # Load-Balancing Water-Filling Heuristic
    # 1. Estimate current machine loads based on contention (proxy for accumulated work)
    #    We assume each "ready" op on a machine adds to its load.
    #    To be more accurate, we could sum remaining_work, but we don't know which machine they *will* go to.
    #    However, we can simulate: if we pick an op, where does it go?
    #    Strategy: For each candidate, find the machine that minimizes the *new* max load.
    
    # Base load estimate: Use contention as a proxy for how "busy" a machine is with ready ops.
    # A more sophisticated approach: Sum of durations of ready ops assigned to that machine?
    # We don't have that info directly. We only have `machine_options` for *ready* ops.
    # Let's assume the "load" of a machine is proportional to the number of ready ops that *prefer* it,
    # or better, we simulate the assignment.
    
    # Let's define a "virtual load" for each machine based on the ready operations.
    # If an op is ready, it will eventually take a machine.
    # If we assume they take their best machine, the load is sum of min_durs for those ops.
    # Let's compute this "potential load" if all ready ops took their best machine.
    potential_loads = {}
    for i in range(n):
        bm = best_machines[i]
        potential_loads[bm] = potential_loads.get(bm, 0.0) + min_durs[i]
    
    # For each candidate operation i:
    # 1. Identify its machine options.
    # 2. For each machine m in options:
    #    - Calculate new_load = potential_loads.get(m, 0) + duration(i, m)
    #    - Calculate new_max_load = max(new_load, max(other_potential_loads))
    #    - Note: This is an approximation because other ops might also move.
    #    - But for a greedy step, we want to pick the (i, m) pair that minimizes the resulting system imbalance.
    #    
    # Alternative: Score based on "Regret" or "Cost".
    # Cost(i) = min over m in options(i) of [ Increase in L-infinity norm ]
    # We want to pick the op with the LOWEST cost (least damage to balance) OR highest urgency?
    # The directive says: "prioritizing operations that minimally skew the global workload dist"
    # This suggests we should pick the op that *can* be placed with minimal skew.
    # However, we must also finish jobs. If we always pick the "easiest" to place, we might starve urgent jobs.
    # Let's combine:
    # Score = w_urgency * Urgency - w_balance * Balance_Cost
    
    # Let's compute the "Balance Cost" for each op.
    # Cost(i) = min_{m in options(i)} ( new_max_load_after_assignment )
    # To make it comparable, we can use the increase in max load.
    
    current_max_load = max(potential_loads.values()) if potential_loads else 0.0
    
    balance_costs = []
    chosen_machines = []
    
    for i in range(n):
        opts = ready_operations[i]["machine_options"]
        min_cost = float('inf')
        best_m_for_balance = None
        
        for m, dur in opts.items():
            # Simulate assigning op i to machine m
            # Remove op i from its "current" best machine load in potential_loads?
            # This is getting complex. Let's simplify.
            # Let's just look at the "slack" or "room" on the machine.
            
            # Heuristic: Prefer machines with lower current potential load.
            # If we assign to m, the load becomes potential_loads.get(m, 0) + dur.
            # The "penalty" is how much this exceeds the current max load, or simply the new load itself.
            
            new_load = potential_loads.get(m, 0.0) + dur
            
            # We want to minimize the new max load.
            # new_max = max(new_load, max(potential_loads excluding m? or including?))
            # If we include m in potential_loads, and we are moving op i to m,
            # we should subtract min_durs[i] from best_machines[i] and add dur to m.
            
            old_best = best_machines[i]
            
            # Adjusted loads if op i moves to m
            # This is O(M) per machine option, so O(N*M*M) total. N=10, M=5. 10*5*5 = 250 ops. Very fast.
            
            # Let's do the proper simulation.
            adjusted_loads = {}
            for k, v in potential_loads.items():
                adjusted_loads[k] = v
            
            # Subtract from old best
            if old_best in adjusted_loads:
                adjusted_loads[old_best] -= min_durs[i]
            
            # Add to new machine
            adjusted_loads[m] = adjusted_loads.get(m, 0.0) + dur
            
            new_max = max(adjusted_loads.values())
            
            if new_max < min_cost:
                min_cost = new_max
                best_m_for_balance = m
        
        balance_costs.append(min_cost)
        chosen_machines.append(best_m_for_balance)

    # Now we have for each op:
    # - urgency
    # - balance_cost (the resulting max load if we optimally place this op)
    
    # We want to pick the op that balances urgency and balance.
    # High urgency should be prioritized.
    # Low balance_cost (better balance) should be prioritized.
    
    # Normalize urgency (higher is better)
    min_u = min(urgencies)
    max_u = max(urgencies)
    if max_u == min_u:
        norm_urg = [1.0] * n
    else:
        norm_urg = [(u - min_u) / (max_u - min_u) for u in urgencies]
        
    # Normalize balance cost (lower is better -> invert)
    min_c = min(balance_costs)
    max_c = max(balance_costs)
    if max_c == min_c:
        norm_bal = [1.0] * n
    else:
        norm_bal = [(max_c - c) / (max_c - min_c) for c in balance_costs]
        
    # Weights
    # The directive emphasizes load balancing. Let's give it significant weight.
    # But we can't ignore urgency or we'll have long makespans.
    # Try w_urg=0.5, w_bal=0.5 initially.
    
    w_urg = 0.5
    w_bal = 0.5
    
    scores = []
    for i in range(n):
        score = w_urg * norm_urg[i] + w_bal * norm_bal[i]
        # Tie breaker: prefer lower duration if scores are close?
        # Or just job id for determinism.
        score += (jobs[i] * 1e-9)
        scores.append(score)
        
    max_score = -1
    best_idx = 0
    for i in range(n):
        if scores[i] > max_score:
            max_score = scores[i]
            best_idx = i
            
    return (best_idx, chosen_machines[best_idx])
# EVOLVE-BLOCK-END
