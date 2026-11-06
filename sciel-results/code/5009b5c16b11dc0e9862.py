"""Optimized dispatch rule for open-shop scheduling."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

# Global state for tracking simulation progress
_sim_state = {}
_DEADLINE = 110.0
_start_time = None

def _reset_sim_state():
    global _sim_state, _start_time
    _sim_state = {
        'machine_load': defaultdict(float),  # Estimated remaining work per machine
        'machine_total_work': defaultdict(float),  # Total work seen per machine
        'call_count': 0,
        'jobs_seen': set(),
        'ops_per_job': defaultdict(int),
        'last_time': 0.0,
    }
    _start_time = time.time()

def _check_time():
    if _start_time is None:
        return False
    return (time.time() - _start_time) > _DEADLINE

def _soft_min(values, temp=1.0):
    """Smooth minimum via log-sum-exp trick."""
    if not values:
        return 0.0
    v = min(values)
    if temp <= 0:
        return v
    # exp(-(x_i - v)/temp) for stability
    exp_sum = sum(math.exp(-(x - v) / temp) for x in values)
    return v - temp * math.log(exp_sum)

def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    
    Strategy:
    1. Track machine loads across calls to make informed machine assignments
    2. Use MWR (Most Work Remaining) as primary job selection criterion
    3. Use soft-min SPT with load balancing for machine selection
    4. Consider operation blocking and machine criticality
    
    Returns: (index, machine) tuple
    """
    # Initialize on first call
    if _start_time is None:
        _reset_sim_state()
    
    if not ready_operations:
        return (0, None)
    
    # Time guard
    if _check_time():
        best_idx = max(range(len(ready_operations)), 
                      key=lambda i: ready_operations[i]["remaining_work"])
        op = ready_operations[best_idx]
        machine_options = op.get("machine_options", {})
        if machine_options:
            best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
        else:
            best_machine = None
        return (best_idx, best_machine)
    
    state = _sim_state
    state['call_count'] += 1
    
    # Update state: track machine loads based on elapsed time
    dt = max(0.0, time - state['last_time'])
    state['last_time'] = time
    
    for op in ready_operations:
        job = op.get('job')
        if job is not None:
            state['jobs_seen'].add(job)
        # Estimate remaining work on each machine from this op
        for m, d in op.get('machine_options', {}).items():
            state['machine_total_work'][m] += d
    
    # Get max remaining work for normalization
    max_work = max(op["remaining_work"] for op in ready_operations)
    if max_work == 0:
        max_work = 1
    
    # Get max remaining ops for normalization
    max_ops = max(op["remaining_ops"] for op in ready_operations)
    if max_ops == 0:
        max_ops = 1
    
    # Compute total work seen across all machines for normalization
    total_all_work = sum(state['machine_total_work'].values())
    if total_all_work == 0:
        total_all_work = 1
    
    scores = []
    best_machines = []
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        machine_options = op.get("machine_options", {})
        if not machine_options:
            scores.append(0)
            best_machines.append(None)
            continue
        
        durations = list(machine_options.values())
        min_duration = min(durations)
        max_duration = max(durations)
        num_machines = len(machine_options)
        
        # Component 1: Most Work Remaining (primary) - normalized
        work_score = remaining_work / max_work
        
        # Component 2: Remaining operations (secondary) - normalized
        ops_score = remaining_ops / max_ops
        
        # Component 3: Urgency - if few ops remain, prioritize to avoid idle time
        urgency_score = 1.0 / (remaining_ops + 1.0) if remaining_ops > 0 else 1.0
        
        # Component 4: Machine selection using soft-min surrogate
        # For each machine, compute a "cost" = duration + load_penalty
        # Use soft-min to smoothly select the best machine
        machine_costs = []
        machine_list = []
        for m, d in machine_options.items():
            # Normalize load: fraction of total work assigned to this machine
            load_frac = state['machine_total_work'].get(m, 0.0) / total_all_work
            # Cost = duration * (1 + alpha * load_frac)
            # Higher load_frac means this machine is busier, so increase cost
            cost = d * (1.0 + 0.3 * load_frac)
            machine_costs.append(cost)
            machine_list.append(m)
        
        # Soft-min to get a smooth "best cost" estimate
        soft_min_cost = _soft_min(machine_costs, temp=0.5)
        
        # Find the actual best machine (hard argmin) for return
        best_machine_idx = machine_costs.index(min(machine_costs))
        best_machine = machine_list[best_machine_idx]
        
        # Normalize machine score: lower cost is better
        # Scale to [0, 1] where 1 is best
        if max_duration > 0:
            machine_score = 1.0 - (soft_min_cost / (max_duration * 1.3))
        else:
            machine_score = 0.5
        machine_score = max(0.0, min(1.0, machine_score))
        
        # Component 5: Flexibility bonus - more options is slightly better
        flex_score = min(num_machines / 5.0, 1.0)
        
        # Component 6: Blocking/criticality detection
        # If this op has few machine options, it's more critical to schedule
        # because it constrains future scheduling
        criticality_bonus = 0.0
        if num_machines <= 1:
            criticality_bonus = 0.25
        elif num_machines == 2:
            criticality_bonus = 0.10
        
        # Near completion bonus
        if remaining_ops <= 1:
            criticality_bonus += 0.20
        
        # Weighted combination
        w_work = 0.45
        w_ops = 0.10
        w_urgency = 0.10
        w_machine = 0.20
        w_flex = 0.05
        w_critical = 0.10
        
        score = (w_work * work_score + 
                 w_ops * ops_score + 
                 w_urgency * urgency_score + 
                 w_machine * machine_score + 
                 w_flex * flex_score +
                 w_critical * criticality_bonus)
        
        scores.append(score)
        best_machines.append(best_machine)
    
    best_idx = max(range(len(ready_operations)), key=lambda i: scores[i])
    return (best_idx, best_machines[best_idx])

# EVOLVE-BLOCK-END
