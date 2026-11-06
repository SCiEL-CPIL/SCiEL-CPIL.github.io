"""Optimized dispatch rule for open-shop scheduling."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

# Global state for tracking simulation progress
_sim_state = {}
_DEADLINE = 115.0
_start_time = None

def _reset_sim_state():
    global _sim_state, _start_time
    _sim_state = {
        'machine_load': defaultdict(float),  # Estimated remaining work per machine
        'machine_busy': set(),  # Machines currently busy (approximated)
        'call_count': 0,
        'total_work_initial': 0,
        'jobs_seen': set(),
        'ops_per_job': defaultdict(int),
        'total_work_seen': 0.0,
    }
    _start_time = time.time()

def _check_time():
    if _start_time is None:
        return False
    return (time.time() - _start_time) > _DEADLINE

def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling using Load-Balancing Water-Filling.
    
    Strategy:
    1. Track machine loads across calls
    2. For each candidate operation, simulate assignment to each compatible machine
    3. Score by the resulting increase in L-infinity norm (max machine load)
    4. Prioritize operations that minimally skew the global workload distribution
    
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
    
    # Update state: track total work and machine loads
    for op in ready_operations:
        job = op.get('job')
        if job is not None:
            state['jobs_seen'].add(job)
            rw = op.get('remaining_work', 0)
            if rw > 0:
                state['total_work_seen'] += rw
            # Estimate remaining work on each machine from this op
            for m, d in op.get('machine_options', {}).items():
                state['machine_load'][m] += d * 0.02  # Slightly higher weight
    
    # Get current max machine load for baseline
    current_loads = list(state['machine_load'].values())
    if not current_loads:
        current_loads = [0.0]
    current_max_load = max(current_loads)
    if current_max_load == 0:
        current_max_load = 1.0
    
    # Soft-max temperature for smooth L-infinity approximation
    # Lower temperature = sharper (closer to true max), higher = smoother
    soft_max_temp = 5.0
    
    def _soft_max(values, temp=soft_max_temp):
        """Compute soft-max approximation of max using log-sum-exp."""
        if not values:
            return 0.0
        max_val = max(values)
        if max_val == 0:
            return 0.0
        # Scale to avoid overflow
        scaled = [v / max_val for v in values]
        log_sum = sum(math.exp((v - max_val) / temp) for v in values)
        return max_val + temp * math.log(log_sum)
    
    scores = []
    best_machines = []
    
    # Get max remaining work for normalization
    max_work = max(op["remaining_work"] for op in ready_operations)
    if max_work == 0:
        max_work = 1
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        machine_options = op.get("machine_options", {})
        if not machine_options:
            scores.append(0)
            best_machines.append(None)
            continue
        
        # For each machine option, simulate the assignment and compute
        # the resulting L-infinity norm increase
        best_machine = None
        best_load_increase = float('inf')
        best_soft_load_increase = float('inf')
        
        for m, d in machine_options.items():
            # Simulate: add this operation's duration to machine m's load
            simulated_loads = list(state['machine_load'].values())
            # Find index of machine m in the loads
            m_idx = None
            all_machines = list(state['machine_load'].keys())
            if m in all_machines:
                m_idx = all_machines.index(m)
                simulated_loads[m_idx] += d
            else:
                simulated_loads.append(d)
            
            # Compute L-infinity norm (max load) after assignment
            new_max_load = max(simulated_loads)
            
            # Compute soft-max for smoother objective
            new_soft_max = _soft_max(simulated_loads)
            
            # Load increase is the key metric: how much does this assignment
            # increase the global max load?
            load_increase = new_max_load - current_max_load
            soft_load_increase = new_soft_max - current_max_load
            
            # Prefer assignments that minimize load increase
            # Use a combination: primary is load increase, secondary is soft-max
            combined_increase = load_increase * 0.7 + soft_load_increase * 0.3
            
            if combined_increase < best_load_increase:
                best_load_increase = combined_increase
                best_machine = m
        
        # Score: lower load increase is better, so invert
        # Also factor in remaining work as a tiebreaker (MWR)
        load_score = -best_load_increase  # Lower increase = higher score
        
        # MWR component: jobs with more remaining work should be prioritized
        # to avoid blocking
        mwr_score = remaining_work / max_work
        
        # Urgency: operations with fewer remaining ops should be completed
        # to free up jobs
        urgency_score = 1.0 / (remaining_ops + 1.0) if remaining_ops > 0 else 1.0
        
        # Flexibility: more machine options means more scheduling freedom
        num_machines = len(machine_options)
        flex_score = min(num_machines / 5.0, 1.0)
        
        # Combined score: primarily minimize load increase, with MWR as tiebreaker
        w_load = 0.60
        w_mwr = 0.25
        w_urgency = 0.10
        w_flex = 0.05
        
        # Normalize load_score: it's negative, so higher (less negative) is better
        # Scale to [0, 1] range for combination
        # Load increase is typically small relative to total work
        load_normalized = max(0.0, min(1.0, 1.0 - abs(best_load_increase) / (max_work + 1.0)))
        
        score = (w_load * load_normalized + 
                 w_mwr * mwr_score + 
                 w_urgency * urgency_score + 
                 w_flex * flex_score)
        
        scores.append(score)
        best_machines.append(best_machine)
    
    best_idx = max(range(len(ready_operations)), key=lambda i: scores[i])
    return (best_idx, best_machines[best_idx])

# EVOLVE-BLOCK-END
