"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Global deadline guard: 120s wall-clock limit, keep safety margin
_DEADLINE = _time.time() + 115.0  # 5s safety margin
_CALL_COUNT = 0


def _elapsed() -> float:
    return _time.time() - _time.time()


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Strategy: Dynamic Urgency/Cost ratio combining:
    1. Job criticality (remaining work) - prioritize jobs with more work left
    2. Operation urgency (fewer remaining ops) - prioritize jobs closer to completion
    3. Machine contention penalty - avoid machines that are heavily loaded
    4. Processing time (SPT) - prefer shorter operations to reduce makespan
    
    Returns: index of the ready operation to start, optionally (index, machine)
    """
    global _CALL_COUNT
    _CALL_COUNT += 1
    
    # Deadline guard: if we're running out of time, fall back to simple rule
    if _time.time() > _DEADLINE:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Pre-compute arrays for speed
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # Machine options: for each op, list of (machine, duration)
    # We need to compute a score for each operation
    
    # Normalize remaining work and ops to [0, 1] range
    rw_min, rw_max = remaining_work.min(), remaining_work.max()
    ro_min, ro_max = remaining_ops.min(), remaining_ops.max()
    
    rw_span = rw_max - rw_min if rw_max > rw_min else 1.0
    ro_span = ro_max - ro_min if ro_max > ro_min else 1.0
    
    rw_norm = (remaining_work - rw_min) / rw_span
    ro_norm = (remaining_ops - ro_min) / ro_span
    
    # For each operation, compute the best machine option score
    # Score = urgency * (1 - contention) * (1 / duration_factor)
    
    # First, gather all machine options and their durations
    # For contention: count how many ready ops can use each machine
    machine_counts = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_counts[m] = machine_counts.get(m, 0) + 1
    
    total_ops = sum(machine_counts.values()) if machine_counts else 1
    
    # Compute scores
    scores = np.zeros(n, dtype=np.float64)
    
    for i, op in enumerate(ready_operations):
        # Urgency: high remaining work AND low remaining ops (closer to done)
        # Combine: weight remaining work heavily, but also consider position in job
        urgency = 0.6 * rw_norm[i] + 0.4 * (1.0 - ro_norm[i])
        
        # Best machine option: pick the one with lowest duration, penalized by contention
        best_score = -np.inf
        for m, dur in op["machine_options"].items():
            if dur <= 0:
                continue
            # Contention penalty: more ops wanting this machine = worse
            contention = machine_counts.get(m, 0) / total_ops
            # Duration penalty: shorter is better
            dur_penalty = 1.0 / (dur + 1e-9)
            
            # Combine: prefer low contention and short duration
            machine_score = (1.0 - 0.3 * contention) * dur_penalty
            if machine_score > best_score:
                best_score = machine_score
        
        if not np.isfinite(best_score):
            best_score = 0.0
        
        scores[i] = urgency * best_score
    
    # Add a small tie-breaker based on operation index for stability
    scores += np.arange(n, dtype=np.float64) * 1e-10
    
    best_idx = int(np.argmax(scores))
    
    # Optionally return machine selection
    # Try to return (index, machine) if the interface supports it
    try:
        best_op = ready_operations[best_idx]
        # Pick the machine with the best (1-contention)/duration ratio
        best_machine = None
        best_m_score = -np.inf
        for m, dur in best_op["machine_options"].items():
            if dur <= 0:
                continue
            contention = machine_counts.get(m, 0) / total_ops
            m_score = (1.0 - 0.3 * contention) / (dur + 1e-9)
            if m_score > best_m_score:
                best_m_score = m_score
                best_machine = m
        if best_machine is not None:
            return (best_idx, best_machine)
    except Exception:
        pass
    
    return best_idx
# EVOLVE-BLOCK-END
