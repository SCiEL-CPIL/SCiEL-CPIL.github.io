"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin before 120s kill


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and optionally which machine).
    
    Uses a refined composite heuristic for flexible job-shop scheduling:
      1. Prioritize operations from jobs with most remaining work (CR-like).
      2. Strongly prefer shorter processing times on the best machine.
      3. Consider remaining operation count as a secondary factor.
      4. Machine selection: pick the machine with shortest duration for the chosen op.
    
    Returns: (index, machine) tuple to explicitly select the best machine,
             or just index if only one option.
    """
    # Deadline guard
    if _time.time() > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Collect data for each ready operation
    best_machines = []
    best_durations = []
    remaining_works = []
    remaining_ops = []
    
    for op in ready_operations:
        mw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        
        machine_options = op.get("machine_options", {})
        if machine_options:
            best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
            best_dur = machine_options[best_machine]
        else:
            best_machine = None
            best_dur = 0
        
        best_machines.append(best_machine)
        best_durations.append(best_dur)
        remaining_works.append(mw)
        remaining_ops.append(ro)
    
    # Composite scoring:
    #   - remaining_work (higher = more urgent) : weight 0.4
    #   - processing_time (lower = better)      : weight 0.4  
    #   - remaining_ops (higher = more work ahead) : weight 0.2
    
    max_rw = max(remaining_works) if remaining_works else 1
    min_dur = min(best_durations) if best_durations else 1
    max_dur = max(best_durations) if best_durations else 1
    max_ro = max(remaining_ops) if remaining_ops else 1
    
    best_idx = 0
    best_score = float('-inf')
    
    for i in range(n):
        rw = remaining_works[i]
        dur = best_durations[i]
        ro = remaining_ops[i]
        
        # Normalize remaining work: higher is better
        rw_norm = rw / max_rw if max_rw > 0 else 0
        
        # Normalize duration: shorter is better
        if max_dur > min_dur:
            dur_norm = 1.0 - (dur - min_dur) / (max_dur - min_dur)
        else:
            dur_norm = 1.0
        
        # Normalize remaining ops: higher is better  
        ro_norm = ro / max_ro if max_ro > 0 else 0
        
        # Weighted composite score
        score = 0.4 * rw_norm + 0.4 * dur_norm + 0.2 * ro_norm
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    # Return (index, machine) to allow explicit machine selection
    if best_machines[best_idx] is not None:
        return (best_idx, best_machines[best_idx])
    else:
        return best_idx
# EVOLVE-BLOCK-END
