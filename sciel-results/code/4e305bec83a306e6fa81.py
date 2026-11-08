"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin before 120s kill


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and optionally which machine).
    
    Heuristic:
    - Primary: Maximize remaining_work (Longest Remaining Processing Time).
      This helps minimize makespan by ensuring long jobs are started early.
    - Secondary: Minimize min_duration (Shortest Processing Time on best machine).
      This breaks ties by preferring operations that can be completed quickly,
      potentially freeing up resources sooner or reducing blocking.
    
    Returns: (index, machine) tuple to explicitly select the best machine,
             or just index if only one option.
    """
    # Deadline guard
    if _time.time() > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # For each operation, find the best (shortest) machine option
    best_machine_indices = []
    best_durations = []
    remaining_works = []
    
    for i, op in enumerate(ready_operations):
        mw = op.get("remaining_work", 0)
        
        # Find the machine with shortest duration for this operation
        machine_options = op.get("machine_options", {})
        if machine_options:
            best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
            best_dur = machine_options[best_machine]
        else:
            best_machine = None
            best_dur = 0
        
        best_machine_indices.append(best_machine)
        best_durations.append(best_dur)
        remaining_works.append(mw)
    
    # Strategy: Prioritize by remaining_work (LRPT-like), then by shortest duration
    # We want to maximize remaining_work first, then minimize duration
    
    # Find max remaining_work to identify top candidates
    max_rw = max(remaining_works) if remaining_works else 0
    
    # Find operations with maximum remaining_work
    max_rw_ops = [i for i in range(n) if remaining_works[i] == max_rw]
    
    if len(max_rw_ops) == 1:
        best_idx = max_rw_ops[0]
    else:
        # Among ops with max remaining_work, pick the one with shortest duration
        best_idx = min(max_rw_ops, key=lambda i: best_durations[i])
    
    # Return (index, machine) to allow explicit machine selection
    if best_machine_indices[best_idx] is not None:
        return (best_idx, best_machine_indices[best_idx])
    else:
        return best_idx
# EVOLVE-BLOCK-END
