"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE_START = _time.time()
_MAX_WALL = 140.0  # Safety margin before the 150s kill


def _check_deadline(fallback):
    """If we're getting close to the wall-clock limit, return fallback immediately."""
    if _time.time() - _DEADLINE_START > _MAX_WALL:
        return fallback
    return None


def dispatch(ready_operations, time):
    # Check deadline first; if we're over, pick the first available (index 0)
    fallback = 0
    early = _check_deadline(fallback)
    if early is not None:
        return early

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        # Return the single operation with its best machine
        op = ready_operations[0]
        if op["machine_options"]:
            best_machine = min(op["machine_options"], key=lambda m: op["machine_options"][m])
            return (0, best_machine)
        return 0

    # Extract basic features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # Exact min duration per operation
    exact_min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)
    
    # Number of machine options (flexibility)
    n_machines = np.array([
        len(op["machine_options"]) if op["machine_options"] else 0
        for op in ready_operations
    ], dtype=np.float64)
    
    # Intensity: average remaining work per remaining op
    safe_ops = np.maximum(remaining_ops, 1.0)
    intensity = remaining_works / safe_ops

    # Z-Score Normalization
    def _z_norm(x):
        if len(x) == 0:
            return np.array([])
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-12:
            return np.zeros_like(x)
        return (x - mean) / std

    z_intensity = _z_norm(intensity)
    z_ops = _z_norm(remaining_ops)
    z_min_dur = _z_norm(exact_min_durations)
    z_flex = _z_norm(n_machines)

    # --- Structural Change: Regret-Based Congestion Model ---
    # For each ready operation, compute a "congestion regret" score.
    # Regret = (work that must be done) * (urgency factor) * (congestion penalty on best machine)
    # This shifts from static weighted sum to a game-theoretic view:
    # prioritize ops that would cause the most future congestion if delayed.
    
    # Build machine load estimates from current ready set
    # machine_load[m] = sum of durations of ops that prefer machine m
    machine_load = {}
    for op in ready_operations:
        if not op["machine_options"]:
            continue
        best_m = min(op["machine_options"], key=lambda m: op["machine_options"][m])
        dur = op["machine_options"][best_m]
        machine_load[best_m] = machine_load.get(best_m, 0.0) + dur

    # Total ready work (for normalization)
    total_ready_work = float(np.sum(remaining_works)) if n > 0 else 1.0
    if total_ready_work < 1e-6:
        total_ready_work = 1.0

    # For each operation, compute regret score
    regret_scores = np.zeros(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        if not op["machine_options"]:
            regret_scores[i] = 0.0
            continue
        
        # Base urgency: high remaining work + high remaining ops = more to finish
        urgency = intensity[i] * (1.0 + 0.3 * (remaining_ops[i] / safe_ops[i]))
        
        # Find best machine and its congestion
        best_m = min(op["machine_options"], key=lambda m: op["machine_options"][m])
        best_dur = op["machine_options"][best_m]
        
        # Congestion penalty: how loaded is the best machine relative to total work?
        load_ratio = machine_load.get(best_m, 0.0) / total_ready_work
        congestion_penalty = 1.0 + 1.5 * load_ratio  # Higher when machine is busy
        
        # Flexibility discount: if many machines available, less regret from delay
        flex_discount = 1.0 / (1.0 + 0.2 * n_machines[i])
        
        # Duration factor: shorter ops have less impact but higher throughput value
        dur_factor = 1.0 / (1.0 + 0.1 * best_dur)
        
        regret_scores[i] = urgency * congestion_penalty * flex_discount * dur_factor

    # Combine regret score with z-normalized features
    # Regret score is the primary signal; features provide secondary refinement
    z_regret = _z_norm(regret_scores)
    
    # Time-aware weight: later in schedule, congestion matters more
    max_remaining = np.max(remaining_works) if len(remaining_works) > 0 else 1.0
    time_norm = min(time / max(max_remaining, 1e-6), 2.0)
    w_regret = 1.0 + 0.5 * time_norm
    w_intensity = 0.6
    w_ops = 0.3
    w_min_dur = -0.2
    w_flex = 0.1

    score = (w_regret * z_regret +
             w_intensity * z_intensity +
             w_ops * z_ops +
             w_min_dur * z_min_dur +
             w_flex * z_flex)

    # Handle NaNs or Infs
    score = np.nan_to_num(score, nan=0.0, posinf=1e9, neginf=-1e9)

    best_idx = int(np.argmax(score))
    
    # Select best machine for the chosen operation
    # Use congestion-aware machine selection: avoid machines with high estimated load
    op = ready_operations[best_idx]
    if op["machine_options"]:
        # Score each machine: prefer shorter duration, penalize high load
        def machine_score(m):
            dur = op["machine_options"][m]
            load = machine_load.get(m, 0.0)
            # Penalize both duration and congestion
            return dur * (1.0 + 0.3 * (load / total_ready_work))
        
        best_machine = min(op["machine_options"].keys(), key=machine_score)
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
