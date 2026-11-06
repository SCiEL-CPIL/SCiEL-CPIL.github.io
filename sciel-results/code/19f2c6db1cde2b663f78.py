"""Job-Shop Scheduling Dispatch Rule"""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Uses a combination of:
    - Critical Path (CP) - remaining work on the longest job chain
    - Weighted Shortest Processing Time (WSPT) - ratio of weight to processing time
    - Remaining operations count - prefer jobs with fewer remaining ops
    - Machine availability - prefer operations with shorter min duration
    """
    if not ready_operations:
        return 0
    
    # Deadline guard
    if _time.time() - _START_TIME > _TIME_LIMIT:
        return 0
    
    n = len(ready_operations)
    
    if n == 1:
        return 0
    
    # Precompute features
    min_durations = np.array([min(op["machine_options"].values()) if op["machine_options"] else 1.0 for op in ready_operations], dtype=np.float64)
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # Machine contention: how many ready ops want each machine
    machine_demand = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_demand[m] = machine_demand.get(m, 0) + 1
    
    # Machine contention score: higher = more congested machines
    contention_scores = np.array([
        sum(machine_demand.get(m, 0) for m in op["machine_options"]) / max(len(op["machine_options"]), 1)
        for op in ready_operations
    ], dtype=np.float64)
    
    # Compute job-level remaining work (critical path proxy)
    job_remaining_work = {}
    for op in ready_operations:
        job = op["job"]
        if job not in job_remaining_work:
            job_remaining_work[job] = op["remaining_work"]
        else:
            job_remaining_work[job] = max(job_remaining_work[job], op["remaining_work"])
    
    # Critical path approximation: max remaining work across all jobs
    current_cp = max(job_remaining_work.values()) if job_remaining_work else 0
    
    # Normalize features
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val < 1e-12:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    # Feature 1: Critical Path Score (higher = more urgent)
    cp_scores = remaining_works / (current_cp + 1e-9)
    cp_norm = normalize(cp_scores)
    
    # Feature 2: WSPT-like score (shorter processing = better)
    spt_norm = 1.0 - normalize(min_durations)
    
    # Feature 3: Fewer remaining ops = better
    rem_ops_norm = 1.0 - normalize(remaining_ops)
    
    # Feature 4: Work density (remaining work / min duration) - higher density = more urgent
    work_density = remaining_works / (min_durations + 1e-9)
    density_norm = normalize(work_density)
    
    # Feature 5: Machine contention - avoid congested machines
    contention_norm = 1.0 - normalize(contention_scores)
    
    # Composite score
    # Weights: CP is most important, then density, then SPT, then contention, then remaining ops
    final_scores = (
        cp_norm * 0.30 +        # Critical path urgency
        density_norm * 0.25 +   # Work density
        spt_norm * 0.20 +       # Short processing time
        contention_norm * 0.15 + # Avoid machine congestion
        rem_ops_norm * 0.10     # Fewer remaining ops
    )
    
    # Tie-breaking: prefer shorter min_duration, then lower contention among close scores
    max_score = np.max(final_scores)
    close_to_max = np.where(final_scores > max_score - 1e-9)[0]
    if len(close_to_max) > 1:
        # First try to break ties by min_duration
        tied_durations = min_durations[close_to_max]
        min_dur_val = np.min(tied_durations)
        dur_close = close_to_max[tied_durations <= min_dur_val + 1e-9]
        
        if len(dur_close) > 1:
            # Break remaining ties by lower contention
            tied_contention = contention_scores[dur_close]
            min_cont_val = np.min(tied_contention)
            cont_close = dur_close[tied_contention <= min_cont_val + 1e-9]
            return int(cont_close[0])
        else:
            return int(dur_close[0])
    
    return int(np.argmax(final_scores))
# EVOLVE-BLOCK-END
