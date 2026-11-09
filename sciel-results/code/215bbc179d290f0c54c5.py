"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Strategy: Stateless Critical Path Heuristic (TRPT/MRP)
# 
# The previous approach tracked machine availability statefully. 
# This approach uses a classic, robust heuristic for Job-Shop Scheduling:
# 1. Primary Key: Maximize Total Remaining Processing Time (TRPT) for the job.
#    - Rationale: Jobs with more work remaining are on the critical path. 
#      Prioritizing them prevents them from being delayed at the end of the schedule,
#      which is the primary driver of makespan in JSSP.
# 2. Secondary Key (Tie-breaker): Minimize the sum of durations across all available machines.
#    - Rationale: If two jobs have the same remaining work, pick the one that is 
#      "cheaper" to process on its available machines. This acts as a load-balancing 
#      proxy without needing stateful machine tracking.
# 3. Tertiary Key: Deterministic tie-breaker (Job ID, Op ID) to ensure stability.

_start_time = None

def dispatch(ready_operations, time):
    global _start_time
    if _start_time is None:
        _start_time = _time.time()
        
    # Deadline guard: if we are running too long, fall back to a very fast heuristic
    if _time.time() - _start_time > 100:
         # Fallback: Just pick the first one or shortest duration
         # In JSSP, if we run out of time, a simple SPT is better than crashing
         min_dur = float('inf')
         best_idx = 0
         for i, op in enumerate(ready_operations):
             durs = op["machine_options"].values()
             if durs:
                 min_d = min(durs)
                 if min_d < min_dur:
                     min_dur = min_d
                     best_idx = i
         return int(best_idx)

    n_ops = len(ready_operations)
    if n_ops == 0:
        return 0
        
    best_score = None
    best_idx = 0

    for i, op in enumerate(ready_operations):
        options = op["machine_options"]
        if not options:
            continue
            
        # 1. Primary: Total Remaining Processing Time (TRPT)
        # Higher is better (more urgent/critical)
        remaining_work = op.get("remaining_work", 0)
        
        # 2. Secondary: Sum of minimum durations? Or sum of all?
        # Let's use the sum of the minimum possible duration for this op across its options.
        # Actually, a better tie-breaker for "load" is the minimum duration available,
        # as it represents the "cost" of this specific operation.
        # However, to balance load, we might want to prefer ops that can run on less busy machines.
        # Since we are stateless, we don't know busyness.
        # Let's stick to a deterministic tie-breaker based on the operation's intrinsic properties.
        # Min duration is a good proxy for "efficiency".
        
        min_dur = min(options.values())
        
        # 3. Tertiary: Job ID and Op ID for determinism
        job_id = op.get("job", 0)
        op_id = op.get("op", 0)

        # We want to MAXIMIZE remaining_work.
        # If remaining_work is tied, we want to MINIMIZE min_dur (do quick things first if work is equal? 
        # Or does it matter? Usually, in JSSP, if work is equal, it's a tie. 
        # Let's try minimizing min_dur to clear the board faster for equal-work jobs).
        
        # Score tuple: (remaining_work, -min_dur, -job_id, -op_id)
        # We use negative values for min_dur and IDs so that max() works for "minimize" logic
        score = (remaining_work, -min_dur, -job_id, -op_id)

        if best_score is None or score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)

# EVOLVE-BLOCK-END
