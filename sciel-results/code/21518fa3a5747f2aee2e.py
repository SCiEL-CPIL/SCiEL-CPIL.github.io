"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = _time.monotonic() + 115.0

def _check_deadline():
    return _time.monotonic() >= _DEADLINE

def dispatch(ready_operations, time):
    if _check_deadline():
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Pre-extract data to minimize overhead inside the loop
    # We need to simulate the "critical path" impact of starting each op.
    # Strategy: For each candidate, estimate the completion time of the job
    # and the machine, and pick the one that minimizes the maximum of these
    # (a proxy for makespan reduction or bottleneck avoidance).
    
    best_idx = 0
    best_score = float('inf')

    # We will compute a "criticality" score for each operation.
    # A lower score is better.
    # Score = max(job_completion_estimate, machine_completion_estimate)
    # But we must account for the fact that starting an op NOW removes it from
    # the "ready" pool, potentially unblocking the next op in the same job.
    
    # To be fast, we approximate:
    # 1. Job Criticality: Sum of remaining durations in the job (after this op).
    #    If we start this op, the job's remaining work is reduced by this op's duration.
    #    The "time to finish job" if we start now is roughly:
    #    current_time + duration_this_op + sum(remaining_durations_after_this_op)
    #    However, remaining_ops and remaining_work are given.
    #    remaining_work usually includes the current op? Let's assume standard definitions:
    #    remaining_work = sum of durations of all ops in the job that are NOT yet completed.
    #    If the current op is "ready", it is not yet completed. So remaining_work includes it.
    #    If we start it, the *future* work for the job is remaining_work - duration.
    
    # 2. Machine Criticality: How long will this machine be busy?
    #    duration_this_op.
    
    # Let's refine the proxy:
    # We want to avoid situations where a long job is stuck because its machine is busy,
    # or a machine is idle while a long job waits.
    
    # Heuristic: Minimize (Job_Remaining_Work + Machine_Duration)
    # Actually, a better proxy for makespan in JSS is often "Longest Remaining Processing Time" (LRPT)
    # but applied to the *critical path*.
    
    # Let's try a simple Lookahead:
    # For each candidate i:
    #   job_id = ready_operations[i]['job']
    #   mach_id, dur = min(ready_operations[i]['machine_options'].items(), key=lambda x: x[1])
    #   
    #   # Estimate 1: When will this job finish?
    #   # It needs to finish this op (dur), then the rest (remaining_work - dur).
    #   # But the rest might be on other machines.
    #   # Approximation: job_finish_time = time + dur + (remaining_work - dur) = time + remaining_work
    #   # This is constant for all ops of the same job. So it doesn't help distinguish ops of the same job.
    
    #   # Estimate 2: When will the machine be free?
    #   # machine_free_time = time + dur
    
    #   # If we pick an op, we satisfy the job's next step and the machine's next step.
    #   # The "bottleneck" is the max of the critical path lengths.
    
    # Let's use a different angle: **Machine Load Balancing + Job Urgency**
    # But the current program already does a version of this.
    
    # Let's try **Simulation of the next event**:
    # If we start op i at time t, it finishes at t + d_i.
    # At that time, the next op of job j becomes ready.
    # The next op of job j has a duration d_next.
    # If d_next is large, and the machine for d_next is likely busy, it's bad.
    
    # Since we don't have future machine availability, we assume worst-case or average.
    
    # Let's try a specific structural change: **Shortest Processing Time (SPT) with Job Priority**
    # Usually SPT is good for minimizing mean flow time, but for makespan, Longest Processing Time (LPT)
    # or Critical Ratio is often better.
    
    # Let's try **Critical Ratio (CR)**:
    # CR = (Time to Due Date) / (Remaining Work). 
    # We don't have due dates.
    
    # Let's try **Weighted Shortest Processing Time (WSPT)** where weight is inverse remaining work.
    
    # Alternative Strategy: **Greedy Critical Path**
    # Calculate the "depth" of the job (number of remaining ops) and "width" (work).
    # Prefer ops that are on the critical path of the job (i.e., high remaining work) AND 
    # are short (to keep the machine free for others).
    
    # This is similar to the current one. Let's look at the inspirations.
    # Inspiration 1: "Critical Path Reduction" - simulates immediate completion.
    
    # Let's implement a lightweight version of Inspiration 1:
    # For each candidate, assume it completes.
    # 1. The job's remaining work decreases by duration.
    # 2. The job's remaining ops decrease by 1.
    # 3. The machine is occupied for 'duration'.
    
    # Score = max( 
    #     job_critical_path_estimate, 
    #     machine_load_estimate 
    # )
    
    # How to estimate job_critical_path_estimate?
    # It's the sum of durations of the remaining ops in the job.
    # Since we don't have the durations of the *future* ops, only remaining_work,
    # we can use remaining_work as a proxy for the job's critical path length.
    
    # So, if we pick op i:
    # new_job_work = remaining_work[i] - duration[i]
    # machine_busy_until = time + duration[i]
    
    # This doesn't seem to distinguish well enough if remaining_work is the dominant term.
    
    # Let's try a different metric: **Idle Time Minimization**
    # If a machine is free now, we should fill it.
    # Which op to pick?
    # If we pick a long op, the machine is busy longer.
    # If we pick a short op, the machine becomes free sooner, potentially allowing another op.
    # So, **Shortest Processing Time (SPT)** is a strong candidate for makespan in JSS?
    # Actually, research suggests SPT is good for flow time, but for makespan, 
    # **Longest Processing Time (LPT)** or **Most Work Remaining (MWR)** is often better.
    
    # The current program uses a mix: Urgency (MWR) * Blocking (Inverse SPT).
    # This effectively prefers "Urgent but Short" ops.
    
    # Let's try the opposite: **Urgent and Long**? No, that blocks the machine.
    
    # Let's try **Pure MWR (Most Work Remaining)**:
    # Always pick the op belonging to the job with the most remaining work.
    # Tie-breaker: Shortest duration on the chosen machine.
    
    # Let's try **Pure SPT (Shortest Processing Time)**:
    # Always pick the op with the shortest duration.
    # Tie-breaker: Most work remaining.
    
    # Let's try **Hybrid: Minimize (Remaining Work / Duration)**
    # This is like "Work per unit time". High value = urgent.
    
    # Let's implement a **Local Search** on the dispatch decision? No, too slow.
    
    # Let's try **Simulation-based Proxy**:
    # For each candidate, calculate a "regret" score.
    # Regret = (Makespan if we pick this) - (Makespan if we pick best alternative)
    # We can't calculate makespan.
    
    # Let's go with **Dynamic Priority: Inverse Remaining Work * Duration**
    # Wait, the current one is: Urgency (RW^2) * (1/(1+Dur)).
    # This favors High RW and Low Dur.
    
    # What if we favor **High RW** and **High Dur**? 
    # No, high dur blocks.
    
    # What if we favor **Low RW** and **Low Dur**? (SPT + Low Urgency) -> Bad for makespan.
    
    # Let's try **Most Operations Remaining (MOR)**:
    # Jobs with more remaining ops are "deeper" in the pipeline.
    
    # Let's try a completely different structural approach:
    # **Round-Robin with Priority**
    # Track the last job served? We don't have state.
    
    # Okay, let's look at the "Inspiration 1" again: "Critical Path Reduction".
    # It simulates the immediate completion.
    # Let's implement a fast approximation of this.
    
    # For each candidate i:
    #   1. Identify the job J and machine M.
    #   2. Assume op i completes.
    #   3. The next op in J is now ready. Let's call its estimated duration D_next.
    #      We don't know D_next. But we know remaining_ops[i] - 1 is the count.
    #      If remaining_ops[i] - 1 > 0, there is more work.
    #   4. The machine M is free at time + D_i.
    
    # Let's use a simple heuristic that is structurally different:
    # **Minimize the Maximum of (Job Remaining Work, Machine Duration)**
    # This balances the two constraints.
    
    best_idx = 0
    best_score = float('inf')
    
    # Extract features once
    rw = np.empty(n)
    dur = np.empty(n)
    ro = np.empty(n)
    
    for i, op in enumerate(ready_operations):
        rw[i] = op['remaining_work']
        ro[i] = op['remaining_ops']
        # Pick the machine with min duration? Or is the machine fixed?
        # "machine_options {machine: duration}" implies we can choose the machine?
        # "fixed operation order" but "machine_options" usually means the op can be done on any of these machines.
        # If we can choose the machine, we should pick the one with the shortest duration to minimize blocking.
        # So, effective duration = min(machine_options.values())
        opts = op['machine_options']
        dur[i] = min(opts.values()) if opts else 0.0

    # Normalize features for comparison
    # We want to minimize a score.
    # Score = max(rw[i], dur[i]) ?
    # Or Score = rw[i] + dur[i] ?
    
    # Let's try **Score = rw[i] * (1 + dur[i])**
    # This penalizes both high work and long duration.
    
    # Or **Score = rw[i] / (1 + dur[i])** -> Maximize this?
    # High work, short duration = High Score.
    
    # Let's try **Maximize: (rw[i] ** 1.5) / (dur[i] + 1)**
    # This is a different non-linearity than the current one.
    
    # Current: (rw/rw_max)^2 * (1/(1+dur/dur_max))
    
    # Let's try **Minimize: (rw[i] + dur[i]) / ro[i]**
    # Work per remaining op?
    
    # Let's try a **Greedy Lookahead** that is very fast:
    # 1. Sort candidates by Duration (SPT).
    # 2. Among the top 3 shortest, pick the one with the highest Remaining Work.
    
    # This is structurally different from a continuous score. It's a discrete selection.
    
    if n <= 3:
        # Just pick the one with max remaining work
        idxs = np.argsort(-rw)
        return int(idxs[0])
    
    # Get indices of the 3 shortest durations
    # Break ties by remaining work
    # We can do this by sorting by (duration, -remaining_work)
    # Actually, let's just find the 3 smallest durations.
    
    # To be efficient, we can use a partial sort or just argsort if n is small (usually < 50)
    # n is number of ready ops. In JSS, this can be up to J*M, but usually small.
    
    # Let's try: Pick the op that minimizes (Duration + Remaining_Work / 10)
    # This is a weighted sum, different from the product.
    
    # Let's try **Minimize Duration, Tie-break by Max Remaining Work**
    # This is pure SPT with urgency tie-break.
    
    # Implementation:
    # 1. Find min duration.
    # 2. Filter candidates with duration == min_duration (or within epsilon).
    # 3. Pick the one with max remaining_work from that set.
    
    min_dur = np.min(dur)
    # Allow a small tolerance to handle floating point or multiple machines with same min
    candidates = np.where(dur <= min_dur + 1e-6)[0]
    
    if len(candidates) == 0:
        return 0
        
    if len(candidates) == 1:
        return int(candidates[0])
        
    # Among the shortest ops, pick the one for the most urgent job
    candidate_rws = rw[candidates]
    best_local = np.argmax(candidate_rws)
    best_idx = int(candidates[best_local])
    
    return best_idx

# EVOLVE-BLOCK-END
