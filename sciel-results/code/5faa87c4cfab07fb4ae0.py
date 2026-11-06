"""Open-shop dispatch rule: Fast multi-criteria heuristic with time guard."""
# EVOLVE-BLOCK-START
import time

_START_TIME = time.time()
# Safety margin: evaluator kills at 150s. We stop heavy logic if close to limit.
# Since dispatch is called many times, we keep it very fast (O(n log n) or O(n)).
_DEADLINE = _START_TIME + 145.0

def _time_left():
    return _DEADLINE - time.time()

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # If we are running out of time, use the absolute fastest fallback (arbitrary pick)
    if _time_left() < 0.05:
        return 0

    n = len(ready_operations)
    
    # If there's only one option, return it
    if n == 1:
        op = ready_operations[0]
        # Pick shortest duration machine for the single op
        best_m = min(op["machine_options"].items(), key=lambda x: x[1])[0]
        return (0, best_m)

    # Heuristic Strategy:
    # We want to minimize makespan.
    # Common strong heuristics for Open Shop:
    # 1. Most Work Remaining (MWR): Prioritize jobs with the most total work left.
    # 2. Shortest Processing Time (SPT): When choosing a machine, pick the one with the shortest duration for that op.
    # 3. Most Operations Remaining (MOR): Prioritize jobs with many steps left (to keep pipeline full).
    
    # The seed (score 1.0021) used MWR + SPT.
    # Let's try a slightly more nuanced scoring that balances "urgency" of the job (remaining work) 
    # with the "efficiency" of the immediate step (duration).
    
    # Alternative: "Weighted Shortest Processing Time" or "Most Critical Operation".
    # Let's define a score for each (op, machine) pair.
    # We want to select the pair that leads to the best makespan.
    
    # A robust approach often used in literature:
    # Score = alpha * remaining_work + beta * remaining_ops - gamma * min_duration
    # We maximize this score.
    
    # Let's try to balance:
    # - High remaining work is good (keeps critical path long? or rather, ensures we don't starve the critical job).
    # - Short duration is good (clears the board quickly).
    
    # However, simply maximizing remaining work (MWR) is a very standard and strong baseline.
    # The improvement in the seed came from explicitly picking the SPT machine.
    
    # Let's try a "Hybrid" score:
    # For each op, calculate:
    #   rw = remaining_work
    #   ro = remaining_ops
    #   min_d = min duration over machines
    #   avg_d = avg duration over machines (optional)
    
    # Let's try to prioritize "Most Work Remaining" but break ties more intelligently.
    # Or, use a combined metric:
    # Metric = rw / (1 + min_d) ? No, scales are different.
    
    # Let's stick to a lexicographic sort but with a small adjustment.
    # 1. Maximize remaining_work.
    # 2. Minimize min_duration.
    # 3. Maximize remaining_ops (tie-breaker).
    
    # Is there a better heuristic?
    # "Longest Remaining Processing Time" (LRPT) on the job level is MWR.
    # "Shortest Remaining Processing Time" (SRPT) on the operation level (SPT machine).
    
    # Let's try a weighted sum to see if it outperforms lexicographic.
    # If we weight remaining_work heavily, it behaves like MWR.
    # If we weight duration heavily, it behaves like SPT.
    
    # Let's try: Score = remaining_work * 10 - min_duration
    # This effectively says: "One unit of remaining work is worth 10 units of time saved by picking a faster machine."
    # This might be too aggressive on duration.
    
    # Let's try: Score = remaining_work - 0.1 * min_duration
    # This prioritizes work, but slightly prefers faster machines.
    
    # Actually, let's look at the "Critical Path" concept.
    # In open shop, the makespan is at least the max over jobs of (sum of durations) and max over machines of (sum of durations).
    # A good heuristic tries to balance these.
    
    # Let's implement a few candidate heuristics and pick the one that seems most robust.
    # Since we can't simulate, we rely on static features.
    
    # Heuristic A (Seed-like): Max rw, then Min min_d.
    # Heuristic B: Max (rw / min_d) ? (Efficiency)
    # Heuristic C: Max rw, then Max ro, then Min min_d.
    
    # Let's try Heuristic C first, as it's a refinement of A.
    # Also, let's consider that if remaining_work is very high, we should definitely pick it.
    
    # Let's compute features for all ops.
    candidates = []
    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        ro = op["remaining_ops"]
        
        # Find best machine (SPT)
        min_d = float('inf')
        best_m = None
        for m, d in op["machine_options"].items():
            if d < min_d:
                min_d = d
                best_m = m
        
        # If min_d is inf (empty machine_options?), skip or handle.
        if min_d == float('inf'):
            # Fallback: pick first machine or 0
            if op["machine_options"]:
                best_m = list(op["machine_options"].keys())[0]
                min_d = op["machine_options"][best_m]
            else:
                best_m = 0
                min_d = 0
        
        # Score for sorting:
        # We want to maximize rw.
        # Secondary: maximize ro (more steps left means more potential to interfere/benefit).
        # Tertiary: minimize min_d (faster completion of this step).
        
        # Sort key: (-rw, -ro, min_d)
        # Wait, if we maximize ro, we prioritize jobs with many steps.
        # If we minimize min_d, we prioritize fast steps.
        
        # Let's try a different combination:
        # Primary: Max rw
        # Secondary: Min min_d
        # This is the seed.
        
        # Let's try: Max (rw - 0.5 * min_d)
        # This blends the two.
        
        # Let's try a portfolio approach:
        # Calculate a score for each op using a few weights, and pick the best?
        # No, that's too complex and might not be better.
        
        # Let's stick to a refined lexicographic order.
        # Is "Max remaining_ops" a better tie-breaker than "Min duration"?
        # If two jobs have same remaining work, the one with more ops might be more "complex" and need attention?
        # Or the one with fewer ops is closer to done?
        
        # Let's try: (-rw, min_d, -ro)
        # 1. Most work.
        # 2. Fastest step.
        # 3. Most ops remaining (tie-breaker).
        
        candidates.append((-rw, min_d, -ro, i, best_m))
        
    # Sort
    candidates.sort()
    
    # Pick the first
    _, _, _, best_idx, best_machine = candidates[0]
    
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
