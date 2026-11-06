"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard
    _start = _time.perf_counter()
    _deadline = _start + 110.0

    best_score = float('inf')
    best_index = 0
    best_machine = None

    # Pre-calculate machine loads based on current ready operations
    # This helps estimate contention. If an op is ready, it's not on the machine yet.
    # We sum durations of all ops that *could* use the machine as a proxy for future contention.
    machine_potential_load = {}
    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        for m, dur in machine_options.items():
            if m not in machine_potential_load:
                machine_potential_load[m] = 0.0
            machine_potential_load[m] += dur

    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        
        if not machine_options:
            continue

        # Strategy: Critical Path Priority + SPT Machine Selection
        # 
        # 1. Operation Priority (Critical Path):
        #    Prioritize operations that are on the critical path of their job.
        #    Proxy: High remaining_work relative to remaining_ops (long average op) 
        #    or simply high remaining_work.
        #    We use a "urgency" score: remaining_work / max(1, remaining_ops).
        #    Higher urgency means this job has heavy work left, so we should start it.
        #
        # 2. Machine Selection (SPT / Least Load):
        #    Among eligible machines, pick the one that minimizes the projected finish time.
        #    Projected Finish = current_time + machine_potential_load[m] + duration.
        #    Since current_time is constant, we minimize (machine_potential_load[m] + duration).
        #    This is effectively SPT with a load adjustment.

        # Calculate the best machine for this operation
        best_m_for_op = None
        best_m_score = float('inf')
        
        for m, dur in machine_options.items():
            # Load proxy: How much work is queued/potential on this machine?
            # Note: In a true simulation, we'd know the exact queue. 
            # Here we use the sum of durations of all ready ops that can run on m.
            # This is an approximation of congestion.
            load = machine_potential_load.get(m, 0.0)
            
            # We want to minimize the time this op finishes.
            # Score for this machine choice:
            m_score = load + dur
            
            if m_score < best_m_score:
                best_m_score = m_score
                best_m_for_op = m

        if best_m_for_op is None:
            continue

        # Now score the operation itself based on its best possible machine
        # We want to pick the operation that, when started on its best machine, 
        # makes the most progress toward clearing the critical path.
        
        # Heuristic: 
        # 1. Minimize the "effective completion time" of this op (best_m_score).
        # 2. Tie-break/Weight by urgency: Prefer ops with more remaining work.
        
        # Let's combine:
        # Primary: Minimize best_m_score (start the thing that finishes soonest on the best machine)
        # Secondary: Maximize remaining_work (if finish times are similar, do the bigger job)
        
        # To make it a single minimization score:
        # score = best_m_score - alpha * remaining_work
        # If alpha is small, it's mostly SPT. If alpha is large, it's mostly urgency.
        # For makespan, clearing critical path (high remaining work) is often better than just SPT.
        # Let's try a weighted sum.
        
        # Normalize remaining_work? No, let's just use a small penalty for low remaining work.
        # Actually, standard WSPT (Weighted Shortest Processing Time) uses 1/p_i.
        # Here, "weight" could be related to criticality.
        
        # Let's try: Score = best_m_score * (1 + 0.1 * (1 / (remaining_work + 1)))
        # This slightly penalizes ops with low remaining work (less critical) if their finish time is same.
        # Or simpler: Score = best_m_score - 0.5 * remaining_work
        # Wait, if remaining_work is large, we subtract more, making score smaller (better).
        # This prioritizes high remaining work.
        
        # Let's use a ratio to balance units.
        # If remaining_work is ~100 and best_m_score is ~10, subtracting 0.5*100 = 50 dominates.
        # Maybe we should just prioritize Critical Path (High Remaining Work) first, then SPT.
        
        # Alternative: Sort by (remaining_work desc, best_m_score asc).
        # But we need a single scalar score for the loop.
        
        # Let's use: score = best_m_score - 0.2 * remaining_work
        # This favors ops that are both quick to finish on the best machine AND have significant work left.
        
        urgency_weight = 0.2
        score = best_m_score - urgency_weight * remaining_work

        if score < best_score:
            best_score = score
            best_index = i
            best_machine = best_m_for_op

        # Deadline check
        if i % 10 == 0 and _time.perf_counter() > _deadline:
            break

    # If no valid machine was found (should not happen), return first
    if best_machine is None:
        return 0

    # Return tuple to explicitly assign the machine
    # This allows the simulator to place the op on the specific machine we calculated as best,
    # rather than relying on a default or internal heuristic.
    return (best_index, best_machine)
# EVOLVE-BLOCK-END
