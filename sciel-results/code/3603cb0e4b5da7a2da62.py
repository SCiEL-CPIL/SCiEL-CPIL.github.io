"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def priority(instance):
    """
    Compute dynamic resource-aware priority scores.
    
    Strategy:
    1. Compute static structural priorities (Bottom Level / Longest Path to End).
    2. If time permits, perform a limited simulation-based refinement:
       - Simulate the serial SGS using the current static priority.
       - Identify "bottleneck" activities that cause resource conflicts or delay successors.
       - Adjust priorities to favor activities that are critical AND have low resource contention, 
         or penalize those that cause high contention.
    
    However, since the serial SGS picks the *highest* priority eligible activity at each step,
    and we only return a static vector, we must estimate a priority vector that, when used by SGS, 
    yields a good schedule.
    
    A more robust approach for a "different strategy" that fits the "dynamic" inspiration:
    We simulate the schedule generation process. At each step, we have a set of eligible activities.
    The SGS picks the one with the highest priority. 
    To optimize this, we can use a "look-ahead" heuristic: 
    For each eligible activity, estimate the resulting makespan if it were chosen now, using a greedy completion.
    This is too expensive for 30 activities * 30 steps * 5 instances if done naively.
    
    Instead, we will use a **Resource-Constrained Critical Path Method (RCCPM)** heuristic.
    1. Compute the static Critical Path (CP) length (Bottom Level).
    2. Compute the "Resource Pressure" for each activity: how much it consumes of the most constrained resource.
    3. The priority is a combination of:
       - High Bottom Level (criticality)
       - Low Resource Pressure (to avoid blocking others)
       - But weighted such that critical path activities are prioritized unless they are extremely resource-heavy compared to alternatives.
    
    Actually, the inspirations suggest that *simulating* the SGS is key.
    Let's implement a **Greedy Look-Ahead** strategy:
    1. Start with static Bottom Level priorities.
    2. Simulate the SGS step-by-step.
    3. At each step, for the set of eligible activities, instead of just picking the max static priority, 
       we can't change the static vector mid-run because the interface requires returning a single vector.
       
    Wait, the interface is `priority(instance) -> scores`. The SGS uses these scores.
    So the SGS is fixed: at time t, among eligible activities, pick argmax(score[i]).
    
    To improve on static Bottom Level, we need a score vector that implicitly accounts for resources.
    
    New Strategy: **Min-Max Regret / Conflict-Aware Priority**
    1. Calculate static Bottom Level (BL).
    2. Calculate "Resource Intensity" (RI) = sum(demands) / duration.
    3. Calculate "Conflict Score": For each activity, how many other activities have high BL and high RI?
    
    Let's try a **Simulated Annealing-like** or **Greedy Simulation** approach where we *estimate* the priority vector.
    Actually, a simpler structural change: **Dynamic Resource Leveling Priority**.
    
    We will compute the priority as:
    P[i] = BL[i] - lambda * (Average Resource Contention at BL[i]'s level)
    
    But we don't know the schedule yet.
    
    Let's go with **Inspiration 3's approach**: Dynamic Resource-Constrained Heuristic.
    We will simulate the SGS once with static BL priorities.
    During this simulation, we record the "delay" caused by resource constraints for each activity.
    Activities that caused delays (or were delayed by others) get adjusted priorities.
    Specifically, if an activity A is on the critical path but had to wait for resources, it should have had higher priority.
    If an activity B is not on the critical path but consumed resources that delayed A, B should have lower priority.
    
    Algorithm:
    1. Compute static BL.
    2. Simulate SGS using BL.
    3. Track resource usage over time.
    4. For each activity, calculate a "penalty" based on how much its resource consumption conflicted with higher-BL activities.
    5. Return BL - penalty.
    """
    
    start_time = time.time()
    time_limit = 120.0  # Safety margin for the evaluator
    # We have 120s, but we should aim to finish in < 10s to be safe.
    
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance.get("predecessors", [[] for _ in range(n)])
    demands = instance.get("demands", [])
    capacities = instance.get("capacities", [])
    
    if not demands or not capacities:
        # Fallback to static if resources not defined
        return _compute_bottom_level(n, durations, predecessors)
    
    # 1. Compute Static Bottom Level (BL)
    bl = _compute_bottom_level(n, durations, predecessors)
    
    # 2. Simulate SGS with static BL to identify conflicts
    # We need to track:
    # - Start times
    # - Resource usage over time
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # In-degree for topological check and eligibility
    in_degree = [0] * n
    for i in range(n):
        for pred in predecessors[i]:
            in_degree[i] += 1
    
    # Simulate SGS
    # State:
    # - remaining_activities: set of not-yet-scheduled activities
    # - current_time: 0
    # - resource_usage: dict of time -> resource usage? Or just track active activities.
    
    # For 30 activities, we can simulate efficiently.
    # We need to know when resources free up.
    
    start_times = [None] * n
    finish_times = [None] * n
    active = []  # List of (finish_time, activity_id)
    scheduled = [False] * n
    current_time = 0
    
    # To efficiently find the next event time, we can just iterate.
    # Since n=30, a simple loop is fine.
    
    # We need to check eligibility: all predecessors finished.
    # And resource feasibility: sum of demands of active + new <= capacity.
    
    # Let's use a list for active activities to check resources.
    # active_list: list of (finish_time, id, demands_vector)
    
    # We need to handle the "eligible" set dynamically.
    # An activity is eligible if all predecessors are finished (start_time is not None and finish_time <= current_time? No, finish_time >= current_time implies it's done? No, finish_time is when it ends).
    # Predecessor i is done if finish_times[i] is not None and finish_times[i] <= current_time.
    
    # Actually, in SGS, we start an activity at current_time.
    # So predecessor must have finish_time <= current_time.
    
    # Let's run the simulation.
    while any(not scheduled[i] for i in range(n)):
        # Find eligible activities
        eligible = []
        for i in range(n):
            if scheduled[i]:
                continue
            # Check predecessors
            is_eligible = True
            for pred in predecessors[i]:
                if finish_times[pred] is None or finish_times[pred] > current_time:
                    is_eligible = False
                    break
            if is_eligible:
                eligible.append(i)
        
        if not eligible:
            # No eligible activities? This implies a cycle or error, or we need to advance time.
            # In SGS, if no activity is eligible, we advance time to the next finish time.
            next_finish = min([ft for ft in finish_times if ft is not None])
            if next_finish <= current_time:
                # Should not happen if we always advance to next finish when stuck
                current_time += 1
            else:
                current_time = next_finish
            continue
        
        # Sort eligible by BL (highest first)
        eligible.sort(key=lambda x: -bl[x])
        
        # Find the first activity that fits in resources
        chosen = -1
        for i in eligible:
            # Check resources
            feasible = True
            for r in range(len(capacities)):
                # Sum demands of active activities for resource r
                usage = sum(a[2][r] for a in active)
                if usage + demands[i][r] > capacities[r]:
                    feasible = False
                    break
            if feasible:
                chosen = i
                break
        
        if chosen == -1:
            # None of the eligible activities fit.
            # Advance time to the next finish time.
            next_finish = min([ft for ft in finish_times if ft is not None])
            if next_finish <= current_time:
                # Edge case: zero duration or simultaneous finish/start
                current_time += 1
            else:
                current_time = next_finish
            # Remove finished activities from active
            active = [a for a in active if a[0] > current_time]
            continue
        
        # Schedule chosen
        scheduled[chosen] = True
        start_times[chosen] = current_time
        finish_times[chosen] = current_time + durations[chosen]
        active.append((finish_times[chosen], chosen, demands[chosen]))
        
        # We don't advance time here; we try to schedule more activities at the same current_time
        # The loop will continue, and eligible will be updated (predecessors of chosen are now finished? No, chosen is finished at future time).
        # Wait, if we schedule an activity, it doesn't become "finished" until its finish time.
        # So its successors are not eligible yet.
        # So we just try to find another eligible activity at the same current_time.
        
        # If no more can be scheduled at current_time, the loop will find no eligible (or none feasible) and advance time.
        
    # Simulation complete.
    # Now, analyze the schedule to adjust priorities.
    # We want to penalize activities that caused resource conflicts for high-BL activities.
    
    # For each activity j, calculate a "conflict score".
    # If activity j was active at time t, and activity i (with high BL) was eligible but couldn't start due to resource conflict with j, then j should be penalized.
    
    # This requires re-simulating or tracking "blocked" events.
    # Let's re-simulate and track blocked activities.
    
    # Reset simulation state
    start_times = [None] * n
    finish_times = [None] * n
    scheduled = [False] * n
    current_time = 0
    active = []
    
    # conflict_score[i] = sum of BL[j] for all j that were blocked by i
    conflict_score = np.zeros(n)
    
    while any(not scheduled[i] for i in range(n)):
        eligible = []
        for i in range(n):
            if scheduled[i]:
                continue
            is_eligible = True
            for pred in predecessors[i]:
                if finish_times[pred] is None or finish_times[pred] > current_time:
                    is_eligible = False
                    break
            if is_eligible:
                eligible.append(i)
        
        if not eligible:
            next_finish = min([ft for ft in finish_times if ft is not None])
            if next_finish <= current_time:
                current_time += 1
            else:
                current_time = next_finish
            active = [a for a in active if a[0] > current_time]
            continue
        
        eligible.sort(key=lambda x: -bl[x])
        
        chosen = -1
        for i in eligible:
            feasible = True
            for r in range(len(capacities)):
                usage = sum(a[2][r] for a in active)
                if usage + demands[i][r] > capacities[r]:
                    feasible = False
                    break
            if feasible:
                chosen = i
                break
        
        if chosen == -1:
            # None fit. Identify which active activities are causing the block.
            # For each active activity a, check if removing it would allow any eligible activity to fit.
            # This is expensive. Simpler: penalize all active activities that are "heavy".
            # Or, more precisely, for each eligible activity e that didn't fit, find which resource is the bottleneck.
            # Then penalize active activities contributing to that resource.
            
            for e in eligible:
                # Find bottleneck resource
                for r in range(len(capacities)):
                    usage = sum(a[2][r] for a in active)
                    if usage + demands[e][r] > capacities[r]:
                        # Resource r is bottleneck.
                        # Penalize active activities that use resource r.
                        # Weight penalty by BL[e] (importance of blocked activity)
                        for a in active:
                            a_id = a[1]
                            if a[2][r] > 0:
                                conflict_score[a_id] += bl[e]
                        break # Only consider first bottleneck resource
            
            next_finish = min([ft for ft in finish_times if ft is not None])
            if next_finish <= current_time:
                current_time += 1
            else:
                current_time = next_finish
            active = [a for a in active if a[0] > current_time]
            continue
        
        # Schedule chosen
        scheduled[chosen] = True
        start_times[chosen] = current_time
        finish_times[chosen] = current_time + durations[chosen]
        active.append((finish_times[chosen], chosen, demands[chosen]))
        
    # Normalize conflict scores
    # Priority = BL - alpha * conflict_score
    # We need to choose alpha.
    # If conflict_score is 0, priority is BL.
    # If conflict_score is high, priority is reduced.
    
    # To ensure stability, we normalize conflict_score by max BL.
    max_bl = np.max(bl)
    if max_bl > 0:
        normalized_conflict = conflict_score / max_bl
    else:
        normalized_conflict = conflict_score
    
    # We want to keep BL dominant but adjust for conflicts.
    # Let's try: Priority = BL * (1 - 0.5 * normalized_conflict / (1 + normalized_conflict))
    # Or simply: Priority = BL - 0.1 * conflict_score
    
    # Let's use a soft penalty to avoid completely inverting the order.
    # Priority = BL - 0.2 * conflict_score
    
    # However, if an activity has high BL but caused huge conflicts, it might drop below non-critical activities.
    # This is the goal: to schedule non-critical, low-conflict activities first to free resources.
    
    # Let's add a time guard.
    if time.time() - start_time > time_limit * 0.5:
        return bl
    
    return bl - 0.2 * conflict_score


def _compute_bottom_level(n, durations, predecessors):
    """Compute longest path from node to end."""
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    in_degree = [0] * n
    for i in range(n):
        for pred in predecessors[i]:
            in_degree[i] += 1
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    if len(topo_order) != n:
        return durations.copy()
    
    longest_path = np.zeros(n)
    for node in reversed(topo_order):
        if not successors[node]:
            longest_path[node] = durations[node]
        else:
            max_succ_path = max(longest_path[s] for s in successors[node])
            longest_path[node] = durations[node] + max_succ_path
    
    return longest_path
# EVOLVE-BLOCK-END
