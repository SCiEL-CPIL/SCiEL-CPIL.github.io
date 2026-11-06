"""
Strategy: Lookahead Simulation with Regret/Waste Analysis
Instead of static graph metrics (CPL, Rank), we simulate the actual greedy packing
process. We run multiple simulations with slight perturbations to the priority order
to identify which tasks are "problematic" (create high waste or block better fits).
The final score is designed to prioritize tasks that, when placed early in a good
context, minimize total waste.

Key Insight: In ALB, the greedy algorithm picks the highest priority eligible task.
If a task has a high priority but creates a lot of unused space (waste) in the
current station that cannot be filled by other eligible tasks, it's a "bad" choice.
However, if it's the *only* large task that fits, it's good.
We estimate this by simulating and tracking the "regret" or "waste" associated with
each task's placement. We want to assign high priority to tasks that are "safe" to
place early (low waste creation) and low priority to tasks that are "risky" (high
waste creation) unless they are strictly necessary.

To make this robust, we run several simulations with different tie-breakers (random
perturbations) and aggregate the waste statistics.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict

def priority(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    predecessors = instance["predecessors"]
    cycle_time = instance.get("cycle_time", 40.0)
    
    # Build graph structures
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        preds = predecessors[i]
        if not isinstance(preds, list):
            preds = list(preds)
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1
            
    # Precompute static metrics for tie-breaking and initial order
    # Topological Sort
    queue = [i for i in range(n) if in_degree[i] == 0]
    topological_order = []
    temp_in_degree = in_degree[:]
    while queue:
        # Sort by in-degree or index for determinism in topological sort itself
        u = queue.pop(0)
        topological_order.append(u)
        for v in successors[u]:
            temp_in_degree[v] -= 1
            if temp_in_degree[v] == 0:
                queue.append(v)
                
    if len(topological_order) < n:
        # Cycle detected, fallback
        topological_order = list(range(n))

    # Calculate Rank from Start (RS) and Rank to End (RE) for a base priority
    rs = [0.0] * n
    for u in topological_order:
        max_pred_rs = 0.0
        for p in predecessors[u]:
            if p >= 0 and p < n: # Safety check
                max_pred_rs = max(max_pred_rs, rs[p])
        rs[u] = times[u] + max_pred_rs

    re = [0.0] * n
    for u in reversed(topological_order):
        max_succ_re = 0.0
        for s in successors[u]:
            max_succ_re = max(max_succ_re, re[s])
        re[u] = times[u] + max_succ_re

    # Base priority: Critical Path Length (RE) is a strong baseline
    base_priority = re.copy()
    
    # Simulation Function: Greedy Packing
    # Returns: (num_stations, waste_per_task)
    # waste_per_task[i] = sum of unused time in the station where task i was placed
    # Note: A task contributes to the waste of the station it is placed in.
    # Actually, waste is a property of the station. We want to attribute waste to tasks.
    # If a task is placed, and the station closes with remaining time R, 
    # the "waste" is R. We can distribute this waste among the tasks in that station
    # proportionally to their time, or just track it per task as the "slack" left after
    # placing it if it were the last one, or simply track the total waste of the station
    # and assign it to the task that "closed" the station or the one that caused the most
    # fragmentation.
    
    # Simpler attribution: 
    # For each station, total_time_used = sum(times of tasks in station).
    # waste = cycle_time - total_time_used.
    # We want to penalize tasks that are placed in stations with high waste.
    # To differentiate tasks within the same station, we can look at the order.
    # The last task placed often determines the "tightness". 
    # Let's track for each task: the waste of the station it was placed in.
    # If two tasks are in the same station, they get the same waste value.
    # To break ties, we can add a small penalty for tasks that are "large" in a "wasteful" station.
    
    def simulate_greedy(priorities, rng=None):
        """
        Runs greedy packing using given priorities.
        Returns total waste per task (list of length n).
        """
        if rng is None:
            rng = random.Random(42)
            
        # We need to simulate the state:
        # available_time per station
        # completed tasks set
        # eligible tasks set
        
        # For efficiency, we maintain a list of eligible tasks.
        # Since n is small (30), we can just scan all tasks to find eligible ones.
        
        completed = [False] * n
        eligible = []
        # Initialize eligible
        for i in range(n):
            if in_degree[i] == 0:
                eligible.append(i)
        
        # Current station state
        current_station_time = 0.0
        current_station_tasks = []
        total_stations = 0
        waste_per_task = [0.0] * n
        
        # We need a priority queue for eligible tasks? 
        # The problem statement says: "stations are filled with the highest-priority eligible task that fits the cycle time."
        # This implies a greedy selection: 
        # 1. Find all eligible tasks.
        # 2. Sort them by priority (descending).
        # 3. Pick the first one that fits in the remaining time of the current station.
        # 4. If none fit, close the station, open a new one, and pick the highest priority eligible task (which must fit as times[i] <= cycle_time is assumed).
        
        # To make this efficient, we can just sort the eligible list by priority each time we need to pick.
        # n=30, so sorting is cheap.
        
        while any(not c for c in completed):
            if not current_station_tasks:
                total_stations += 1
                current_station_time = 0.0
                current_station_tasks = []
                
            # Get eligible tasks that are not completed
            # Eligible list is maintained externally, but we need to filter by "not completed"
            # Actually, we update eligible list when tasks complete.
            
            # Sort eligible by priority descending
            # We need to handle ties? The prompt says "highest-priority". 
            # If priorities are unique, no tie. If not, we need a tie-breaker.
            # For simulation, we can use the provided priority list. 
            # If we want to simulate "random tie-breaking", we can perturb priorities.
            
            # Filter eligible to those not completed
            active_eligible = [t for t in eligible if not completed[t]]
            
            if not active_eligible:
                # This should not happen if graph is valid and we update eligible correctly
                break
                
            # Sort by priority descending
            # If we want to simulate different tie-breaks, we can add noise here.
            # But for the "waste" calculation, we want to see the "typical" behavior.
            # Let's use the base priority for the main simulation.
            
            active_eligible.sort(key=lambda x: -priorities[x])
            
            # Find the first task that fits
            placed_task = -1
            for t in active_eligible:
                if current_station_time + times[t] <= cycle_time:
                    placed_task = t
                    break
            
            if placed_task == -1:
                # No task fits in current station. Close it.
                # Calculate waste for this station
                waste = cycle_time - current_station_time
                # Attribute waste to tasks in this station
                # How to attribute? 
                # Option A: All tasks in station get `waste`.
                # Option B: The task that "caused" the closure? 
                # Let's use Option A for simplicity, but weighted by time?
                # If a task is large, it "uses up" the space. 
                # Let's just assign the station's waste to each task in it.
                for t in current_station_tasks:
                    waste_per_task[t] += waste
                    
                # Open new station
                current_station_time = 0.0
                current_station_tasks = []
                
                # Now, the highest priority eligible task MUST fit (assuming times[i] <= cycle_time)
                # So we pick the first one in the sorted list
                if active_eligible:
                    placed_task = active_eligible[0]
            else:
                # Task fits
                pass
            
            # Place the task
            if placed_task != -1:
                completed[placed_task] = True
                current_station_time += times[placed_task]
                current_station_tasks.append(placed_task)
                
                # Remove from eligible
                eligible.remove(placed_task)
                
                # Add successors to eligible if all predecessors are completed
                for s in successors[placed_task]:
                    # Check if all preds of s are completed
                    all_preds_done = True
                    for p in predecessors[s]:
                        if not completed[p]:
                            all_preds_done = False
                            break
                    if all_preds_done:
                        eligible.append(s)
                        
        # Close last station
        if current_station_tasks:
            waste = cycle_time - current_station_time
            for t in current_station_tasks:
                waste_per_task[t] += waste
            total_stations += 1
            
        return waste_per_task

    # We will run multiple simulations with slightly different priorities to get a robust estimate.
    # The idea: If a task consistently incurs high waste across different plausible orderings, 
    # it might be "bad" to prioritize it too high if it leads to fragmentation, 
    # OR it might be that it's just a large task.
    
    # Actually, a better metric: 
    # We want to assign HIGH priority to tasks that should be placed EARLY.
    # If a task is placed early, it affects many subsequent decisions.
    # If placing it early leads to LOW total waste, it's good.
    # If placing it early leads to HIGH total waste, it's bad.
    
    # Alternative approach: "Regret" of placing a task LATE.
    # If we force a task to be placed late (by lowering its priority), does the solution get worse?
    # This is expensive to test for each task.
    
    # Let's stick to the waste attribution.
    # We run K simulations.
    # In each simulation, we use a priority vector P_k.
    # We get waste_per_task_k.
    # We want to construct a final priority vector P_final.
    
    # Hypothesis: Tasks that are placed in "tight" stations (low waste) are "good" tasks to prioritize?
    # Not necessarily. 
    # Let's look at the "Criticality" from the simulation perspective.
    # If a task is a "bottleneck" (long path), it should be prioritized.
    # The simulation can reveal if the greedy algorithm is "struggling" with certain tasks.
    
    # Let's try a different angle: 
    # Use the simulation to estimate the "expected position" or "station index" of each task.
    # Tasks that are placed in earlier stations are "more critical" for the flow.
    # We can compute the average station index for each task across K simulations.
    # Then, set priority = - (avg_station_index) + base_criticality.
    # This encourages tasks that "naturally" fall early in good packings to be prioritized further.
    
    def simulate_with_station_tracking(priorities, rng=None):
        """
        Returns: (num_stations, station_index_per_task)
        """
        if rng is None:
            rng = random.Random(42)
            
        completed = [False] * n
        eligible = [i for i in range(n) if in_degree[i] == 0]
        
        current_station_time = 0.0
        current_station_tasks = []
        station_index_per_task = [0] * n
        current_station_idx = 0
        
        while any(not c for c in completed):
            if not current_station_tasks:
                current_station_idx += 1
                current_station_time = 0.0
                current_station_tasks = []
                
            active_eligible = [t for t in eligible if not completed[t]]
            if not active_eligible:
                break
                
            # Sort by priority
            active_eligible.sort(key=lambda x: -priorities[x])
            
            placed_task = -1
            for t in active_eligible:
                if current_station_time + times[t] <= cycle_time:
                    placed_task = t
                    break
            
            if placed_task == -1:
                # Close station
                current_station_time = 0.0
                current_station_tasks = []
                # Re-sort and pick first (must fit)
                if active_eligible:
                    placed_task = active_eligible[0]
            
            if placed_task != -1:
                completed[placed_task] = True
                station_index_per_task[placed_task] = current_station_idx
                current_station_time += times[placed_task]
                current_station_tasks.append(placed_task)
                
                eligible.remove(placed_task)
                
                for s in successors[placed_task]:
                    all_preds_done = True
                    for p in predecessors[s]:
                        if not completed[p]:
                            all_preds_done = False
                            break
                    if all_preds_done:
                        eligible.append(s)
                        
        return current_station_idx, station_index_per_task

    # Run K simulations
    K = 20  # Number of simulations
    avg_station_idx = np.zeros(n)
    total_stations_sum = 0
    
    # Use base priority (RE) for the first simulation
    # For subsequent simulations, add small random noise to break ties and explore variations
    for k in range(K):
        if time.time() > deadline:
            break
            
        if k == 0:
            p_sim = base_priority
        else:
            # Add noise proportional to the scale of priorities
            noise_scale = np.std(base_priority) * 0.1
            p_sim = base_priority + np.random.normal(0, noise_scale, n)
            
        try:
            num_stations, station_idx = simulate_with_station_tracking(p_sim)
            total_stations_sum += num_stations
            avg_station_idx += np.array(station_idx)
        except Exception:
            # Fallback if simulation fails
            break
            
    # Normalize
    if K > 0:
        avg_station_idx /= K
    else:
        # Fallback to static metrics
        avg_station_idx = np.zeros(n)
        # If simulation failed, use rank from start as proxy for "earliness"
        for i in range(n):
            avg_station_idx[i] = rs[i]
            
    # Combine:
    # We want high priority for:
    # 1. High Critical Path (RE) - Standard
    # 2. Low Average Station Index - Tasks that are placed early in good packings
    #    (This reinforces the "good" early placements)
    
    # Normalize station index to [0, 1]
    max_station = np.max(avg_station_idx) if np.max(avg_station_idx) > 0 else 1
    min_station = np.min(avg_station_idx) if np.min(avg_station_idx) < max_station else 0
    range_station = max_station - min_station
    if range_station == 0:
        norm_station = np.zeros(n)
    else:
        norm_station = (avg_station_idx - min_station) / range_station
        
    # Normalize RE to [0, 1]
    max_re = np.max(re) if np.max(re) > 0 else 1
    min_re = np.min(re) if np.min(re) < max_re else 0
    range_re = max_re - min_re
    if range_re == 0:
        norm_re = np.zeros(n)
    else:
        norm_re = (np.array(re) - min_re) / range_re
        
    # Weighted Sum
    # We trust the simulation's "earliness" signal strongly.
    # If a task is placed early in most good packings, it should be prioritized.
    # If a task is placed late, it might be less critical or dependent on others.
    
    # Score = 0.5 * norm_re + 0.5 * (1 - norm_station)
    # (1 - norm_station) means "early" is high score.
    
    scores = 0.5 * norm_re + 0.5 * (1.0 - norm_station)
    
    # Add a small duration component to break ties and favor larger tasks slightly if everything else is equal
    # (LPT is often good for ALB)
    max_time = np.max(times) if np.max(times) > 0 else 1
    norm_time = times / max_time
    
    scores = 0.8 * scores + 0.2 * norm_time
    
    # Ensure finite
    if not np.all(np.isfinite(scores)):
        scores = np.array(re, dtype=float)
        
    return scores

# EVOLVE-BLOCK-END
