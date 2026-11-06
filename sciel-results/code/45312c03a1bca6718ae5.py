"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _simulate_sgs(instance, priority_scores):
    """
    Serial Schedule Generation Scheme.
    Returns the makespan.
    """
    N = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # (N, R)
    capacities = np.asarray(instance["capacities"], dtype=float)  # (R,)
    R = len(capacities)
    
    # Build predecessor and successor lists
    preds = instance.get("predecessors", [[] for _ in range(N)])
    # Normalize predecessors to list of lists
    preds_list = []
    for i in range(N):
        if isinstance(preds, list) and i < len(preds):
            p = preds[i]
            if isinstance(p, (list, tuple, np.ndarray)):
                preds_list.append(list(p))
            elif p is None:
                preds_list.append([])
            else:
                preds_list.append([p])
        else:
            preds_list.append([])
    
    # In-degree count for each activity
    in_degree = np.zeros(N, dtype=int)
    successors = [[] for _ in range(N)]
    for i in range(N):
        for p in preds_list[i]:
            in_degree[i] += 1
            successors[p].append(i)
    
    # State
    # start_times[i] = start time of activity i
    start_times = np.full(N, -1.0)
    # Resource usage over time is complex to track incrementally with varying start times.
    # Instead, we maintain a list of "events" or simply check resource availability at potential start times.
    # For N=30, we can afford to check all past activities for resource conflict.
    
    # To make resource checking efficient, we can maintain a list of (end_time, demand_vector) for active/finished jobs.
    # Or simply iterate through all scheduled jobs.
    # Given N=30, O(N^2) per step is fine.
    
    scheduled = [False] * N
    eligible = []
    
    # Initialize eligible set
    for i in range(N):
        if in_degree[i] == 0:
            eligible.append(i)
    
    # We need to pick the highest priority eligible activity that can be scheduled at the earliest feasible time.
    # Actually, Serial SGS typically:
    # 1. Find set of eligible activities.
    # 2. Pick the one with highest priority.
    # 3. Schedule it at the earliest time >= current_time such that resources are available.
    # 4. Update state.
    # 5. Repeat.
    
    # Wait, the standard Serial SGS (Klein et al.) works like this:
    # The schedule is built by inserting activities one by one in the order defined by the priority list.
    # However, the prompt says: "a serial schedule-generation scheme starts the highest-priority eligible activity under precedence and renewable-resource limits."
    # This implies a dynamic selection: at each step, look at all currently eligible (all predecessors scheduled) activities, pick the one with the highest priority score, and schedule it at the earliest feasible time.
    
    # Let's implement this dynamic selection.
    
    current_time = 0.0
    # To find earliest feasible time, we need to check resource constraints.
    # Let's keep a list of scheduled activities: (start, end, demand)
    scheduled_jobs = []
    
    for _ in range(N):
        if not eligible:
            # This shouldn't happen if the graph is acyclic
            break
        
        # Pick highest priority from eligible
        # priority_scores are such that higher is better?
        # The seed program returned durations, and "Longest duration first" implies higher score = higher priority.
        # So we pick max priority.
        max_prio = -np.inf
        best_act = -1
        for i in eligible:
            if priority_scores[i] > max_prio:
                max_prio = priority_scores[i]
                best_act = i
        
        # Schedule best_act
        # Find earliest start time t >= current_time such that for all resources r,
        # sum of demands of jobs active at t + demand[best_act, r] <= capacity[r]
        
        # Candidate start times: current_time, and end times of all scheduled jobs that are >= current_time.
        # Why? Because resource usage is piecewise constant, changing only at job end times.
        candidate_times = [current_time]
        for s, e, d in scheduled_jobs:
            if e >= current_time:
                candidate_times.append(e)
        
        # Sort and unique
        candidate_times = sorted(set(candidate_times))
        
        feasible_start = None
        for t in candidate_times:
            # Check resources at time t
            # We need to sum demands of all jobs j where s_j <= t < e_j
            usage = np.zeros(R)
            for s, e, d in scheduled_jobs:
                if s <= t < e:
                    usage += d
            if np.all(usage + demands[best_act] <= capacities):
                feasible_start = t
                break
        
        if feasible_start is None:
            # This should not happen if capacities are sufficient for single job
            # But just in case, push to infinity? Or handle error.
            # For robustness, let's assume it's feasible at some point.
            # Actually, if a single job doesn't fit, it's infeasible.
            # But the problem implies feasible instances.
            feasible_start = current_time # Fallback
            
        # Schedule it
        start_times[best_act] = feasible_start
        end_time = feasible_start + durations[best_act]
        scheduled_jobs.append((feasible_start, end_time, demands[best_act]))
        
        # Update eligible set
        eligible.remove(best_act)
        
        # Update successors
        for succ in successors[best_act]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                eligible.append(succ)
        
        # Update current_time?
        # In Serial SGS, "current_time" is often just the time when we last added a job?
        # No, the next job can start at any time >= 0.
        # The "current_time" in my loop above was used to restrict candidate times to >= current_time.
        # Actually, the earliest feasible time for the next job doesn't necessarily have to be >= the start time of the previous job.
        # It just needs to be >= 0 and satisfy resources.
        # However, since we are building a schedule, the "current_time" concept in some SGS variants refers to the time horizon being extended.
        # But in the "list scheduling" style SGS, we just find the earliest feasible time for the selected activity.
        # My candidate_times logic: I should consider all end times of scheduled jobs, and 0.
        # I don't need to restrict to >= previous start time.
        # Let's re-evaluate candidate times.
        # The resource profile changes only at start and end times of jobs.
        # Since we are adding a job, the relevant "change points" are the end times of existing jobs.
        # Also, the job could start at 0.
        # So candidate_times should be {0} U {e_j | j in scheduled_jobs}.
        
        # Let's refine the loop.
        # I will re-implement the scheduling part to be cleaner.
        
    # The loop above has a flaw: `candidate_times` is computed inside the loop but `scheduled_jobs` changes.
    # And I used `current_time` which I didn't update properly.
    # Let's rewrite the scheduling logic clearly.
    
    # Actually, I'll just return the max end time.
    if not scheduled_jobs:
        return 0.0
    makespan = max(e for s, e, d in scheduled_jobs)
    return makespan


def _simulate_sgs_correct(instance, priority_scores):
    """
    Correct Serial SGS implementation.
    """
    N = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # (N, R)
    capacities = np.asarray(instance["capacities"], dtype=float)  # (R,)
    R = len(capacities)
    
    preds = instance.get("predecessors", [[] for _ in range(N)])
    preds_list = []
    for i in range(N):
        if isinstance(preds, list) and i < len(preds):
            p = preds[i]
            if isinstance(p, (list, tuple, np.ndarray)):
                preds_list.append(list(p))
            elif p is None:
                preds_list.append([])
            else:
                preds_list.append([p])
        else:
            preds_list.append([])
    
    in_degree = np.zeros(N, dtype=int)
    successors = [[] for _ in range(N)]
    for i in range(N):
        for p in preds_list[i]:
            in_degree[i] += 1
            successors[p].append(i)
    
    scheduled_jobs = [] # List of (start, end, demand_vector)
    eligible = [i for i in range(N) if in_degree[i] == 0]
    
    for _ in range(N):
        if not eligible:
            break
        
        # Select highest priority
        best_act = max(eligible, key=lambda i: priority_scores[i])
        
        # Find earliest feasible start time
        # Candidate times: 0 and all end times of scheduled jobs
        candidate_times = [0.0]
        for s, e, d in scheduled_jobs:
            candidate_times.append(e)
        
        # Sort unique
        candidate_times = sorted(set(candidate_times))
        
        feasible_start = None
        for t in candidate_times:
            # Check resource availability at time t
            usage = np.zeros(R)
            for s, e, d in scheduled_jobs:
                if s <= t < e:
                    usage += d
            if np.all(usage + demands[best_act] <= capacities):
                feasible_start = t
                break
        
        if feasible_start is None:
            # Fallback: if no candidate time works, it might be due to floating point or infeasibility.
            # Try a slightly later time? Or assume infeasible.
            # For safety, let's just pick the last candidate time + epsilon?
            # Or better, if the job fits in capacity alone, it must fit eventually.
            # The candidate times cover all intervals where resource usage is constant.
            # If it doesn't fit in any interval, it's infeasible.
            feasible_start = candidate_times[-1] if candidate_times else 0.0
        
        end_time = feasible_start + durations[best_act]
        scheduled_jobs.append((feasible_start, end_time, demands[best_act]))
        
        # Update eligibility
        eligible.remove(best_act)
        for succ in successors[best_act]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                eligible.append(succ)
    
    if not scheduled_jobs:
        return 0.0
    makespan = max(e for s, e, d in scheduled_jobs)
    return makespan


def priority(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    N = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    R = len(capacities)
    
    # Calculate critical path length (backward pass)
    # First, topological sort
    preds = instance.get("predecessors", [[] for _ in range(N)])
    preds_list = []
    for i in range(N):
        if isinstance(preds, list) and i < len(preds):
            p = preds[i]
            if isinstance(p, (list, tuple, np.ndarray)):
                preds_list.append(list(p))
            elif p is None:
                preds_list.append([])
            else:
                preds_list.append([p])
        else:
            preds_list.append([])
    
    in_degree = np.zeros(N, dtype=int)
    successors = [[] for _ in range(N)]
    for i in range(N):
        for p in preds_list[i]:
            in_degree[i] += 1
            successors[p].append(i)
    
    # Topological sort (Kahn's algorithm)
    queue = [i for i in range(N) if in_degree[i] == 0]
    topo_order = []
    temp_in_deg = in_degree.copy()
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            temp_in_deg[v] -= 1
            if temp_in_deg[v] == 0:
                queue.append(v)
    
    # Backward pass for latest start times (assuming finish by infinity, so we compute longest path from node to sink)
    # Let's compute "critical path length" as the longest path from the node to any terminal node.
    # L[i] = duration[i] + max(L[j] for j in successors[i])
    L = np.zeros(N)
    # Process in reverse topological order
    for i in reversed(topo_order):
        max_succ = 0
        for j in successors[i]:
            if L[j] > max_succ:
                max_succ = L[j]
        L[i] = durations[i] + max_succ
    
    # Base priority: Critical Path Length
    base_prio = L.copy()
    
    # We can try to improve this using a simple local search or perturbation.
    # However, for N=30, a full local search might be too slow if not careful.
    # Let's try a simple hill climbing on the priority scores.
    # Objective: Minimize makespan from SGS.
    
    best_prio = base_prio.copy()
    best_makespan = _simulate_sgs_correct(instance, best_prio)
    
    # Local search: perturb one activity's priority slightly and see if it improves.
    # Or swap priorities of two activities?
    # Since priority is a continuous value, we can add noise.
    # But SGS is sensitive to ordering.
    # Let's try random restarts with small perturbations of the base priority.
    
    num_iterations = 500
    step_size = 0.1
    
    for k in range(num_iterations):
        if time.time() > deadline:
            break
        
        # Create a perturbed priority
        # Option 1: Add Gaussian noise
        # Option 2: Swap two random activities' priorities
        
        # Let's try swapping two random activities
        i, j = random.sample(range(N), 2)
        
        # Create new priority vector
        new_prio = best_prio.copy()
        # Swap? Or just perturb?
        # If we swap, the relative order of i and j changes.
        # Let's try a simple random perturbation of one element.
        idx = random.randint(0, N-1)
        new_prio[idx] = best_prio[idx] + random.gauss(0, step_size)
        
        # Simulate
        makespan = _simulate_sgs_correct(instance, new_prio)
        
        if makespan < best_makespan:
            best_prio = new_prio
            best_makespan = makespan
        else:
            # Revert (implicit since we didn't update best_prio)
            pass
    
    return best_prio
# EVOLVE-BLOCK-END
