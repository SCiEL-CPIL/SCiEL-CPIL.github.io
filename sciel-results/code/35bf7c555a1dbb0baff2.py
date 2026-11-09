import numpy as np
from collections import deque
import time
import random

def _simulate_sgs(instance, scores):
    """
    Simulates the Serial Schedule Generation Scheme (SGS) given a set of priority scores.
    Returns the makespan.
    """
    durations = instance["durations"]
    predecessors = instance["predecessors"]
    demands = instance["demands"]
    capacities = instance["capacities"]
    n = len(durations)
    
    # Convert to numpy for fast operations if not already
    if not isinstance(durations, np.ndarray):
        durations = np.array(durations)
    if not isinstance(demands, np.ndarray):
        demands = np.array(demands)
    if not isinstance(capacities, np.ndarray):
        capacities = np.array(capacities)
        
    # Sort activities by priority (highest first)
    # We need to handle ties or just sort stably. 
    # SGS picks the highest priority *eligible* activity.
    # Efficient way: Keep a list of eligible activities. 
    # Since n is small (30), a simple loop finding max priority among eligible is O(n^2) or O(n^2 log n), which is fine.
    
    # Precompute successors for quick eligibility check? 
    # Actually, SGS works by:
    # 1. Set t = 0.
    # 2. Identify eligible activities: all predecessors scheduled, and resources available at t.
    # 3. Pick highest priority eligible.
    # 4. Schedule it at t. Update resources.
    # 5. If no eligible, advance t to next event (min finish time of active).
    
    # Data structures
    scheduled = [False] * n
    start_times = [0.0] * n
    end_times = [0.0] * n
    remaining_preds = [len(predecessors[j]) for j in range(n)]
    
    # Track resource usage over time.
    # Since n=30, max makespan is sum of durations. 
    # We can use an event-based simulation.
    # Active activities: list of (end_time, idx)
    active = []
    current_time = 0.0
    scheduled_count = 0
    
    # To efficiently find available resources at time t, we maintain a list of active activities.
    # When time advances, we remove finished activities from 'active' and free their resources.
    # Resource usage at time t: sum of demands of active activities.
    
    # Optimization: Pre-compute demand vectors? 
    # demands is (n, R).
    
    while scheduled_count < n:
        # 1. Advance time if no eligible activities can start at current_time
        # Check eligibility at current_time
        eligible = []
        # Calculate current resource usage
        # This is O(active * R). Active <= n.
        current_usage = np.zeros(len(capacities))
        for end_t, idx in active:
            current_usage += demands[idx]
            
        # Check each unscheduled activity
        for j in range(n):
            if scheduled[j]:
                continue
            if remaining_preds[j] > 0:
                continue
            # Check resources
            # If demands[j] <= capacities - current_usage
            if np.all(demands[j] <= (capacities - current_usage)):
                eligible.append(j)
                
        if eligible:
            # Pick highest priority
            best_j = max(eligible, key=lambda x: scores[x])
            # Schedule best_j at current_time
            scheduled[best_j] = True
            start_times[best_j] = current_time
            end_times[best_j] = current_time + durations[best_j]
            active.append((end_times[best_j], best_j))
            scheduled_count += 1
            
            # Update predecessors of best_j's successors
            # We need a successor list for this
            # Let's build it outside or inside? Inside is safer for isolation.
            # Actually, building it every iteration is O(n^2). n=30, so 900 ops. Fine.
            # But let's build it once outside for efficiency in the loop? 
            # No, let's just do it simply.
            
            # Wait, I need successors to update remaining_preds.
            # Let's pass successors or build it.
            # I'll build it in the main function and pass it.
            
            # For now, in this simulation, I'll just update remaining_preds by checking all unscheduled activities?
            # No, that's O(n^2).
            # Let's assume I have a `successors` list.
            
            # I will modify the function signature to accept successors.
            # Or just compute it here.
            
            # Let's refactor to compute successors in main and pass down.
            pass # Placeholder for logic below
            
            # Actually, let's rewrite the simulation to be cleaner with precomputed successors.
            
        else:
            # No eligible activity. Advance time to next event.
            if not active:
                # Deadlock? Should not happen if instance is valid.
                # But if all remaining have preds, and no active, we are stuck.
                # This implies a cycle or invalid instance.
                return float('inf')
            
            # Find min end time
            min_end = min(end_t for end_t, idx in active)
            current_time = min_end
            
            # Remove all activities that finish at min_end
            new_active = []
            for end_t, idx in active:
                if end_t < min_end:
                    # This shouldn't happen if we always jump to min
                    new_active.append((end_t, idx))
                # else it finishes now, so we drop it (freeing resources)
            active = new_active

    return max(end_times)

def priority(instance):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance["predecessors"]
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    n = len(durations)
    
    # Build successors
    successors = [[] for _ in range(n)]
    for j in range(n):
        for i in predecessors[j]:
            successors[i].append(j)
            
    # Topological sort for CPM
    in_degree = [len(predecessors[j]) for j in range(n)]
    queue = deque([j for j in range(n) if in_degree[j] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(topo_order) != n:
        return durations.copy()
        
    # 1. Compute CPM Slack
    ES = [0.0] * n
    EF = [0.0] * n
    for j in topo_order:
        if predecessors[j]:
            ES[j] = max(EF[i] for i in predecessors[j])
        else:
            ES[j] = 0.0
        EF[j] = ES[j] + durations[j]
        
    makespan_cpm = max(EF)
    
    LF = [makespan_cpm] * n
    LS = [makespan_cpm] * n
    for j in reversed(topo_order):
        if successors[j]:
            LF[j] = min(LS[k] for k in successors[j])
        else:
            LF[j] = makespan_cpm
        LS[j] = LF[j] - durations[j]
        
    Slack = [LS[j] - ES[j] for j in range(n)]
    
    # 2. Compute Successor Depth (Longest path to end, including self)
    # DD[j] = durations[j] + max(DD[k] for k in successors[j])
    DD = [0.0] * n
    for j in reversed(topo_order):
        if successors[j]:
            DD[j] = durations[j] + max(DD[k] for k in successors[j])
        else:
            DD[j] = durations[j]
            
    # 3. Compute Resource Load Intensity
    # I_j = sum_r (demands[j, r] / capacities[r])
    safe_cap = np.where(capacities > 0, capacities, 1.0)
    resource_impact = np.sum(demands / safe_cap, axis=1)
    
    # Normalize features to [0, 1] range roughly for stable weighting
    def normalize(arr):
        arr = np.array(arr)
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val == min_val:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    slack_norm = normalize(Slack)
    dd_norm = normalize(DD)
    impact_norm = normalize(resource_impact)
    
    # We want to MINIMIZE makespan.
    # High priority (score) should lead to good schedule.
    # Heuristic: 
    # - Low Slack is good (critical path). So score should be high when Slack is low. -> -slack_norm
    # - High DD is good (long tail). So score should be high when DD is high. -> +dd_norm
    # - High Impact is good (bottleneck). So score should be high when Impact is high. -> +impact_norm
    
    # Base scores for 3 features
    # Feature 1: -slack_norm (higher is better)
    # Feature 2: dd_norm (higher is better)
    # Feature 3: impact_norm (higher is better)
    
    # We will search for weights w1, w2, w3 in [0, 1] such that 
    # Score_j = w1 * (-slack_norm[j]) + w2 * dd_norm[j] + w3 * impact_norm[j]
    # minimizes the makespan produced by SGS.
    
    # Local Search / Random Restart
    best_score = float('inf')
    best_weights = (1.0, 0.0, 0.0) # Default to Min Slack
    
    # Initialize with a few random starts and local improvement
    num_restarts = 10
    max_iterations = 50
    step_size = 0.1
    
    # Pre-compute feature matrix for fast dot products
    # F is (n, 3)
    F = np.column_stack((-slack_norm, dd_norm, impact_norm))
    
    # Helper to compute scores from weights
    def get_scores(weights):
        return F @ weights
        
    # Helper to evaluate weights
    def evaluate(weights):
        scores = get_scores(weights)
        # Simulate SGS
        # We need a fast SGS. 
        # Let's implement a lightweight SGS inside this function to avoid global state issues.
        
        # Since n=30, we can afford a slightly slower SGS if we do many iterations?
        # 10 restarts * 50 iters * 30 steps * O(n) check = 10*50*30*30 = 450,000 ops. Very fast.
        
        scheduled = [False] * n
        remaining_preds = [len(predecessors[j]) for j in range(n)]
        active = [] # list of (end_time, idx)
        current_time = 0.0
        scheduled_count = 0
        end_times = [0.0] * n
        
        # Pre-compute successors for this instance (already done in outer scope, but let's use it)
        # successors is in closure.
        
        while scheduled_count < n:
            # Check eligibility
            eligible = []
            # Current usage
            # Optimization: Instead of summing active every time, we can maintain a usage array.
            # But active list is small.
            current_usage = np.zeros(len(capacities))
            for end_t, idx in active:
                current_usage += demands[idx]
                
            for j in range(n):
                if scheduled[j] or remaining_preds[j] > 0:
                    continue
                if np.all(demands[j] <= (capacities - current_usage)):
                    eligible.append(j)
                    
            if eligible:
                # Pick max score
                best_j = max(eligible, key=lambda x: scores[x])
                scheduled[best_j] = True
                start_t = current_time
                end_t = current_time + durations[best_j]
                end_times[best_j] = end_t
                active.append((end_t, best_j))
                scheduled_count += 1
                
                # Update preds
                for k in successors[best_j]:
                    remaining_preds[k] -= 1
            else:
                if not active:
                    return float('inf')
                min_end = min(end_t for end_t, idx in active)
                current_time = min_end
                # Remove finished
                new_active = []
                for end_t, idx in active:
                    if end_t > min_end:
                        new_active.append((end_t, idx))
                active = new_active
                
        return max(end_times)
        
    # Search
    for r in range(num_restarts):
        # Random initial weights
        w = np.random.rand(3)
        w /= np.sum(w) # Normalize to sum to 1? Or just keep in [0,1]?
        # Let's keep them in [0, 1] independently to allow "weak" features.
        w = np.random.rand(3)
        
        current_w = w.copy()
        current_score = evaluate(current_w)
        
        for i in range(max_iterations):
            if time.time() > deadline:
                break
                
            # Perturb
            idx = np.random.randint(3)
            delta = np.random.uniform(-step_size, step_size)
            new_w = current_w.copy()
            new_w[idx] += delta
            # Clip to [0, 1]
            new_w = np.clip(new_w, 0, 1)
            
            new_score = evaluate(new_w)
            
            if new_score < current_score:
                current_w = new_w
                current_score = new_score
            else:
                # Simple hill climbing: if we fail, we might restart or just continue?
                # To avoid local minima, let's do a random jump occasionally.
                if np.random.rand() < 0.1:
                    current_w = np.random.rand(3)
                    current_score = evaluate(current_w)
                    
        if current_score < best_score:
            best_score = current_score
            best_weights = current_w
            
    # Construct final scores using best weights
    final_scores = F @ best_weights
    
    return final_scores
