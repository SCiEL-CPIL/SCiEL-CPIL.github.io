import numpy as np
import time
import itertools

def priority(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]
    
    n = len(durations)
    n_res = len(capacities)
    
    # Pre-compute static features
    # 1. Successors
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # 2. Forward Pass (ES, EF)
    es = np.zeros(n)
    ef = np.zeros(n)
    for i in range(n):
        for p in predecessors[i]:
            es[i] = max(es[i], ef[p])
        ef[i] = es[i] + durations[i]
    
    project_end_static = np.max(ef)
    
    # 3. Backward Pass (LS, LF)
    ls = np.zeros(n)
    lf = np.zeros(n)
    for i in range(n-1, -1, -1):
        if not successors[i]:
            lf[i] = project_end_static
        else:
            lf[i] = min(ls[s] for s in successors[i])
        ls[i] = lf[i] - durations[i]
        
    total_float = ls - es
    tail = project_end_static - ef
    
    # 4. Resource Features
    if n_res > 0:
        # Max demand ratio per activity
        max_demand_ratio = np.max(demands / np.maximum(capacities, 1e-9), axis=1)
        # Total demand weight
        weighted_demand = np.sum(demands / np.maximum(capacities, 1e-9), axis=1)
    else:
        max_demand_ratio = np.zeros(n)
        weighted_demand = np.zeros(n)
        
    # Normalization
    max_float = np.max(total_float) if np.max(total_float) > 0 else 1.0
    max_tail = np.max(tail) if np.max(tail) > 0 else 1.0
    max_dur = np.max(durations) if np.max(durations) > 0 else 1.0
    max_res = np.max(max_demand_ratio) if np.max(max_demand_ratio) > 0 else 1.0
    max_wres = np.max(weighted_demand) if np.max(weighted_demand) > 0 else 1.0
    
    # Base normalized features (higher is "more important" for the score)
    # We want to pick activities with LOW float, HIGH tail, HIGH duration, HIGH resource demand?
    # Let's define the score as a linear combination of normalized features.
    
    feat_float = (max_float - total_float) / max_float  # 1 for critical, 0 for max float
    feat_tail = tail / max_tail
    feat_dur = durations / max_dur
    feat_res = max_demand_ratio / max_res
    feat_wres = weighted_demand / max_wres
    
    # Internal Simulation Function
    def simulate(weights):
        """
        Simulates serial schedule generation.
        weights: (w_float, w_tail, w_dur, w_res, w_wres)
        Returns makespan.
        """
        # Calculate scores for all activities based on weights
        scores = (
            weights[0] * feat_float +
            weights[1] * feat_tail +
            weights[2] * feat_dur +
            weights[3] * feat_res +
            weights[4] * feat_wres
        )
        
        current_time = 0
        finished = [False] * n
        resource_usage = np.zeros(n_res)
        
        # To speed up, we can just iterate n times
        # In a serial scheme, we pick the highest priority eligible activity.
        # We need to determine eligibility.
        
        # Optimization: Pre-compute predecessor counts? 
        # Since n is small (30), simple check is fine.
        
        for _ in range(n):
            eligible = []
            for i in range(n):
                if finished[i]:
                    continue
                # Check if all predecessors are finished
                is_eligible = True
                for p in predecessors[i]:
                    if not finished[p]:
                        is_eligible = False
                        break
                if is_eligible:
                    eligible.append(i)
                    
            if not eligible:
                # Should not happen in valid DAG
                break
                
            # Find max score among eligible
            best_idx = -1
            best_score = -np.inf
            for i in eligible:
                if scores[i] > best_score:
                    best_score = scores[i]
                    best_idx = i
                    
            # Check resource availability? 
            # The serial scheme usually *waits* if resources aren't available?
            # OR does it just schedule it at the earliest possible time?
            # "Serial schedule-generation scheme starts the highest-priority eligible activity under precedence and renewable-resource limits."
            # This usually means:
            # 1. Find eligible set (precedence met).
            # 2. Sort by priority.
            # 3. For the highest priority one, find the earliest time t >= current_time 
            #    such that resources are available for its entire duration?
            #    NO, serial scheme usually just schedules at the *earliest possible time* 
            #    considering resources.
            #    Actually, standard Serial SGS:
            #    - Generate eligible set E_t.
            #    - Select activity j with highest priority.
            #    - Determine earliest start time t_j >= t such that resource constraints are met 
            #      for [t_j, t_j + d_j].
            #    - Schedule j at t_j.
            #    - Update t = t_j + d_j? No, t stays or advances to next event?
            #    Usually, in serial SGS, we advance time to the start of the scheduled activity.
            
            # Let's implement the "Earliest Start Time" logic for the selected activity.
            # This requires checking resource availability.
            
            # Find earliest start for best_idx
            earliest_start = current_time
            # We need to check if resources are available from earliest_start to earliest_start + durations[best_idx]
            # If not, we might need to wait until a resource frees up.
            # Since n is small, we can simulate forward.
            
            # Get current resource usage profile? 
            # We only track `resource_usage` as a scalar, which is WRONG for overlapping activities.
            # We need to know *when* resources are free.
            
            # Correct Simulation State:
            # We need to track the timeline of resource usage.
            # Since n=30, we can maintain a list of (end_time, activity) for each resource.
            
            # Let's restart the simulation logic with proper resource tracking.
            pass # Placeholder, will rewrite below
            
    # Revised Simulation Logic
    def simulate_v2(weights):
        scores = (
            weights[0] * feat_float +
            weights[1] * feat_tail +
            weights[2] * feat_dur +
            weights[3] * feat_res +
            weights[4] * feat_wres
        )
        
        finished = [False] * n
        # For each resource, a list of (end_time, amount_used) intervals? 
        # Or simpler: a list of activities currently running or scheduled.
        # Since we schedule one by one, we can just keep a list of (end_time, res_idx, amount) for active/scheduled jobs.
        
        # To check availability, we need to know the max usage in the interval [start, start+dur].
        # With 30 activities, we can just keep a list of scheduled activities: (start, end, res_usage_vector).
        
        scheduled = []
        makespan = 0
        
        for _ in range(n):
            # Find eligible
            eligible = []
            for i in range(n):
                if finished[i]:
                    continue
                if all(finished[p] for p in predecessors[i]):
                    eligible.append(i)
            
            if not eligible:
                break
                
            # Select highest priority
            best_idx = max(eligible, key=lambda x: scores[x])
            
            # Find earliest start time
            # Start from current makespan? No, start from 0?
            # The activity can start as soon as predecessors are done AND resources are free.
            # Predecessors are done, so their end times are known.
            # But we don't track start/end times of all activities in a way that allows easy "min end time of preds".
            # We can compute ES (earliest start based on precedence) statically.
            # So earliest_start_candidate = max(es[best_idx], current_time_if_we_were_sequential?)
            # In Serial SGS, we don't have a "current_time" that advances for all. 
            # We just pick the next activity. Its start time is determined by:
            # 1. Precedence: max(ES_pred)
            # 2. Resources: earliest t >= ES_pred such that resources are available.
            
            es_pred = es[best_idx]
            
            # Check resources
            # We have a list of scheduled activities: (start, end, demands_vector)
            # We need to find the smallest t >= es_pred such that for all r:
            # sum(demands[k][r] for k in scheduled if start_k < t+dur and end_k > t) + demands[best_idx][r] <= capacities[r]
            
            # Since n is small, we can just iterate t from es_pred upwards?
            # Or better: The only times resources change are at start or end times of scheduled activities.
            # Candidate times: es_pred, and all end times of scheduled activities that are > es_pred.
            
            candidate_times = [es_pred]
            for (s, e, d_vec) in scheduled:
                if e > es_pred:
                    candidate_times.append(e)
            
            # Also, we might need to check if resources are free *at* es_pred.
            # If not, we wait until some activity ends.
            
            valid_start = None
            for t in sorted(set(candidate_times)):
                if t < es_pred:
                    continue
                # Check resource usage at t
                usage = np.zeros(n_res)
                for (s, e, d_vec) in scheduled:
                    if s < t + durations[best_idx] and e > t:
                        usage += d_vec
                if np.all(usage + demands[best_idx] <= capacities):
                    valid_start = t
                    break
            
            if valid_start is None:
                # Should not happen if capacities are sufficient
                valid_start = es_pred
            
            # Schedule activity
            scheduled.append((valid_start, valid_start + durations[best_idx], demands[best_idx]))
            finished[best_idx] = True
            makespan = max(makespan, valid_start + durations[best_idx])
            
            if time.time() > deadline:
                return makespan
                
        return makespan

    # Search for best weights
    best_makespan = float('inf')
    best_scores = None
    
    # Grid of weights
    # w_float, w_tail, w_dur, w_res, w_wres
    # Sum of weights doesn't strictly matter, but normalization helps.
    
    # Try a set of promising configurations
    weight_configs = []
    
    # 1. Pure Critical Path (Min Float)
    weight_configs.append((1.0, 0.0, 0.0, 0.0, 0.0))
    
    # 2. Min Float + Max Resource
    for w_res in [0.1, 0.2, 0.5, 1.0]:
        weight_configs.append((1.0, 0.0, 0.0, w_res, 0.0))
        
    # 3. Min Float + Max Tail
    for w_tail in [0.1, 0.2, 0.5]:
        weight_configs.append((1.0, w_tail, 0.0, 0.0, 0.0))
        
    # 4. Min Float + Duration
    for w_dur in [0.1, 0.2, 0.5]:
        weight_configs.append((1.0, 0.0, w_dur, 0.0, 0.0))
        
    # 5. Combinations
    for w_f in [0.5, 1.0]:
        for w_t in [0.0, 0.5]:
            for w_r in [0.0, 0.5, 1.0]:
                weight_configs.append((w_f, w_t, 0.0, w_r, 0.0))
                
    # 6. Random search if time permits
    np.random.seed(42)
    for _ in range(50):
        w = np.random.rand(5)
        w /= np.sum(w) # Normalize
        weight_configs.append(tuple(w))
        
    for w in weight_configs:
        if time.time() > deadline:
            break
        m = simulate_v2(w)
        if m < best_makespan:
            best_makespan = m
            # Recalculate scores for these weights
            best_scores = (
                w[0] * feat_float +
                w[1] * feat_tail +
                w[2] * feat_dur +
                w[3] * feat_res +
                w[4] * feat_wres
            )
            
    if best_scores is None:
        # Fallback to default
        best_scores = feat_float
        
    return best_scores
