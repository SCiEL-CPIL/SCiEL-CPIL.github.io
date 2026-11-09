import numpy as np
import time
import random

def priority(instance):
    """
    Meta-Heuristic Priority Synthesis.
    
    Instead of using a fixed formula for priority scores, this function searches 
    for the best linear combination of heuristic features (Slack, Intensity, 
    Depth, Duration) that minimizes the makespan when used in a Serial SGS.
    
    It uses a fast local search (hill climbing with random restarts) within a 
    strict time budget to find instance-specific weights.
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    # --- 1. Parse Instance ---
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance["predecessors"]
    demands = np.asarray(instance["demands"], dtype=float) # shape (n, m)
    capacities = np.asarray(instance["capacities"], dtype=float) # shape (m,)
    
    n = len(durations)
    m = len(capacities)
    
    if n == 0:
        return np.array([])
    
    # --- 2. Pre-compute Static Features ---
    # Build adjacency and in-degrees
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for j in range(n):
        for i in predecessors[j]:
            if i is not None:
                successors[i].append(j)
                in_degree[j] += 1
                
    # Topological Sort
    queue = [j for j in range(n) if in_degree[j] == 0]
    topo_order = []
    temp_in_deg = in_degree.copy()
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            temp_in_deg[v] -= 1
            if temp_in_deg[v] == 0:
                queue.append(v)
                
    if len(topo_order) != n:
        # Cycle fallback: return durations
        return durations.copy()
        
    # CPM Calculations
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
        
    slack = [LS[j] - ES[j] for j in range(n)]
    
    # Feature Extraction
    # F1: Inverse Slack (normalized 0-1, higher is better)
    min_s = min(slack)
    max_s = max(slack)
    s_range = max_s - min_s if max_s > min_s else 1.0
    feat_slack = [1.0 - (slack[j] - min_s) / s_range for j in range(n)]
    
    # F2: Resource Intensity (max demand/capacity ratio)
    feat_intensity = [0.0] * n
    for j in range(n):
        max_ratio = 0.0
        for r in range(m):
            if capacities[r] > 0:
                ratio = demands[j][r] / capacities[r]
                if ratio > max_ratio:
                    max_ratio = ratio
        feat_intensity[j] = max_ratio
    max_int = max(feat_intensity) if feat_intensity else 1.0
    if max_int > 0:
        feat_intensity = [f / max_int for f in feat_intensity]
        
    # F3: Depth (longest path from start in nodes, normalized)
    depth = [0] * n
    for j in topo_order:
        if predecessors[j]:
            depth[j] = max(depth[i] for i in predecessors[j]) + 1
        else:
            depth[j] = 1
    max_d = max(depth) if depth else 1
    feat_depth = [d / max_d for d in depth]
    
    # F4: Duration (normalized)
    max_dur = max(durations) if n > 0 else 1.0
    feat_dur = [durations[j] / max_dur for j in range(n)]
    
    # Stack features: shape (n, 4)
    features = np.column_stack((feat_slack, feat_intensity, feat_depth, feat_dur))
    
    # --- 3. Serial SGS Simulation Function ---
    def simulate_sgs(weights):
        """
        Runs Serial SGS using priority scores = features @ weights.
        Returns makespan.
        """
        # Compute scores
        scores = features @ weights
        
        # Initialize state
        remaining_in_deg = in_degree.copy()
        eligible = [j for j in range(n) if remaining_in_deg[j] == 0]
        
        start_times = [0.0] * n
        end_times = [0.0] * n
        scheduled = []
        
        # Pre-compute resource usage intervals for efficiency? 
        # With n=30, O(n^2) check is fast enough.
        
        while len(scheduled) < n:
            if not eligible:
                break
                
            # Select highest score
            best_j = -1
            best_score = -float('inf')
            for j in eligible:
                if scores[j] > best_score:
                    best_score = scores[j]
                    best_j = j
                    
            if best_j == -1:
                break
                
            # Determine earliest start time
            prec_end = 0.0
            for i in predecessors[best_j]:
                if end_times[i] > prec_end:
                    prec_end = end_times[i]
                    
            # Candidate times: prec_end and end times of scheduled activities
            candidates = {prec_end}
            for k in scheduled:
                if end_times[k] > prec_end:
                    candidates.add(end_times[k])
            candidates = sorted(candidates)
            
            start_t = prec_end
            found = False
            for t in candidates:
                feasible = True
                # Check resource constraints
                for r in range(m):
                    usage = 0.0
                    for k in scheduled:
                        if start_times[k] <= t < end_times[k]:
                            usage += demands[k][r]
                    if usage + demands[best_j][r] > capacities[r] + 1e-9:
                        feasible = False
                        break
                if feasible:
                    start_t = t
                    found = True
                    break
                    
            if not found:
                # Should not happen if instance is feasible, but fallback to last candidate
                start_t = candidates[-1] if candidates else prec_end
                
            start_times[best_j] = start_t
            end_times[best_j] = start_t + durations[best_j]
            scheduled.append(best_j)
            
            # Update eligibility
            for v in successors[best_j]:
                remaining_in_deg[v] -= 1
                if remaining_in_deg[v] == 0:
                    eligible.append(v)
                    
            eligible.remove(best_j)
            
        makespan = max(end_times) if end_times else 0.0
        return makespan

    # --- 4. Local Search for Best Weights ---
    
    # Initialize with a good heuristic: High Slack weight, High Intensity weight
    # weights = [w_slack, w_int, w_depth, w_dur]
    best_weights = np.array([10.0, 5.0, 1.0, 0.5])
    best_makespan = simulate_sgs(best_weights)
    
    # Simple Random Restart Hill Climbing
    # We have ~140s. Each simulation is very fast (n=30).
    # Let's do many iterations.
    
    # Define a perturbation function
    def perturb(weights, magnitude=0.5):
        new_w = weights.copy()
        # Randomly adjust one or two weights
        for _ in range(random.randint(1, 2)):
            idx = random.randint(0, 3)
            new_w[idx] += random.uniform(-magnitude, magnitude)
            # Keep weights positive to maintain "higher score = higher priority" logic
            # Although negative weights could theoretically be useful, 
            # sticking to positive is safer for this heuristic structure.
            if new_w[idx] < 0.1:
                new_w[idx] = 0.1
        return new_w

    # Run search
    current_weights = best_weights.copy()
    current_makespan = best_makespan
    
    iteration = 0
    # Estimate iterations: simulation takes ~0.1-0.5ms. 140s allows ~100k-1M iterations.
    # We will loop until time limit.
    
    while time.time() - start_time < time_limit - 5.0: # 5s safety buffer
        # Check time more frequently in inner loop?
        # The simulation itself is the bottleneck.
        
        # Try a perturbation
        new_weights = perturb(current_weights, magnitude=0.2)
        
        # Simulate
        new_makespan = simulate_sgs(new_weights)
        
        if new_makespan < current_makespan:
            current_weights = new_weights
            current_makespan = new_makespan
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_weights = current_weights.copy()
        else:
            # With some probability, accept worse solution to escape local minima (Simulated Annealing lite)
            # But since we don't have a temperature schedule, simple hill climbing is safer/faster.
            # However, pure hill climbing might get stuck.
            # Let's do a random restart if stuck for a while?
            # For now, simple hill climbing with small steps is robust.
            pass
        
        iteration += 1
        
        # Occasional random restart if stuck (simple heuristic: if no improvement in last 1000 steps)
        # This adds complexity. Let's stick to hill climbing with varying step size.
        # Actually, let's add a simple restart mechanism every ~5000 iterations or if stuck.
        # But we don't track "stuck" easily here.
        
        # Alternative: Multi-start.
        # If we have time, we can do multiple independent hill climbs.
        # But the loop above is a single climb.
        
        # Let's modify: Every 1000 iterations, try a random restart from best_weights.
        if iteration % 1000 == 0:
            # 20% chance to restart from best
            if random.random() < 0.2:
                current_weights = best_weights.copy()
                current_makespan = best_makespan
                # Perturb significantly
                current_weights = perturb(current_weights, magnitude=2.0)
                current_makespan = simulate_sgs(current_weights)

    # --- 5. Generate Final Scores ---
    # Use the best weights found
    final_scores = features @ best_weights
    
    return final_scores

