import numpy as np
import time as time_module
import random

def _soft_max(values, temperature=1.0):
    """Smooth max via log-sum-exp for differentiable-like surrogate."""
    if not values:
        return 0.0
    vals = np.array(values, dtype=np.float64)
    max_val = np.max(vals)
    if max_val < -1e10:
        return max_val
    shifted = vals - max_val
    exp_sum = np.sum(np.exp(shifted / temperature))
    return max_val + temperature * np.log(exp_sum)

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    # Deadline guard
    deadline = time_module.time() + 145.0  # 5s safety margin before 150s kill
    
    # --- MODEL-AND-OPTIMIZE APPROACH ---
    # We parameterize the dispatch decision with a set of heuristic weights and 
    # use a lightweight local search / beam search over the decision space.
    # The "model" is: score(op) = w1 * remaining_work + w2 * min_duration_inv + 
    #                           w3 * machine_balance + w4 * job_priority
    # We optimize these weights on the fly using a short lookahead simulation.
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # Precompute features for each ready operation
    features = []
    for op in ready_operations:
        rw = op["remaining_work"]
        durations = list(op["machine_options"].values())
        min_dur = min(durations) if durations else 0
        max_dur = max(durations) if durations else 0
        avg_dur = sum(durations) / len(durations) if durations else 0
        
        # Number of remaining ops in this job
        rem_ops = op.get("remaining_ops", [])
        n_rem = len(rem_ops) if isinstance(rem_ops, list) else 0
        
        # Feature: how many machines are options
        n_machines = len(op["machine_options"]) if op["machine_options"] else 1
        
        features.append({
            "remaining_work": rw,
            "min_dur": min_dur,
            "max_dur": max_dur,
            "avg_dur": avg_dur,
            "min_dur_inv": 1.0 / (min_dur + 1.0),
            "n_rem_ops": n_rem,
            "n_machines": n_machines,
            # Critical path estimate: remaining work / max parallelism
            "critical_path": rw / max(n_machines, 1)
        })
    
    # --- STRATEGY 1: Lookahead Beam Search ---
    # Check if we can simulate forward (i.e., remaining_ops contains actual op dicts)
    can_simulate = False
    if ready_operations:
        sample_op = ready_operations[0]
        rem_ops = sample_op.get("remaining_ops", None)
        if isinstance(rem_ops, list) and len(rem_ops) > 0:
            if isinstance(rem_ops[0], dict) and "machine_options" in rem_ops[0]:
                can_simulate = True
    
    if can_simulate and n <= 10:
        # Beam search with width 3, depth 2
        # State: (list of ready ops, elapsed_time)
        # We simulate: pick an op, assign it to its best machine (min duration),
        # compute completion, update state.
        
        def simulate_one_step(ready_ops, t, chosen_idx):
            """Simulate executing chosen_idx, return new ready ops and new time."""
            op = ready_ops[chosen_idx]
            durations = list(op["machine_options"].values())
            min_dur = min(durations) if durations else 0
            comp_t = t + min_dur
            
            new_ready = []
            for j, other in enumerate(ready_ops):
                if j == chosen_idx:
                    rem = op.get("remaining_ops", [])
                    if isinstance(rem, list) and len(rem) > 0:
                        next_op = rem[0].copy()
                        # Update remaining_work: subtract what was done
                        next_op["remaining_work"] = max(0, op["remaining_work"] - min_dur)
                        # Update remaining_ops
                        if isinstance(rem, list) and len(rem) > 1:
                            next_op["remaining_ops"] = rem[1:]
                        else:
                            next_op["remaining_ops"] = []
                        new_ready.append(next_op)
                else:
                    # Other ops stay ready (open shop: parallel execution allowed)
                    new_ready.append(other)
            return new_ready, comp_t
        
        def estimate_makespan(ready_ops, current_time):
            """Estimate remaining makespan from current state."""
            if not ready_ops:
                return current_time
            max_rw = max(op["remaining_work"] for op in ready_ops)
            # Rough estimate: max of (current_time + critical path of each job)
            # We use max remaining work as a lower bound
            return max(current_time, current_time + max_rw * 0.5)
        
        # Beam search: start with all candidates, keep top-beam_width
        beam_width = 3
        # First level: evaluate each candidate
        candidates = []
        for i in range(n):
            if time_module.time() > deadline:
                break
            new_ready, new_t = simulate_one_step(ready_operations, time, i)
            score = estimate_makespan(new_ready, new_t)
            candidates.append((score, i))
        
        if not candidates:
            # Fallback
            return int(np.argmax([f["remaining_work"] for f in features]))
        
        # Sort by score, keep beam_width
        candidates.sort(key=lambda x: x[0])
        top_k = candidates[:beam_width]
        
        # Second level: for each top candidate, look one more step
        best_score = float('inf')
        best_idx = top_k[0][1]
        
        for level1_score, level1_idx in top_k:
            if time_module.time() > deadline:
                break
            new_ready, new_t = simulate_one_step(ready_operations, time, level1_idx)
            
            # Now from this state, try each possible next choice
            for j in range(len(new_ready)):
                if time_module.time() > deadline:
                    break
                new_ready2, new_t2 = simulate_one_step(new_ready, new_t, j)
                score2 = estimate_makespan(new_ready2, new_t2)
                # Total score: combine level1 and level2
                # Use max to represent makespan-like behavior
                total_score = max(level1_score, score2)
                if total_score < best_score:
                    best_score = total_score
                    best_idx = level1_idx
        
        return best_idx
    
    # --- STRATEGY 2: Weighted Heuristic with Optimized Weights ---
    # If simulation isn't possible or n is large, use a weighted score
    # We try several weight configurations and pick the one that minimizes
    # a proxy makespan estimate.
    
    weight_sets = [
        # (w1: remaining_work, w2: min_dur_inv, w3: n_rem_ops, w4: critical_path)
        (1.0, 0.1, 0.5, 0.3),
        (1.0, 0.0, 0.5, 0.0),
        (0.8, 0.2, 0.3, 0.5),
        (1.0, 0.05, 0.2, 0.4),
        (0.5, 0.3, 0.5, 0.5),
        (1.0, 0.15, 0.0, 0.5),
    ]
    
    # Evaluate each weight set using a quick proxy:
    # For each weight set, score all ops, pick best, then estimate makespan
    best_overall_score = float('inf')
    best_choice = 0
    
    for w1, w2, w3, w4 in weight_sets:
        if time_module.time() > deadline:
            break
        
        scores = []
        for f in features:
            s = w1 * f["remaining_work"] + w2 * f["min_dur_inv"] * f["remaining_work"] + \
                w3 * f["n_rem_ops"] * 10.0 + w4 * f["critical_path"]
            scores.append(s)
        
        chosen = int(np.argmax(scores))
        
        # Estimate resulting makespan: 
        # The chosen op takes min_dur. After it completes, the job's remaining work decreases.
        # Other jobs' remaining work is unchanged.
        # Makespan estimate = max over all jobs of (their remaining work after this step)
        # Since we don't track per-job totals easily, use:
        # estimate = max(current_time + chosen_min_dur, max_remaining_work_of_others)
        
        chosen_min_dur = features[chosen]["min_dur"]
        comp_t = time + chosen_min_dur
        
        # Max remaining work among other ops
        max_rw_others = 0
        for i, f in enumerate(features):
            if i != chosen:
                max_rw_others = max(max_rw_others, f["remaining_work"])
        
        # The chosen job's remaining work after this op
        chosen_rw_after = max(0, features[chosen]["remaining_work"] - chosen_min_dur)
        
        # Overall makespan estimate: at least comp_t, and at least max_rw_others (they still need work)
        est = max(comp_t, max_rw_others, chosen_rw_after)
        
        if est < best_overall_score:
            best_overall_score = est
            best_choice = chosen
    
    return best_choice
