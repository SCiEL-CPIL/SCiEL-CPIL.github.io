"""Rewritten program for RCPSP priority scoring using MCTS-inspired lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict
import heapq
import itertools
import random

def _fast_simulate(instance, priority_vec):
    """
    Fast serial schedule generation scheme.
    Returns makespan.
    Optimized for speed: uses heaps and precomputed structures.
    """
    n = len(instance["durations"])
    if n == 0:
        return 0.0
    
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    if predecessors is None:
        predecessors = [set() for _ in range(n)]
    else:
        preds = []
        for p in predecessors:
            if p is None:
                preds.append(set())
            else:
                preds.append(set(p))
        predecessors = preds
        
    # Precompute successors
    successors = [set() for _ in range(n)]
    for i in range(n):
        for j in predecessors[i]:
            successors[j].add(i)
            
    # Resource setup
    n_resources = 0
    if demands is not None and capacities is not None:
        n_resources = len(capacities)
        cap_vector = np.array(capacities, dtype=float)
        demand_matrix = np.zeros((n, n_resources), dtype=float)
        for i in range(n):
            if isinstance(demands[i], (list, tuple, np.ndarray)):
                for r in range(n_resources):
                    demand_matrix[i, r] = float(demands[i][r])
            else:
                demand_matrix[i, 0] = float(demands[i])
    else:
        n_resources = 0
        cap_vector = np.zeros(0)
        demand_matrix = np.zeros((n, 0))
        
    # State
    remaining = set(range(n))
    scheduled_count = 0
    current_time = 0.0
    active_jobs = [] # (finish_time, job_id)
    finish_times = [0.0] * n
    start_times = [0.0] * n
    
    # In-degree for eligibility
    in_deg = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            in_deg[i] += 1
            
    # Eligible list (jobs with in_deg 0)
    # We need to update in_deg as jobs finish.
    # To be fast, we maintain a list of eligible jobs.
    eligible = [i for i in range(n) if in_deg[i] == 0]
    
    # Priority cache for eligible jobs? 
    # We sort eligible by priority_vec each time we need to pick.
    # Since n=30, sorting is cheap.
    
    while remaining:
        if not eligible:
            # Advance time
            if not active_jobs:
                break # Should not happen
            min_finish, job_id = heapq.heappop(active_jobs)
            current_time = min_finish
            # Mark job as finished
            # Actually, we need to mark ALL jobs finishing at this time
            while active_jobs and active_jobs[0][0] == current_time:
                _, jid = heapq.heappop(active_jobs)
                finish_times[jid] = current_time
                # Update successors
                for succ in successors[jid]:
                    in_deg[succ] -= 1
                    if in_deg[succ] == 0:
                        eligible.append(succ)
                        remaining.discard(succ) # Wait, succ is still in remaining until scheduled?
                        # No, 'remaining' should contain unscheduled jobs.
                        # When a job is scheduled, it is removed from remaining.
                        # When a job finishes, it is removed from active_jobs.
                        # Let's clarify state:
                        # 'remaining': jobs not yet started.
                        # 'active_jobs': jobs started but not finished.
                        # 'scheduled': jobs finished.
                        # When a job is picked: remove from 'remaining', add to 'active_jobs'.
                        # When a job finishes: remove from 'active_jobs', mark as finished.
                        
                        # Correction:
                        # eligible: jobs in 'remaining' with in_deg 0.
                        # When we pick a job j:
                        #   remove j from 'remaining'
                        #   add j to 'active_jobs'
                        #   (j is no longer in 'eligible')
                        # When a job j finishes:
                        #   remove j from 'active_jobs'
                        #   for each succ s of j:
                        #       in_deg[s] -= 1
                        #       if in_deg[s] == 0:
                        #           s is now eligible (it is still in 'remaining')
                        #           add s to 'eligible'
                        
                        pass # Logic handled below
        
        if not eligible:
            continue
            
        # Pick best eligible job
        # Sort eligible by priority_vec (descending)
        # Also check resource availability
        best_job = -1
        best_priority = -1e9
        current_usage = np.zeros(n_resources) if n_resources > 0 else None
        
        # Calculate current usage from active_jobs
        if n_resources > 0:
            for ft, jid in active_jobs:
                if start_times[jid] <= current_time < ft:
                    current_usage += demand_matrix[jid]
                    
        # We need to find a job that fits.
        # Since n is small, we can just iterate sorted eligible.
        eligible_sorted = sorted(eligible, key=lambda x: priority_vec[x], reverse=True)
        
        placed = False
        for j in eligible_sorted:
            if n_resources == 0:
                placed = True
                break
            new_usage = current_usage + demand_matrix[j]
            if np.all(new_usage <= cap_vector + 1e-9):
                placed = True
                break
                
        if not placed:
            # No job fits. Advance time.
            if not active_jobs:
                # Deadlock? Force place highest priority
                j = eligible_sorted[0]
                placed = True
            else:
                min_finish, _ = active_jobs[0]
                current_time = min_finish
                # Process all finishing at current_time
                while active_jobs and active_jobs[0][0] == current_time:
                    _, jid = heapq.heappop(active_jobs)
                    finish_times[jid] = current_time
                    for succ in successors[jid]:
                        in_deg[succ] -= 1
                        if in_deg[succ] == 0:
                            eligible.append(succ)
                continue
                
        # Schedule best_job
        if placed:
            # j is the last one checked in loop
            # Re-find j
            for j in eligible_sorted:
                if n_resources == 0:
                    break
                new_usage = current_usage + demand_matrix[j]
                if np.all(new_usage <= cap_vector + 1e-9):
                    break
            
            # Remove from eligible
            eligible.remove(j)
            remaining.discard(j)
            
            # Start job
            start_times[j] = current_time
            finish_times[j] = current_time + durations[j]
            heapq.heappush(active_jobs, (finish_times[j], j))
            
            # Note: j is not "finished" yet, so its successors are not updated yet.
            # They will be updated when j finishes.
            
    # Finish remaining active jobs
    while active_jobs:
        min_finish, jid = heapq.heappop(active_jobs)
        current_time = min_finish
        finish_times[jid] = current_time
        for succ in successors[jid]:
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                # It's finished, so it doesn't need to be scheduled.
                # But we need to ensure all jobs are accounted for.
                pass
                
    makespan = max(finish_times) if n > 0 else 0.0
    return makespan

def _run_mcts_beam(instance, beam_width=20, max_depth=30):
    """
    Run a Beam Search / MCTS to find good schedules.
    Returns a list of (makespan, schedule_sequence) tuples.
    """
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    if predecessors is None:
        predecessors = [set() for _ in range(n)]
    else:
        preds = []
        for p in predecessors:
            if p is None:
                preds.append(set())
            else:
                preds.append(set(p))
        predecessors = preds
        
    successors = [set() for _ in range(n)]
    for i in range(n):
        for j in predecessors[i]:
            successors[j].add(i)
            
    n_resources = 0
    if demands is not None and capacities is not None:
        n_resources = len(capacities)
        cap_vector = np.array(capacities, dtype=float)
        demand_matrix = np.zeros((n, n_resources), dtype=float)
        for i in range(n):
            if isinstance(demands[i], (list, tuple, np.ndarray)):
                for r in range(n_resources):
                    demand_matrix[i, r] = float(demands[i][r])
            else:
                demand_matrix[i, 0] = float(demands[i])
    else:
        n_resources = 0
        cap_vector = np.zeros(0)
        demand_matrix = np.zeros((n, 0))
        
    # Base heuristic for tie-breaking in beam
    # Use Critical Path
    in_deg = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            in_deg[i] += 1
            
    from collections import deque
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    temp_in_deg = in_deg[:]
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            temp_in_deg[succ] -= 1
            if temp_in_deg[succ] == 0:
                queue.append(succ)
                
    if len(topo_order) < n:
        return []
        
    longest_to_end = [0.0] * n
    for node in reversed(topo_order):
        max_succ = 0.0
        for succ in successors[node]:
            max_succ = max(max_succ, longest_to_end[succ])
        longest_to_end[node] = durations[node] + max_succ
        
    # Beam Search
    # State: (current_time, remaining_set, active_jobs_list, in_deg_list, sequence)
    # To save memory, we don't store full history, just the sequence of scheduled jobs.
    
    initial_in_deg = in_deg[:]
    initial_remaining = set(range(n))
    initial_active = []
    initial_time = 0.0
    
    beam = [(initial_time, frozenset(initial_remaining), tuple(initial_active), initial_in_deg[:], [])]
    
    all_schedules = []
    
    for depth in range(max_depth):
        new_beam = []
        for state in beam:
            current_time, remaining_f, active_tuple, in_deg_list, seq = state
            active_jobs = list(active_tuple)
            remaining = set(remaining_f)
            
            if not remaining:
                # Calculate makespan
                # Finish active jobs
                if active_jobs:
                    max_ft = max(ft for ft, _ in active_jobs)
                    makespan = max(current_time, max_ft)
                else:
                    makespan = current_time
                all_schedules.append((makespan, seq))
                continue
                
            # Find eligible
            eligible = [j for j in remaining if in_deg_list[j] == 0]
            
            if not eligible:
                # Advance time
                if not active_jobs:
                    break
                min_finish = min(ft for ft, _ in active_jobs)
                current_time = min_finish
                # Update active and in_deg
                new_active = []
                new_in_deg = in_deg_list[:]
                for ft, jid in active_jobs:
                    if ft == current_time:
                        for succ in successors[jid]:
                            new_in_deg[succ] -= 1
                    else:
                        new_active.append((ft, jid))
                active_jobs = new_active
                in_deg_list = new_in_deg
                # Re-evaluate eligible
                eligible = [j for j in remaining if in_deg_list[j] == 0]
                if not eligible:
                    continue # Should not happen

            # Calculate current usage
            current_usage = np.zeros(n_resources) if n_resources > 0 else None
            if n_resources > 0:
                for ft, jid in active_jobs:
                    if start_times_cache.get(jid, 0) <= current_time < ft: # Need start times?
                        # We don't store start times in state. 
                        # We can assume if active, it started before current_time.
                        # But we need to know if it's still running.
                        # Since we advance to min_finish, all active jobs with ft > current_time are running.
                        current_usage += demand_matrix[jid]
                        
            # Generate children
            children = []
            for j in eligible:
                if n_resources > 0:
                    new_usage = current_usage + demand_matrix[j]
                    if not np.all(new_usage <= cap_vector + 1e-9):
                        continue
                # Create new state
                new_remaining = remaining - {j}
                new_active = active_jobs + [(current_time + durations[j], j)]
                # in_deg doesn't change when starting a job, only when finishing.
                new_seq = seq + [j]
                # Heuristic score for beam pruning
                # Prefer jobs with high critical path
                score = longest_to_end[j]
                children.append((score, (current_time, frozenset(new_remaining), tuple(new_active), in_deg_list[:], new_seq)))
                
            # Sort children by heuristic score
            children.sort(key=lambda x: x[0], reverse=True)
            # Add top children to new_beam
            for _, child_state in children[:beam_width]:
                new_beam.append(child_state)
                
        # Prune new_beam to beam_width
        # We need to evaluate the "promisingness" of each state.
        # We can use the heuristic of the next best move or just keep the ones that came from high-scoring parents.
        # For simplicity, we just keep the first beam_width states from the sorted children of all parents?
        # Better: Merge all children, sort by a global heuristic, keep top beam_width.
        
        # Flatten and sort
        all_children = []
        # Re-do: The loop above added children to new_beam directly.
        # Let's restart the beam expansion logic to be cleaner.
        pass
        
    # The above loop was messy. Let's implement a cleaner Beam Search.
    
    # State: (time, remaining_mask, active_list, in_deg_tuple, seq)
    # Using tuples for immutability.
    
    def get_eligible(remaining_mask, in_deg_tuple):
        elig = []
        for j in range(n):
            if (remaining_mask >> j) & 1:
                if in_deg_tuple[j] == 0:
                    elig.append(j)
        return elig
        
    # Initial state
    init_mask = (1 << n) - 1
    init_in_deg = tuple(in_deg)
    init_active = tuple()
    beam = [(0.0, init_mask, init_active, init_in_deg, ())]
    
    for step in range(n):
        next_beam = []
        for time, mask, active_t, in_deg_t, seq in beam:
            active_list = list(active_t)
            remaining = [j for j in range(n) if (mask >> j) & 1]
            
            # Check if we need to advance time
            # Find eligible
            eligible = [j for j in remaining if in_deg_t[j] == 0]
            
            if not eligible:
                if not active_list:
                    continue
                min_ft = min(ft for ft, _ in active_list)
                time = min_ft
                # Update active and in_deg
                new_active = []
                new_in_deg = list(in_deg_t)
                for ft, jid in active_list:
                    if ft == time:
                        for succ in successors[jid]:
                            new_in_deg[succ] -= 1
                    else:
                        new_active.append((ft, jid))
                active_list = new_active
                in_deg_t = tuple(new_in_deg)
                eligible = [j for j in remaining if in_deg_t[j] == 0]
                if not eligible:
                    continue

            # Calculate usage
            usage = np.zeros(n_resources) if n_resources > 0 else None
            if n_resources > 0:
                for ft, jid in active_list:
                    # Assume all active jobs are running at 'time'
                    # This is true if we just advanced to min_ft, those with ft>time are running.
                    # Those with ft==time just finished, so we removed them.
                    usage += demand_matrix[jid]
                    
            candidates = []
            for j in eligible:
                if n_resources > 0:
                    new_usage = usage + demand_matrix[j]
                    if not np.all(new_usage <= cap_vector + 1e-9):
                        continue
                new_mask = mask & ~(1 << j)
                new_active = tuple(active_list + [(time + durations[j], j)])
                new_seq = seq + (j,)
                # Heuristic: Critical Path of j
                h = longest_to_end[j]
                candidates.append((h, (time, new_mask, new_active, in_deg_t, new_seq)))
                
            # Sort candidates by heuristic
            candidates.sort(key=lambda x: x[0], reverse=True)
            # Add top candidates
            for _, c_state in candidates[:beam_width]:
                next_beam.append(c_state)
                
        # Prune next_beam
        if not next_beam:
            break
        # Sort next_beam by a global estimate? 
        # We can just keep the first beam_width from the merged list?
        # Or sort by the last added job's heuristic?
        # Let's sort by the heuristic of the *next* potential move? Too expensive.
        # Just keep the ones that came from the best parents?
        # The current logic adds top beam_width children from *each* parent.
        # This results in beam_width * beam_width states.
        # We need to prune to beam_width.
        # Heuristic for pruning: Sum of critical paths of scheduled jobs? Or just keep the first beam_width.
        # Let's sort next_beam by the last job's critical path (proxy for quality)
        next_beam.sort(key=lambda x: longest_to_end[x[4][-1]] if x[4] else 0, reverse=True)
        beam = next_beam[:beam_width]
        
    # Calculate makespans for final beam states
    results = []
    for time, mask, active_t, in_deg_t, seq in beam:
        if mask == 0:
            # All scheduled
            if active_t:
                max_ft = max(ft for ft, _ in active_t)
                makespan = max(time, max_ft)
            else:
                makespan = time
            results.append((makespan, list(seq)))
            
    return results

def priority(instance):
    """
    Score each activity using MCTS-derived static priorities.
    """
    n = len(instance["durations"])
    if n == 0:
        return np.array([], dtype=float)
        
    # Run Beam Search / MCTS
    # Adjust beam width and depth based on n
    beam_width = 20
    schedules = _run_mcts_beam(instance, beam_width=beam_width, max_depth=n)
    
    if not schedules:
        # Fallback to critical path
        return _fallback_priority(instance)
        
    # Calculate average start time (rank) for each activity
    # Rank 0 = first, Rank 1 = second, etc.
    avg_rank = np.zeros(n)
    count = np.zeros(n)
    
    for makespan, seq in schedules:
        for idx, job in enumerate(seq):
            avg_rank[job] += idx
            count[job] += 1
            
    # Normalize ranks
    # If a job never appears in a schedule (should not happen), handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_rank_norm = np.where(count > 0, avg_rank / count, n)
        
    # Critical Path
    cp = _compute_critical_path(instance)
    
    # Combine: High CP is good, Low Rank (early) is good.
    # Priority = CP - alpha * Rank
    alpha = 0.5
    scores = cp - alpha * avg_rank_norm
    
    # Normalize
    if scores.max() > 0:
        scores = scores / scores.max()
        
    return scores

def _compute_critical_path(instance):
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance.get("predecessors", None)
    
    if predecessors is None:
        predecessors = [set() for _ in range(n)]
    else:
        preds = []
        for p in predecessors:
            if p is None:
                preds.append(set())
            else:
                preds.append(set(p))
        predecessors = preds
        
    successors = [set() for _ in range(n)]
    for i in range(n):
        for j in predecessors[i]:
            successors[j].add(i)
            
    in_deg = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            in_deg[i] += 1
            
    from collections import deque
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    temp_in_deg = in_deg[:]
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            temp_in_deg[succ] -= 1
            if temp_in_deg[succ] == 0:
                queue.append(succ)
                
    if len(topo_order) < n:
        return durations.copy()
        
    longest_to_end = [0.0] * n
    for node in reversed(topo_order):
        max_succ = 0.0
        for succ in successors[node]:
            max_succ = max(max_succ, longest_to_end[succ])
        longest_to_end[node] = durations[node] + max_succ
        
    return np.array(longest_to_end)

def _fallback_priority(instance):
    return _compute_critical_path(instance)
# EVOLVE-BLOCK-END
