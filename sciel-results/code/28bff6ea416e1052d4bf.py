"""Assembly Line Balancing - Direct structural approach with critical path analysis."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque
import heapq
import random

def priority(instance):
    """
    Score each assembly task for the priority rule.
    Uses a direct structural approach:
    1. Compute critical path length (longest path from task to end of line)
    2. Compute number of successors (positional breadth)
    3. Combine into a priority score that favors tasks that are on the critical path
       and have many successors (to unlock more downstream tasks early)
    4. Per-instance: try several weight combinations and return the one that minimizes stations.
    """
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build graph structures
    adj = [[] for _ in range(n)]  # adj[i] = successors of i
    preds_list = []
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        preds_list.append(preds)
        for p in preds:
            adj[p].append(i)
    
    # Topological sort
    in_deg = np.array([len(p) for p in preds_list], dtype=int)
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    in_deg_tmp = in_deg.copy()
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in adj[node]:
            in_deg_tmp[succ] -= 1
            if in_deg_tmp[succ] == 0:
                queue.append(succ)
    
    if len(topo_order) < n:
        # Cycle detected; fall back to task times
        return times.copy()
    
    # Compute longest path to end (LPE) for each task
    lpe = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        max_succ = 0.0
        for succ in adj[node]:
            val = times[succ] + lpe[succ]
            if val > max_succ:
                max_succ = val
        lpe[node] = times[node] + max_succ
    
    # Compute longest path from start (LPS) for each task
    lps = np.zeros(n, dtype=float)
    for node in topo_order:
        for succ in adj[node]:
            val = times[node] + lps[node]
            if val > lps[succ]:
                lps[succ] = val
    
    # Number of successors (transitive)
    num_succ = np.zeros(n, dtype=int)
    for node in reversed(topo_order):
        for succ in adj[node]:
            num_succ[node] += num_succ[succ] + 1
    
    # Position in topological order (0-based, 0 = earliest)
    pos_in_topo = np.zeros(n, dtype=float)
    for idx, node in enumerate(topo_order):
        pos_in_topo[node] = idx
    
    # Normalized position (0 = first, 1 = last)
    if n > 1:
        norm_pos = pos_in_topo / (n - 1)
    else:
        norm_pos = np.zeros(n, dtype=float)
    
    # Candidate priority scores
    candidates = {}
    
    # Critical path length (most important structural measure)
    candidates["lpe"] = lpe.copy()
    
    # Critical path + successor count
    candidates["lpe_succ"] = lpe + 0.01 * num_succ.astype(float)
    
    # Critical path scaled by successor count
    candidates["lpe_times_succ"] = lpe * (1.0 + 0.05 * np.log1p(num_succ.astype(float)))
    
    # Longest path from start
    candidates["lps"] = lps.copy()
    
    # Task time
    candidates["time"] = times.copy()
    
    # Task time + critical path
    candidates["time_lpe"] = times + 0.1 * lpe
    
    # Inverse topological position (earlier is better)
    candidates["early_pos"] = 1.0 - norm_pos
    
    # Combined: critical path weighted by how much it constrains the line
    candidates["combo1"] = lpe * (1.0 + 0.02 * num_succ.astype(float))
    
    candidates["combo2"] = 0.7 * lpe + 0.3 * lps
    
    candidates["combo3"] = lpe + 0.05 * num_succ.astype(float) - 0.01 * norm_pos
    
    candidates["combo4"] = (lpe / (lpe.max() + 1e-9)) * (1.0 + 0.1 * num_succ.astype(float) / (num_succ.max() + 1e-9))
    
    candidates["combo5"] = lpe * (1.0 + 0.03 * np.sqrt(num_succ.astype(float)))
    
    candidates["combo6"] = lpe + 0.1 * num_succ.astype(float)
    
    candidates["combo7"] = 0.8 * lpe + 0.2 * times * (1.0 + 0.05 * num_succ.astype(float))
    
    candidates["combo8"] = lpe * np.exp(0.01 * num_succ.astype(float))
    
    # New candidates: more diverse structural features
    # Slack: how much room is left after placing this task on its critical path
    total_time = times.sum()
    min_stations_lb = int(np.ceil(total_time / cycle_time))
    slack = lpe - times  # remaining work after this task
    candidates["slack"] = slack.copy()
    
    # Weighted critical path: emphasize tasks that are bottlenecks
    # A task is a bottleneck if removing it would reduce the critical path
    # Approximate: tasks where lpe is close to the maximum lpe
    max_lpe = lpe.max()
    if max_lpe > 0:
        bottleneck_score = 1.0 - (lpe / max_lpe)
        candidates["bottleneck"] = bottleneck_score.copy()
        candidates["lpe_bottleneck"] = lpe * (1.0 + 0.5 * bottleneck_score)
    
    # Degree-based: direct successors count
    direct_succ = np.array([len(adj[i]) for i in range(n)], dtype=float)
    candidates["lpe_direct"] = lpe + 0.1 * direct_succ
    
    # Rank-based: rank by LPE, then break ties by number of successors
    lpe_rank = np.argsort(np.argsort(-lpe)).astype(float)
    candidates["lpe_rank"] = -lpe_rank  # lower rank = higher priority
    
    # Hybrid: normalized LPE + normalized successors
    lpe_norm = lpe / (lpe.max() + 1e-9)
    succ_norm = num_succ.astype(float) / (num_succ.max() + 1e-9)
    candidates["norm_combo"] = lpe_norm + 0.2 * succ_norm
    
    # Geometric mean of LPE and (1 + successors)
    candidates["geo_mean"] = np.sqrt(lpe * (1.0 + num_succ.astype(float)))
    
    # Logarithmic: log(1 + LPE) + 0.1 * log(1 + successors)
    candidates["log_combo"] = np.log1p(lpe) + 0.1 * np.log1p(num_succ.astype(float))
    
    # New: time-normalized combinations
    time_norm = times / (times.max() + 1e-9)
    candidates["lpe_time_norm"] = lpe_norm + 0.2 * time_norm
    
    # New: inverse position weighted
    inv_pos = 1.0 / (1.0 + norm_pos)
    candidates["lpe_invpos"] = lpe_norm + 0.15 * inv_pos
    
    # New: product of normalized features
    candidates["lpe_prod"] = lpe_norm * (1.0 + 0.3 * succ_norm)
    
    # New: rank-based with secondary keys
    # Primary: LPE rank, Secondary: successor count rank
    succ_rank = np.argsort(np.argsort(-num_succ.astype(float))).astype(float)
    candidates["rank_combo"] = -(lpe_rank * 1.0 + succ_rank * 0.1)
    
    # New: weighted by how many predecessors (later tasks have more context)
    pred_count = np.array([len(p) for p in preds_list], dtype=float)
    candidates["lpe_pred"] = lpe + 0.05 * pred_count
    
    # Additional structural features
    # Weighted sum of successor times
    succ_time_sum = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        for succ in adj[node]:
            succ_time_sum[node] += times[succ] + succ_time_sum[succ]
    candidates["lpe_succ_time"] = lpe + 0.05 * succ_time_sum / (succ_time_sum.max() + 1e-9)
    
    # Critical path position: how early this task appears on the critical path
    # Tasks that appear early on the critical path should be prioritized
    crit_path_pos = np.zeros(n, dtype=float)
    max_lpe_val = lpe.max()
    for node in topo_order:
        # Find the critical path through this node
        crit_path_pos[node] = lps[node] / (max_lpe_val + 1e-9)
    candidates["crit_pos"] = crit_path_pos.copy()
    
    # Combined critical path and position
    candidates["lpe_crit_pos"] = lpe * (1.0 + 0.1 * crit_path_pos)
    
    def simulate(priority_scores):
        """Fast greedy simulation using a heap of eligible tasks sorted by priority."""
        completed = [False] * n
        stations = 0
        current_load = 0.0
        remaining_count = n
        
        # pending[t] = number of uncompleted predecessors of t
        pending = [len(p) for p in preds_list]
        
        # Heap entries: (-priority, task_id)
        heap = []
        
        # Initially eligible: tasks with no predecessors that fit
        for t in range(n):
            if pending[t] == 0 and times[t] <= cycle_time + 1e-9:
                heapq.heappush(heap, (-priority_scores[t], t))
        
        while remaining_count > 0:
            # Pop highest-priority eligible task that fits remaining capacity
            placed = False
            while heap:
                neg_pri, t = heapq.heappop(heap)
                if completed[t]:
                    continue
                if times[t] <= cycle_time - current_load + 1e-9:
                    # Place it
                    completed[t] = True
                    current_load += times[t]
                    remaining_count -= 1
                    placed = True
                    # Check successors
                    for s in adj[t]:
                        pending[s] -= 1
                        if pending[s] == 0 and times[s] <= cycle_time + 1e-9:
                            heapq.heappush(heap, (-priority_scores[s], s))
                    break
            
            if not placed:
                # No eligible task fits in current station; start new station
                stations += 1
                current_load = 0.0
                # Re-push all eligible (pending==0, not completed) tasks
                for t in range(n):
                    if not completed[t] and pending[t] == 0 and times[t] <= cycle_time + 1e-9:
                        heapq.heappush(heap, (-priority_scores[t], t))
            
            if not heap:
                # Shouldn't happen for feasible instances; break to avoid infinite loop
                stations += 1
                break
        
        return stations
    
    # Evaluate candidates and pick the best
    best_priority = None
    best_stations = float('inf')
    
    for name, p in candidates.items():
        if time.time() > DEADLINE - 30:
            break
        stations = simulate(p)
        if stations < best_stations:
            best_stations = stations
            best_priority = p.copy()
    
    # Try some additional parameter sweeps if time allows
    if best_priority is not None and time.time() < DEADLINE - 30:
        # Sweep over lpe + alpha * num_succ
        for alpha in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15]:
            if time.time() > DEADLINE - 30:
                break
            p = lpe + alpha * num_succ.astype(float)
            stations = simulate(p)
            if stations < best_stations:
                best_stations = stations
                best_priority = p.copy()
        
        # Sweep over a*lpe + b*lps
        for a in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            if time.time() > DEADLINE - 30:
                break
            b = 1.0 - a
            p = a * lpe + b * lps
            stations = simulate(p)
            if stations < best_stations:
                best_stations = stations
                best_priority = p.copy()
        
        # Sweep over lpe * (1 + beta * sqrt(num_succ))
        for beta in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]:
            if time.time() > DEADLINE - 30:
                break
            p = lpe * (1.0 + beta * np.sqrt(num_succ.astype(float)))
            stations = simulate(p)
            if stations < best_stations:
                best_stations = stations
                best_priority = p.copy()
        
        # Sweep over lpe + gamma * direct_succ
        for gamma in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
            if time.time() > DEADLINE - 30:
                break
            p = lpe + gamma * direct_succ
            stations = simulate(p)
            if stations < best_stations:
                best_stations = stations
                best_priority = p.copy()
        
        # Sweep over normalized combinations
        for w in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]:
            if time.time() > DEADLINE - 30:
                break
            p = lpe_norm + w * succ_norm
            stations = simulate(p)
            if stations < best_stations:
                best_stations = stations
                best_priority = p.copy()
        
        # New: sweep over time-weighted combinations
        time_norm = times / (times.max() + 1e-9)
        for w in [0.0, 0.1, 0.2, 0.3, 0.5]:
            if time.time() > DEADLINE - 30:
                break
            p = lpe_norm + w * time_norm
            stations = simulate(p)
            if stations < best_stations:
                best_stations = stations
                best_priority = p.copy()
        
        # New: sweep over inverse position weighted
        inv_pos = 1.0 / (1.0 + norm_pos)
        for w in [0.0, 0.1, 0.2, 0.3]:
            if time.time() > DEADLINE - 30:
                break
            p = lpe_norm + w * inv_pos
            stations = simulate(p)
            if stations < best_stations:
                best_stations = stations
                best_priority = p.copy()
        
        # New: sweep over critical path position
        for w in [0.0, 0.05, 0.1, 0.15, 0.2]:
            if time.time() > DEADLINE - 30:
                break
            p = lpe * (1.0 + w * crit_path_pos)
            stations = simulate(p)
            if stations < best_stations:
                best_stations = stations
                best_priority = p.copy()
    
    if best_priority is None:
        best_priority = lpe.copy()
    
    # Final local-search polish: swap-based local search on the best priority vector
    # Use a more systematic approach: try swapping tasks that are close in priority
    if time.time() < DEADLINE - 5:
        rng = random.Random(42)
        base = best_priority.copy()
        best_base_stations = best_stations
        
        # Sort tasks by priority to focus on nearby swaps
        sorted_tasks = np.argsort(-base)
        
        # Multi-pass swap local search
        for pass_num in range(8):
            if time.time() > DEADLINE - 5:
                break
            
            improved = False
            # Try systematic swaps between nearby tasks in priority order
            for i_idx in range(min(n, 30)):
                if time.time() > DEADLINE - 5:
                    break
                i = sorted_tasks[i_idx]
                for j_idx in range(i_idx + 1, min(i_idx + 15, n)):
                    if time.time() > DEADLINE - 5:
                        break
                    j = sorted_tasks[j_idx]
                    
                    # Skip if priorities are too different
                    if abs(base[i] - base[j]) > 2.0 * (np.std(base) + 1e-9):
                        continue
                    
                    # Swap priorities
                    p = base.copy()
                    p[i], p[j] = p[j], p[i]
                    stations = simulate(p)
                    if stations < best_base_stations:
                        best_base_stations = stations
                        base = p.copy()
                        improved = True
                    elif stations == best_base_stations and rng.random() < 0.2:
                        # Accept ties with some probability to escape plateaus
                        base = p.copy()
            
            # If no improvement in this pass, break early
            if not improved:
                break
        
        if best_base_stations < best_stations:
            best_stations = best_base_stations
            best_priority = base.copy()
    
    # Additional refinement: try re-ranking with small offsets
    if time.time() < DEADLINE - 3:
        rng2 = random.Random(123)
        base2 = best_priority.copy()
        for trial in range(80):
            if time.time() > DEADLINE - 3:
                break
            # Add small random offset to a random subset of tasks
            p = base2.copy()
            subset_size = rng2.randint(2, 6)
            indices = rng2.sample(range(n), subset_size)
            for idx in indices:
                p[idx] += rng2.uniform(-0.3, 0.3) * (np.std(base2) + 1e-9)
            stations = simulate(p)
            if stations < best_stations:
                best_stations = stations
                best_priority = p.copy()
                base2 = p.copy()
    
    return np.asarray(best_priority, dtype=float)

# EVOLVE-BLOCK-END
