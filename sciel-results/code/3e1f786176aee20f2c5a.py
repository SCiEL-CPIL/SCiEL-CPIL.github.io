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
    DEADLINE = 135.0  # safety margin before 150s limit
    
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
        if time.time() > DEADLINE:
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
    
    if best_priority is None:
        best_priority = lpe.copy()
    
    # Final local-search polish: perturb the best priority vector and re-simulate
    # This helps escape local optima of the static heuristic
    if time.time() < DEADLINE - 5:
        rng = random.Random(42)
        base = best_priority.copy()
        scale = float(np.std(base)) if np.std(base) > 1e-9 else 1.0
        # Try several perturbation magnitudes
        for mag in [0.05, 0.1, 0.2, 0.3, 0.5]:
            if time.time() > DEADLINE - 5:
                break
            for trial in range(30):
                if time.time() > DEADLINE - 5:
                    break
                # Add small Gaussian noise
                p = base + rng.gauss(0, mag * scale)
                stations = simulate(p)
                if stations < best_stations:
                    best_stations = stations
                    best_priority = p.copy()
                    base = p.copy()
    
    return np.asarray(best_priority, dtype=float)

# EVOLVE-BLOCK-END
