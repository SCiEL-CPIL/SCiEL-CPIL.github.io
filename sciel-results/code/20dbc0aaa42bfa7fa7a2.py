"""RCPSP priority via Simulated Annealing on non-linear multiplicative features."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque
from heapq import heappush, heappop

def priority(instance):
    """
    Compute a priority score for each activity in a Resource-Constrained
    Project Scheduling Problem (RCPSP) instance.
    
    Strategy:
    1. Compute slack, duration, resource criticality, and downstream count.
    2. Define a non-linear multiplicative priority function with exponents.
    3. Optimize exponents via Simulated Annealing on makespan.
    4. Fast SGS with resource profile (integer) or direct check (float).
    """
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin

    # --- Extract instance components ---
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    preds = None
    for key in ["predecessors", "preds", "predecessors_list", "pre", "pred"]:
        if key in instance:
            preds = instance[key]
            break
    if preds is None:
        preds = [[] for _ in range(n)]

    succs = None
    for key in ["successors", "succs", "successors_list", "succ"]:
        if key in instance:
            succs = instance[key]
            break

    demands = None
    for key in ["demands", "resource_demands", "resources", "dem", "resource"]:
        if key in instance:
            demands = instance[key]
            break
    if demands is None:
        demands = np.zeros((n, 1))

    capacities = None
    for key in ["capacities", "capacity", "resource_capacities", "caps", "cap"]:
        if key in instance:
            capacities = instance[key]
            break
    if capacities is None:
        capacities = [1.0]

    demands = np.asarray(demands, dtype=float)
    if demands.ndim == 1:
        demands = demands.reshape(n, 1)
    R = demands.shape[1]
    capacities = np.asarray(capacities, dtype=float)
    if capacities.ndim == 0:
        capacities = np.full(R, float(capacities))

    # Build successor lists if not provided
    if succs is None:
        succs = [set() for _ in range(n)]
        for j in range(n):
            for i in (preds[j] if isinstance(preds[j], (list, tuple, set, np.ndarray)) else []):
                idx = int(i)
                if 0 <= idx < n:
                    succs[idx].add(j)
        succs = [list(s) for s in succs]

    # Build pred sets for fast lookup
    pred_set = [set() for _ in range(n)]
    for j in range(n):
        p = preds[j]
        if isinstance(p, (list, tuple, set, np.ndarray)):
            for i in p:
                idx = int(i)
                if 0 <= idx < n:
                    pred_set[j].add(idx)

    # --- Topological order ---
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        in_degree[j] = len(pred_set[j])

    q = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_deg_copy = in_degree.copy()
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)

    if len(topo_order) < n:
        for i in range(n):
            if i not in topo_order:
                topo_order.append(i)

    # --- Forward pass ---
    ES = np.zeros(n)
    EF = np.zeros(n)
    for u in topo_order:
        for v in succs[u]:
            if EF[u] > ES[v]:
                ES[v] = EF[u]
        EF[u] = ES[u] + durations[u]

    makespan_forward = float(np.max(EF)) if n > 0 else 0.0

    # --- Backward pass ---
    LS = np.zeros(n)
    LF = np.zeros(n)
    for u in reversed(topo_order):
        if len(succs[u]) == 0:
            LF[u] = makespan_forward
        else:
            LF[u] = float(np.min([LS[v] for v in succs[u]]))
        LS[u] = LF[u] - durations[u]

    slack = LS - ES

    # --- Compute base features ---
    eps = 1e-9
    
    # Resource criticality: max demand ratio
    max_demand_ratio = np.zeros(n)
    for i in range(n):
        for r in range(R):
            if capacities[r] > 0:
                ratio = demands[i, r] / capacities[r]
                if ratio > max_demand_ratio[i]:
                    max_demand_ratio[i] = ratio

    # Downstream count: number of transitive successors
    downstream_count = np.zeros(n, dtype=float)
    for i in range(n):
        visited = set()
        stack = [i]
        while stack:
            node = stack.pop()
            for v in succs[node]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        downstream_count[i] = len(visited)

    # Normalize features to [0, 1] to ensure stability in power operations
    max_dur = np.max(durations) + eps
    max_slack = np.max(slack) + eps
    max_crit = np.max(max_demand_ratio) + eps
    max_down = np.max(downstream_count) + eps
    
    # Features for the multiplicative model
    # We want high priority for: High Duration, Low Slack, High Crit, High Downstream
    # f_dur: High is good
    # f_slack_inv: High is good (1/(slack+1))
    # f_crit: High is good
    # f_down: High is good
    
    f_dur = durations / max_dur
    # Avoid division by zero or very small numbers causing huge values
    f_slack_inv = 1.0 / (slack + 1.0) 
    # Normalize f_slack_inv to [0,1] roughly
    max_f_slack_inv = np.max(f_slack_inv)
    f_slack_inv = f_slack_inv / max_f_slack_inv

    f_crit = max_demand_ratio / max_crit
    f_down = downstream_count / max_down

    # Pre-compute log-features for fast exponentiation? 
    # score = prod(f_i ** w_i) = exp(sum(w_i * log(f_i)))
    # This is faster and more numerically stable.
    log_f_dur = np.log(f_dur + 1e-6)
    log_f_slack_inv = np.log(f_slack_inv + 1e-6)
    log_f_crit = np.log(f_crit + 1e-6)
    log_f_down = np.log(f_down + 1e-6)
    
    log_features = np.vstack([log_f_dur, log_f_slack_inv, log_f_crit, log_f_down]).T # (n, 4)

    # Check if we can use integer profile
    is_integer = all(np.isclose(d, round(d)) for d in durations) and all(np.isclose(c, round(c)) for c in capacities)
    if is_integer:
        durations_int = np.round(durations).astype(int)
        capacities_int = np.round(capacities).astype(int)
        demands_int = np.round(demands).astype(int)
    else:
        durations_int = None
        capacities_int = None
        demands_int = None

    def simulate_sgs(prio_scores):
        """
        Serial SGS.
        """
        if n == 0:
            return 0.0
        
        start_t = np.full(n, np.inf)
        finish_t = np.full(n, np.inf)
        
        unscheduled = set(range(n))
        unscheduled_preds_count = np.array([len(pred_set[j]) for j in range(n)], dtype=int)
        
        eligible_heap = []
        for j in range(n):
            if unscheduled_preds_count[j] == 0:
                heappush(eligible_heap, (-prio_scores[j], j))
        
        scheduled = []  # List of indices scheduled
        
        while unscheduled:
            if not eligible_heap:
                # Fallback
                for j in unscheduled:
                    start_t[j] = ES[j]
                    finish_t[j] = ES[j] + durations[j]
                break
            
            neg_prio, j = heappop(eligible_heap)
            
            if j not in unscheduled:
                continue
            
            # Compute earliest start from precedence
            es_j = 0.0
            for p in pred_set[j]:
                cand = start_t[p] + durations[p]
                if cand > es_j:
                    es_j = cand
            
            # Candidate times: es_j and finish times of scheduled activities >= es_j
            cand_times = {es_j}
            for k in scheduled:
                if finish_t[k] >= es_j:
                    cand_times.add(finish_t[k])
            
            cand_times = sorted(cand_times)
            
            best_start = None
            
            if is_integer:
                # Use integer profile for speed
                max_t = int(np.sum(durations_int)) + 1
                # We need to check feasibility for each candidate t
                # Instead of rebuilding profile, we check overlap
                # With n=30, simple check is fine, but let's try to optimize
                # Actually, for integer, we can pre-compute a usage array?
                # No, because we schedule sequentially.
                
                for t in cand_times:
                    t_int = int(round(t))
                    end_t_int = t_int + durations_int[j]
                    
                    feasible = True
                    # Check resources
                    # We can sum demands of all scheduled activities overlapping [t, end_t)
                    # Optimization: Only check if t is a valid integer
                    if not np.isclose(t, t_int):
                        continue
                    
                    for r in range(R):
                        usage = 0
                        for k in scheduled:
                            s_k = int(round(start_t[k]))
                            f_k = int(round(finish_t[k]))
                            if s_k < end_t_int and f_k > t_int:
                                usage += demands_int[k, r]
                        
                        if usage + demands_int[j, r] > capacities_int[r]:
                            feasible = False
                            break
                    
                    if feasible:
                        best_start = t_int
                        break
            else:
                for t in cand_times:
                    end_t = t + durations[j]
                    feasible = True
                    for r in range(R):
                        usage = 0.0
                        for k in scheduled:
                            if start_t[k] < end_t and finish_t[k] > t:
                                usage += demands[k, r]
                        
                        if usage + demands[j, r] > capacities[r] + 1e-9:
                            feasible = False
                            break
                    
                    if feasible:
                        best_start = t
                        break
            
            if best_start is None:
                # Should not happen, but fallback
                best_start = es_j
            
            start_t[j] = best_start
            finish_t[j] = best_start + durations[j]
            unscheduled.discard(j)
            scheduled.append(j)
            
            for v in succs[j]:
                unscheduled_preds_count[v] -= 1
                if unscheduled_preds_count[v] == 0 and v in unscheduled:
                    heappush(eligible_heap, (-prio_scores[v], v))
        
        makespan = float(np.max(finish_t)) if n > 0 else 0.0
        return makespan

    # --- Simulated Annealing ---
    # Exponents for [dur, slack_inv, crit, down]
    # Initial guess
    w0 = np.array([1.0, 1.0, 1.0, 1.0])
    
    def objective(w):
        # score = exp(sum(w * log_features))
        # We want to maximize priority score, but SGS picks highest priority.
        # So prio_scores = exp(log_features @ w)
        # However, we just need the relative order.
        # We can just use log_scores = log_features @ w
        # And pass that to SGS which uses negative for heap.
        # But the problem asks for "priority scores". The SGS uses them to rank.
        # The returned scores should be such that higher is better for the evaluator's SGS?
        # "starts the highest-priority eligible activity"
        # So higher score = higher priority.
        
        log_scores = log_features @ w
        # Convert to linear scores? 
        # The SGS implementation above takes `prio_scores` and pushes `-prio_scores`.
        # So it expects higher value = higher priority.
        # We can return log_scores directly as the priority scores.
        return simulate_sgs(log_scores)

    # SA Parameters
    T = 10.0
    T_min = 0.01
    alpha = 0.95
    max_iter = 1000
    
    current_w = w0.copy()
    current_score = objective(current_w)
    best_w = current_w.copy()
    best_score = current_score
    
    for i in range(max_iter):
        if time.time() - start_time > DEADLINE:
            break
        
        # Generate neighbor
        # Perturb one random exponent
        idx = random.randint(0, 3)
        delta = random.gauss(0, 1.0)
        new_w = current_w.copy()
        new_w[idx] += delta
        
        # Clip exponents to reasonable range
        new_w = np.clip(new_w, 0.1, 5.0)
        
        new_score = objective(new_w)
        
        # Acceptance criterion
        if new_score < current_score: # Lower makespan is better
            current_w = new_w
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_w = current_w.copy()
        else:
            # Accept with probability exp(-(new_score - current_score) / T)
            # Note: new_score > current_score (worse makespan)
            diff = new_score - current_score
            if diff == 0:
                accept = True
            else:
                accept = random.random() < np.exp(-diff / T)
            
            if accept:
                current_w = new_w
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_w = current_w.copy()
        
        T *= alpha
        if T < T_min:
            T = T_min

    # Final scores
    final_scores = log_features @ best_w
    return final_scores
# EVOLVE-BLOCK-END
