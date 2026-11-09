import numpy as np
import time
from collections import deque

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

def _compute_min_required_capacities(instance):
    """
    Use a time-expanded network flow formulation to estimate the minimum
    resource capacity required for each activity to be scheduled without
    delay on its critical path. This gives us a 'bottleneck score' per activity.
    
    We model the problem as: given fixed start times (from CPM), what is the
    minimum capacity needed for each resource such that all activities can be
    scheduled at their earliest start times? The inverse of this required capacity
    indicates how much that activity constrains the resource profile.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]
    
    n = len(durations)
    n_resources = len(capacities) if capacities.ndim > 0 else 0
    
    if n == 0 or n_resources == 0:
        return np.zeros(n, dtype=float)
    
    # Build successor list
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
    
    # Topological sort
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    if len(topo_order) != n:
        return np.zeros(n, dtype=float)
    
    # Compute earliest start times (forward pass)
    earliest_start = np.zeros(n, dtype=float)
    for u in topo_order:
        if predecessors[u]:
            earliest_start[u] = max(earliest_start[p] + durations[p] for p in predecessors[u])
    
    # Compute latest start times (backward pass)
    project_end = max(earliest_start[i] + durations[i] for i in range(n))
    latest_start = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if not successors[u]:
            latest_start[u] = project_end - durations[u]
        else:
            latest_start[u] = min(latest_start[v] - durations[v] for v in successors[u])
    
    # For each resource, compute the minimum capacity required if all activities
    # are scheduled at their earliest start times.
    # Then compute a "slack" measure: how much each activity contributes to
    # the peak demand on each resource.
    
    # Time horizon
    max_time = int(np.ceil(project_end)) + 1
    
    # For each resource, build the demand profile over time
    # and compute the peak demand
    min_required = np.zeros(n_resources, dtype=float)
    
    for k in range(n_resources):
        if capacities[k] <= 0:
            continue
        
        # Build demand profile: for each time unit, sum demands of active activities
        # An activity i is active during [earliest_start[i], earliest_start[i] + durations[i])
        demand_profile = np.zeros(max_time + 1, dtype=float)
        
        for i in range(n):
            if demands[i, k] <= 0:
                continue
            start_t = int(earliest_start[i])
            end_t = int(earliest_start[i] + durations[i])
            if end_t > max_time:
                end_t = max_time
            for t in range(start_t, end_t):
                demand_profile[t] += demands[i, k]
        
        min_required[k] = np.max(demand_profile)
    
    # Now compute for each activity its contribution to the peak demand
    # across all resources. This is the "bottleneck score".
    bottleneck_score = np.zeros(n, dtype=float)
    
    for i in range(n):
        start_t = int(earliest_start[i])
        end_t = int(earliest_start[i] + durations[i])
        if end_t > max_time:
            end_t = max_time
        
        for k in range(n_resources):
            if capacities[k] <= 0 or demands[i, k] <= 0:
                continue
            
            # Rebuild demand profile for this resource (inefficient but correct)
            # Actually, let's compute it more efficiently using prefix sums
            
            # The activity i contributes demands[i, k] to each time unit in
            # [start_t, end_t). We want to know if this activity is part of
            # the peak demand.
            
            # A simpler approach: compute the peak demand with and without activity i
            # This is O(n * max_time * n_resources) which might be too slow.
            
            # Instead, use a heuristic: the activity's "bottleneck contribution" is
            # proportional to its demand relative to the minimum required capacity,
            # weighted by how tight the resource is.
            
            if min_required[k] > 1e-10:
                # How much of the peak demand does this activity contribute?
                # Approximate: demand[i,k] / min_required[k]
                bottleneck_score[i] += demands[i, k] / min_required[k]
    
    # Normalize
    max_score = np.max(bottleneck_score)
    if max_score > 1e-10:
        bottleneck_score /= max_score
    
    return bottleneck_score


def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher priority (score) is scheduled first among eligible activities.

    Design (CPM + resource-aware family, refined):
    1. Compute Critical Path Length (CPL): longest path from each activity to project end.
       Primary, dominant priority signal.
    2. Compute a smooth resource-contention signal using log-sum-exp soft-max over
       demand/capacity ratios. This accounts for multiple scarce resources.
    3. Compute a resource-weighted critical path tail to capture chains of
       resource-heavy activities.
    4. Small tie-break by duration and successor count for determinism.

    The combination uses a multiplicative boost: critical-path structure is the
    primary driver, and resource contention amplifies it for activities that
    both sit on long paths AND consume scarce resources.
    """
    if time.time() - _START_TIME > _DEADLINE:
        return np.asarray(instance["durations"], dtype=float)

    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]

    n = len(durations)
    if n == 0:
        return np.array([])

    # Build successor list and in-degrees
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1

    # Topological sort (Kahn's algorithm)
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(topo_order) != n:
        # Fallback: just use durations as priority
        return durations.copy()

    rev_topo = list(reversed(topo_order))

    # Compute Critical Path Length (CPL) - longest path from activity to project end
    cpl = np.zeros(n, dtype=float)
    for u in rev_topo:
        if not successors[u]:
            cpl[u] = durations[u]
        else:
            cpl[u] = durations[u] + max(cpl[v] for v in successors[u])

    if time.time() - _START_TIME > _DEADLINE:
        return cpl.copy()

    # Number of direct successors - tie-breaker for bottleneck potential
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)

    # Resource pressure signals
    n_resources = len(capacities) if capacities.ndim > 0 else 0
    sum_ratio = np.zeros(n, dtype=float)
    max_ratio = np.zeros(n, dtype=float)
    soft_max_ratio = np.zeros(n, dtype=float)

    if n_resources > 0:
        safe_cap = np.where(capacities > 0, capacities, 1.0)
        # ratios[i, k] = demands[i, k] / safe_cap[k]
        ratios = demands / safe_cap  # shape (n, n_resources)

        for i in range(n):
            row = ratios[i]
            # Only consider resources with positive capacity and positive demand
            mask = (capacities > 0) & (demands[i] > 0)
            if np.any(mask):
                active = row[mask]
                sum_ratio[i] = np.sum(active)
                max_ratio[i] = np.max(active)
                # Log-sum-exp soft-max with temperature
                temp = 0.15
                m = float(np.max(active))
                if m > 1e-10:
                    soft_max_ratio[i] = m + temp * np.log(np.sum(np.exp((active - m) / temp)))
                else:
                    soft_max_ratio[i] = 0.0

    if time.time() - _START_TIME > _DEADLINE:
        return cpl.copy()

    # Resource-weighted critical path tail:
    # For each activity, the longest path where edge weights are boosted by
    # the resource pressure of the successor. Captains chains of resource-heavy
    # activities as more constraining.
    resource_tail = np.zeros(n, dtype=float)
    if n_resources > 0:
        for u in rev_topo:
            if not successors[u]:
                resource_tail[u] = durations[u]
            else:
                max_rt = 0.0
                for v in successors[u]:
                    w = 1.0 + 0.3 * sum_ratio[v]
                    cand = durations[v] * w + resource_tail[v]
                    if cand > max_rt:
                        max_rt = cand
                resource_tail[u] = durations[u] + max_rt

    if time.time() - _START_TIME > _DEADLINE:
        return cpl.copy()

    # Normalize components (max-normalization for scale invariance)
    def safe_norm(arr):
        m = float(np.max(arr))
        if m <= 1e-10:
            return np.zeros_like(arr, dtype=float)
        return arr / m

    norm_cpl = safe_norm(cpl)
    norm_rt = safe_norm(resource_tail)
    norm_sum = safe_norm(sum_ratio)
    norm_max = safe_norm(max_ratio)
    norm_soft = safe_norm(soft_max_ratio)
    norm_dur = safe_norm(durations)
    norm_succ = safe_norm(succ_count)

    # Compute bottleneck score from time-expanded network flow
    if time.time() - _START_TIME < _DEADLINE - 5:
        bottleneck_score = _compute_min_required_capacities(instance)
        norm_bottleneck = safe_norm(bottleneck_score)
    else:
        norm_bottleneck = np.zeros(n, dtype=float)

    # Combine:
    # Primary structural signal: blend of pure CPL and resource-weighted tail
    cp_combined = 0.7 * norm_cpl + 0.3 * norm_rt

    # Resource contention: blend of soft-max (smooth max) and sum
    resource_term = 0.5 * norm_soft + 0.3 * norm_max + 0.2 * norm_sum

    # Multiplicative boost: critical path primary, resource contention secondary
    score = cp_combined * (1.0 + resource_term)

    # Add bottleneck score as an additional resource-aware signal
    score = score + 0.15 * norm_bottleneck

    # Small additive tie-breakers
    score = score + 0.05 * norm_dur + 0.03 * norm_succ

    return score
