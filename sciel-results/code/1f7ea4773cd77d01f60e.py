"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_cpm_priorities(instance):
    """Compute CPM-based priorities: forward/backward pass, float, resource pressure."""
    N = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # (N, R)
    capacities = np.asarray(instance["capacities"], dtype=float)  # (R,)
    R = len(capacities)
    
    # Build predecessor/successor lists
    preds = instance.get("predecessors", None)
    if preds is None:
        # Try alternate key names
        for key in ["preds", "predecessor", "predecessors_list"]:
            if key in instance:
                preds = instance[key]
                break
        if preds is None:
            preds = [[] for _ in range(N)]
    else:
        # Normalize: ensure list of lists
        if isinstance(preds, np.ndarray):
            preds = preds.tolist()
        preds = [list(p) for p in preds]
    
    succs = [[] for _ in range(N)]
    for j in range(N):
        for i in preds[j]:
            succs[i].append(j)
    
    # Topological order
    in_degree = np.zeros(N, dtype=int)
    for j in range(N):
        for i in preds[j]:
            in_degree[j] += 1
    
    from collections import deque
    queue = deque([i for i in range(N) if in_degree[i] == 0])
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in succs[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Forward pass: ES, EF
    ES = np.zeros(N, dtype=float)
    EF = np.zeros(N, dtype=float)
    for u in topo:
        if preds[u]:
            ES[u] = max(EF[i] for i in preds[u])
        EF[u] = ES[u] + durations[u]
    
    makespan_cpm = max(EF) if N > 0 else 0.0
    
    # Backward pass: LS, LF
    LS = np.zeros(N, dtype=float)
    LF = np.zeros(N, dtype=float)
    for u in reversed(topo):
        if succs[u]:
            LF[u] = min(LS[v] for v in succs[u])
        else:
            LF[u] = makespan_cpm
        LS[u] = LF[u] - durations[u]
    
    # Total float
    TF = LS - ES  # Total float
    
    # Resource pressure: for each activity, max over resources of (demand / capacity)
    # Higher pressure = more resource-constrained
    if R > 0 and np.all(capacities > 0):
        pressure = demands / capacities  # (N, R)
        res_pressure = np.max(pressure, axis=1)  # (N,)
    else:
        res_pressure = np.zeros(N, dtype=float)
    
    # Successor count (how many activities directly or indirectly depend on this)
    # Use topological order to count reachable successors
    succ_count = np.zeros(N, dtype=int)
    for u in reversed(topo):
        succ_count[u] = len(succs[u]) + sum(succ_count[v] for v in succs[u])
    
    # Depth from start (number of predecessors in path)
    depth = np.zeros(N, dtype=int)
    for u in topo:
        if preds[u]:
            depth[u] = max(depth[i] for i in preds[u]) + 1
    
    # Build priority: we want higher priority = schedule sooner
    # Components:
    # 1. Low float is critical -> high priority: use -TF (normalized)
    # 2. High resource pressure -> high priority
    # 3. High successor count -> high priority (unlocks more work)
    # 4. Long duration on critical path -> high priority
    
    # Normalize each component to [0, 1]
    def normalize(x):
        x = np.asarray(x, dtype=float)
        rng = x.max() - x.min()
        if rng < 1e-12:
            return np.zeros_like(x)
        return (x - x.min()) / rng
    
    norm_float_inv = normalize(-TF)  # High when float is low (critical)
    norm_pressure = normalize(res_pressure)
    norm_succ = normalize(succ_count)
    norm_dur = normalize(durations)
    norm_depth_inv = normalize(-depth)  # Earlier in chain = more critical
    
    # Weighted combination
    priority = (
        0.35 * norm_float_inv +
        0.25 * norm_pressure +
        0.20 * norm_succ +
        0.10 * norm_dur +
        0.10 * norm_depth_inv
    )
    
    return priority


def priority(instance):
    """
    Returns a priority score for each activity.
    Higher priority = scheduled sooner in serial SGS.
    Uses CPM-based analysis with resource pressure and successor impact.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    try:
        N = len(instance["durations"])
        if N == 0:
            return np.array([], dtype=float)
        
        # Try CPM-based priority
        prio = _compute_cpm_priorities(instance)
        
        # Ensure it's a numpy array of correct length
        prio = np.asarray(prio, dtype=float)
        if len(prio) != N:
            # Fallback to duration-based
            prio = np.asarray(instance["durations"], dtype=float)
        
        return prio
        
    except Exception:
        # Fallback: duration-based priority
        try:
            return np.asarray(instance["durations"], dtype=float)
        except Exception:
            N = len(instance.get("durations", []))
            return np.zeros(N, dtype=float)
# EVOLVE-BLOCK-END
