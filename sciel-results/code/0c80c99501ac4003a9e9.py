"""Resource-Constrained Project Scheduling: Hybrid Critical Path & Resource Pressure Priority.

Family: Static Heuristic (Hybrid of Total Float and Resource Intensity).
priority(instance) -> one score per activity (higher = scheduled earlier).
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 140.0  # safety margin under the 150 s wall-clock kill


def _topo_order(pred_lists, n):
    """Return a topological order (list of ints)."""
    indeg = np.zeros(n, dtype=int)
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in pred_lists[i]:
            succ[p].append(i)
            indeg[i] += 1
    stack = [i for i in range(n) if indeg[i] == 0]
    order = []
    while stack:
        u = stack.pop()
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    if len(order) != n:  # cycle guard: fall back to natural order
        order = list(range(n))
    return order, succ


def priority(instance):
    t0 = time.time()
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    preds = instance.get("predecessors", None)
    if preds is None:
        preds = [[] for _ in range(n)]
    pred_lists = []
    for i in range(n):
        p = preds[i]
        if p is None:
            p = []
        pred_lists.append([int(x) for x in p])

    order, succ = _topo_order(pred_lists, n)

    # 1. Compute Critical Path Metrics (Head and Tail)
    # Head: Longest path from source to node
    head = np.zeros(n, dtype=float)
    for u in order:
        best = 0.0
        for p in pred_lists[u]:
            best = max(best, head[p] + durations[p])
        head[u] = best

    # Tail: Longest path from node to sink
    tail = np.zeros(n, dtype=float)
    for u in reversed(order):
        best = 0.0
        for v in succ[u]:
            best = max(best, tail[v])
        tail[u] = durations[u] + best

    # Total Float (Slack) = (Head + Tail) - Project Duration
    proj_len = float(np.max(head + durations + tail)) if n else 1.0
    # Note: Standard float is usually defined relative to the critical path length.
    # Low float means high priority.
    # We want higher score for higher priority, so we invert float.
    # Score component 1: Criticality (High Tail + Low Float)
    # Using Tail as a proxy for importance, and (proj_len - (head+tail)) as slack.
    # Actually, simpler: Prioritize by Tail (longest remaining path).
    # And prioritize by Low Slack.
    slack = proj_len - (head + durations + tail)
    # Normalize slack to be positive (smaller is better)
    slack_norm = slack / (proj_len + 1e-9)

    # 2. Compute Resource Pressure
    demands = np.asarray(instance.get("demands", np.zeros((n, 0))), dtype=float)
    if demands.ndim == 1:
        demands = demands.reshape(-1, 1)
    capacities = np.asarray(instance.get("capacities", instance.get("resource_capacities", [])), dtype=float)
    if capacities.ndim == 0:
        capacities = capacities.reshape(1)
    nr = demands.shape[1] if demands.ndim == 2 else 0
    
    if nr == 0:
        res_pressure = np.zeros(n, dtype=float)
    else:
        if capacities.size < nr:
            capacities = np.full(nr, 1.0)
        cap = np.where(capacities > 0, capacities, 1.0)
        # Resource Intensity: Demand / Capacity
        # Higher intensity means the activity consumes a larger fraction of the resource.
        # Scheduling high-intensity activities early can help if they are long, 
        # but can also cause contention. 
        # A common heuristic is "Maximum Resource Intensity" (MRI).
        intensity = demands / cap[None, :]
        # Aggregate intensity across resources (e.g., max or sum)
        # Using sum to capture total load, or max for bottleneck.
        # Let's use a weighted sum where scarce resources (low cap) matter more.
        # Actually, just sum of (demand/capacity) is a good load indicator.
        res_pressure = np.sum(intensity, axis=1)
        
        # Normalize res_pressure
        max_pressure = np.max(res_pressure)
        if max_pressure > 0:
            res_pressure = res_pressure / max_pressure

    # 3. Combine Scores
    # We want to prioritize:
    # 1. Activities with high Tail (critical path)
    # 2. Activities with low Slack (less flexibility)
    # 3. Activities with high Resource Pressure (to handle scarcity early)
    
    # Normalize Tail
    max_tail = np.max(tail) if n > 0 else 1.0
    tail_norm = tail / (max_tail + 1e-9)
    
    # Invert Slack Norm (so low slack becomes high score)
    # slack_norm is 0..1 (approx). Low slack -> high priority.
    slack_score = 1.0 - slack_norm
    
    # Weights: Tail is usually the strongest predictor.
    # Slack is closely related to Tail/Head.
    # Resource Pressure is a tie-breaker or secondary factor.
    
    # Hybrid Score:
    # 0.6 * Tail + 0.3 * SlackScore + 0.1 * ResPressure
    score = 0.6 * tail_norm + 0.3 * slack_score + 0.1 * res_pressure
    
    # 4. Refinement: Non-linear sharpening to amplify differences
    # This helps distinguish between activities with similar scores
    # and pushes critical activities further ahead
    if n > 0:
        min_score = np.min(score)
        max_score = np.max(score)
        if max_score > min_score:
            # Normalize to [0, 1]
            score_norm = (score - min_score) / (max_score - min_score)
            # Apply power transformation to sharpen (exponent > 1 boosts high scores)
            # 1.5 is a moderate sharpening that preserves order
            score_sharp = score_norm ** 1.5
            # Re-scale back
            score = min_score + score_sharp * (max_score - min_score)
    
    return score

# EVOLVE-BLOCK-END
