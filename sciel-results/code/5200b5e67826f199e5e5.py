"""RCPSP priority via Resource Unlock Heuristic."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque

def priority(instance):
    """
    Compute a priority score for each activity in a Resource-Constrained
    Project Scheduling Problem (RCPSP) instance.
    
    Strategy:
    1. Compute Total Float (TF) via forward/backward pass.
    2. Compute 'Resource Unlock Value': Estimate how much scheduling this activity
       enables the critical path of successors. We approximate this by summing
       the durations of the most critical successors (based on their own TF).
    3. Combine TF (lower is better) and Unlock Value (higher is better) into a
       single priority score.
    """
    # --- Extract instance components ---
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if n == 0:
        return np.array([])

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

    # Build successor lists if not provided
    if succs is None:
        succs = [[] for _ in range(n)]
        for j in range(n):
            p = preds[j]
            if isinstance(p, (list, tuple, set, np.ndarray)):
                for i in p:
                    idx = int(i)
                    if 0 <= idx < n:
                        succs[idx].append(j)
            elif isinstance(p, int):
                idx = int(p)
                if 0 <= idx < n:
                    succs[idx].append(j)
        
        # Ensure unique successors
        succs = [list(set(s)) for s in succs]

    # Build pred sets for fast lookup
    pred_set = [set() for _ in range(n)]
    for j in range(n):
        p = preds[j]
        if isinstance(p, (list, tuple, set, np.ndarray)):
            for i in p:
                idx = int(i)
                if 0 <= idx < n:
                    pred_set[j].add(idx)
        elif isinstance(p, int):
            idx = int(p)
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
        # Fallback if cycle or error
        for i in range(n):
            if i not in topo_order:
                topo_order.append(i)

    # --- Forward pass (Earliest Start/Finish) ---
    ES = np.zeros(n)
    EF = np.zeros(n)
    for u in topo_order:
        for v in succs[u]:
            if EF[u] > ES[v]:
                ES[v] = EF[u]
        EF[u] = ES[u] + durations[u]

    makespan_forward = float(np.max(EF)) if n > 0 else 0.0

    # --- Backward pass (Latest Start/Finish) ---
    LS = np.zeros(n)
    LF = np.zeros(n)
    for u in reversed(topo_order):
        if len(succs[u]) == 0:
            LF[u] = makespan_forward
        else:
            LF[u] = float(np.min([LS[v] for v in succs[u]]))
        LS[u] = LF[u] - durations[u]

    # Total Float
    TF = LS - ES

    # --- Compute Resource Unlock Value ---
    # Hypothesis: An activity has high "unlock value" if it precedes activities
    # that are themselves critical (low TF) or have high resource demands.
    # Since we don't have resource demands in the simplified logic above (we skipped extraction for speed),
    # we rely on the structural criticality of successors.
    
    # Let's extract demands if available to enhance the heuristic, as per the "Resource Unlock" directive.
    demands = None
    for key in ["demands", "resource_demands", "resources", "dem", "resource"]:
        if key in instance:
            demands = instance[key]
            break
    
    # If demands are available, compute a "Weight" for each node based on resource consumption
    # and propagate this weight backwards to estimate "unlock value".
    
    node_weight = np.ones(n) # Default weight 1
    
    if demands is not None:
        demands_arr = np.asarray(demands, dtype=float)
        if demands_arr.ndim == 1:
            demands_arr = demands_arr.reshape(n, 1)
        # Sum of demands as a proxy for resource "heaviness"
        # Normalize to [0, 1] range roughly
        max_demand_sum = np.max(np.sum(demands_arr, axis=1))
        if max_demand_sum > 0:
            node_weight = np.sum(demands_arr, axis=1) / max_demand_sum
        else:
            node_weight = np.ones(n)

    # Calculate "Unlock Value" for each node:
    # Sum of (Successor's Weight * Successor's Duration) / (1 + Successor's TF)
    # This prioritizes unlocking successors that are heavy and critical.
    
    unlock_value = np.zeros(n)
    
    # Process in reverse topological order to compute downstream impact
    # We want to know: if I am delayed, how much does it push back the critical path?
    # Approximation: Sum over successors of (Criticality_Succ * Weight_Succ * Dur_Succ)
    
    # To be more accurate, we can propagate a "Downstream Criticality" score.
    # DownstreamCrit[i] = Max(DownstreamCrit[j] + Dur[j]) for j in Succs[i]
    # But we want a "Unlock" metric which is about *enabling* work.
    
    # Let's use a simpler, robust metric:
    # Priority = (1 / (1 + TF)) + (Sum of Successor Durations * Successor Weight) / (1 + Avg Successor TF)
    
    for u in reversed(topo_order):
        if len(succs[u]) == 0:
            unlock_value[u] = 0.0
        else:
            # Sum of weighted durations of successors
            # Weighted by how critical they are (inverse TF)
            sum_weighted_dur = 0.0
            count = 0
            for v in succs[u]:
                # If successor is critical (low TF), it contributes more to unlock value
                # Use 1/(1+TF) as criticality factor
                crit_factor = 1.0 / (1.0 + TF[v])
                sum_weighted_dur += durations[v] * node_weight[v] * crit_factor
                count += 1
            
            # Normalize by count to avoid biasing nodes with many successors
            if count > 0:
                unlock_value[u] = sum_weighted_dur / count
            else:
                unlock_value[u] = 0.0

    # --- Final Score Calculation ---
    # We want higher score = higher priority.
    # TF: Lower is better. So use 1/(1+TF) or similar.
    # Unlock: Higher is better.
    
    # Normalize TF part to [0, 1]
    max_tf = np.max(TF) + 1e-9
    tf_score = 1.0 - (TF / max_tf)  # 1.0 for critical, 0.0 for max float
    
    # Normalize Unlock part to [0, 1]
    max_unlock = np.max(unlock_value) + 1e-9
    unlock_score = unlock_value / max_unlock
    
    # Combine: 
    # We generally trust Total Float (Standard RCPSP heuristic) more, 
    # but add Unlock Value to break ties or prioritize resource-heavy critical paths.
    # Weight: 0.7 * TF + 0.3 * Unlock
    
    final_scores = 0.7 * tf_score + 0.3 * unlock_score
    
    # Ensure finite
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return final_scores
# EVOLVE-BLOCK-END
