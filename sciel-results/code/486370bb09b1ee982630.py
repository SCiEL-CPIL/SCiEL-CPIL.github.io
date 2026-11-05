"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def priority(instance):
    """
    Computes a priority score for each activity to be used by a serial schedule-generation scheme.
    Higher scores indicate higher priority.
    Uses a combination of Successor Critical Path (SCP), Total Float (TF), and resource pressure.
    """
    start_time = time.time()
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if n == 0:
        return np.array([])
    
    # Get predecessors list
    pred_list = instance.get("predecessors", [])
    if not pred_list:
        pred_list = [[] for _ in range(n)]
        
    # Build successor list and indegree
    succ_list = [[] for _ in range(n)]
    indegree = [0] * n
    for i in range(n):
        for p in pred_list[i]:
            if p is not None and 0 <= p < n:
                succ_list[p].append(i)
                indegree[i] += 1

    # Topological Sort (Kahn's Algorithm)
    q = deque()
    for i in range(n):
        if indegree[i] == 0:
            q.append(i)
            
    topo_order = []
    # Use a copy of indegree for topo sort to not mess up original if needed elsewhere (though we don't use it later)
    indegree_copy = indegree[:]
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in succ_list[u]:
            indegree_copy[v] -= 1
            if indegree_copy[v] == 0:
                q.append(v)
                
    if len(topo_order) < n:
        # Fallback for cycles: return durations as priority
        return durations.copy()

    # 1. Forward Pass: Earliest Start (ES) and Earliest Finish (EF)
    ES = np.zeros(n)
    EF = np.zeros(n)
    for u in topo_order:
        EF[u] = ES[u] + durations[u]
        for v in succ_list[u]:
            if EF[u] > ES[v]:
                ES[v] = EF[u]
                
    # Project Duration (Critical Path Length ignoring resources)
    C = np.max(EF) if n > 0 else 0.0

    # 2. Backward Pass: Latest Finish (LF) and Latest Start (LS)
    LF = np.full(n, C)
    LS = np.full(n, C)
    for u in reversed(topo_order):
        if not succ_list[u]:
            LF[u] = C
        else:
            min_succ_ls = C
            for v in succ_list[u]:
                if LS[v] < min_succ_ls:
                    min_succ_ls = LS[v]
            LF[u] = min_succ_ls
        LS[u] = LF[u] - durations[u]

    # 3. Total Float (TF)
    TF = LS - ES
    TF = np.maximum(TF, 0)

    # 4. Successor Critical Path (SCP)
    # SCP[u] = duration[u] + max(SCP[v] for v in succ_list[u])
    SCP = np.zeros(n)
    for u in reversed(topo_order):
        if not succ_list[u]:
            SCP[u] = durations[u]
        else:
            max_scp_succ = 0
            for v in succ_list[u]:
                if SCP[v] > max_scp_succ:
                    max_scp_succ = SCP[v]
            SCP[u] = durations[u] + max_scp_succ

    # 5. Predecessor Critical Path (PCP)
    # PCP[u] = duration[u] + max(PCP[p] for p in pred_list[u])
    # This helps identify activities that are late in the project but have long paths to them
    PCP = np.zeros(n)
    for u in topo_order:
        if not pred_list[u]:
            PCP[u] = durations[u]
        else:
            max_pcp_pred = 0
            for p in pred_list[u]:
                if PCP[p] > max_pcp_pred:
                    max_pcp_pred = PCP[p]
            PCP[u] = durations[u] + max_pcp_pred

    # 6. Calculate Base Priority: SCP / (TF + epsilon)
    epsilon = 1e-6
    base_scores = SCP / (TF + epsilon)

    # 7. Incorporate Resource Demand
    demands = instance.get("demands")
    capacities = instance.get("capacities")
    
    pressure = np.zeros(n)
    if demands is not None and capacities is not None:
        try:
            D = np.asarray(demands, dtype=float)
            C_cap = np.asarray(capacities, dtype=float)
            # Handle shape issues
            if D.ndim == 1:
                D = D.reshape(-1, 1)
            if C_cap.ndim == 0:
                C_cap = np.array([C_cap.item()])
            
            # Avoid division by zero in capacities
            C_cap_safe = np.where(C_cap == 0, 1.0, C_cap)
            
            # Ensure dimensions match
            if D.shape[1] >= C_cap.shape[0]:
                # Sum of demand/capacity for each resource
                pressure = np.sum(D[:, :len(C_cap)] / C_cap_safe, axis=1)
            else:
                # If demands has fewer columns than capacities, pad or handle
                # Usually demands matches capacities
                pressure = np.sum(D / C_cap_safe[:D.shape[1]], axis=1)
            
            # Normalize pressure to have mean 0 and std 1? Or just scale.
            # Let's normalize to [0, 1] range roughly for stability
            if np.max(pressure) > 0:
                pressure = pressure / np.max(pressure)
                
        except (ValueError, TypeError, IndexError, AttributeError):
            pressure = np.zeros(n)

    # 8. Combine Scores
    # Strategy: 
    # High SCP is good (long tail).
    # Low TF is good (critical).
    # High Pressure is good (bottleneck usage).
    # We want to boost high pressure activities that are critical (low TF).
    
    # Score = Base_Score * (1 + alpha * pressure)
    # But we also want to consider PCP? 
    # Usually SCP is the dominant factor.
    # Let's try a weighted sum or product.
    
    # Normalize base_scores to be comparable to pressure?
    # Base scores can be large (time units). Pressure is [0,1].
    # Let's normalize base_scores to [0,1] as well for mixing?
    # Or just add a small term.
    
    # Alternative: Score = SCP / (TF + 1) + beta * pressure * SCP / (max_SCP + 1)
    
    # Let's stick to the multiplicative boost which worked well before but tune it.
    # The previous version used (1.0 + pressure).
    # Maybe we should weight it by criticality?
    # Critical activities (TF=0) should have pressure matter more?
    # If TF is high, pressure doesn't matter as much because we can schedule it later.
    # So, weight = 1 / (1 + TF) ?
    
    # Let's try: Score = SCP / (TF + eps) * (1 + pressure * (1 / (1 + TF)))
    # This boosts pressure for critical tasks more than for non-critical ones.
    
    # However, if TF is 0, 1/(1+0) = 1. If TF is large, it's small.
    # This seems reasonable.
    
    # Let's also consider that high PCP might indicate it's hard to start? No, PCP is time from start.
    # High PCP means it's deep in the project.
    
    # Let's refine the formula:
    # Primary: SCP / (TF + 1)  (Standard CPM priority)
    # Secondary: Pressure
    
    # We want to maximize the score for the most "urgent" activity.
    # Urgent = Long tail (SCP) + Critical (Low TF) + Resource Heavy (Pressure)
    
    # Let's use:
    # score = (SCP / (TF + 1)) * (1 + pressure)
    # This is what the current code does.
    
    # Let's try adding PCP to the numerator?
    # No, SCP is the path from node to end. PCP is start to node.
    # Total path length through node = PCP + SCP - duration?
    # Critical path length C.
    # If PCP + SCP > C, it's on critical path? No, that's not right.
    # TF = C - PCP - SCP + duration? 
    # Actually, TF = LS - ES = (LF - dur) - ES.
    # LF = min(LS of succs).
    # It's safer to stick to TF.
    
    # Let's try a different combination:
    # Score = SCP + pressure * SCP / (TF + 1)
    # This adds a term that is 0 if pressure is 0.
    # If pressure is high and TF is low, this term is large.
    
    # Let's try:
    # score = SCP / (TF + 1)
    # score += 0.5 * pressure * (SCP / (TF + 1))  # 50% boost for max pressure
    # This is equivalent to score * (1 + 0.5 * pressure)
    
    # The current code uses (1 + pressure). Maybe 1.5 or 2.0?
    # Or maybe we should normalize SCP first?
    
    max_scp = np.max(SCP)
    if max_scp > 0:
        norm_scp = SCP / max_scp
    else:
        norm_scp = SCP
        
    max_tf = np.max(TF)
    
    # Let's try a linear combination of normalized components
    # P1: Normalized SCP
    # P2: Normalized Criticality (1 - TF/max_tf)
    # P3: Normalized Pressure
    
    # Score = w1*P1 + w2*P2 + w3*P3
    
    # But SCP and Criticality are correlated.
    
    # Let's stick to the proven CPM formula but enhance the pressure term.
    # The pressure term should be scaled so it doesn't dominate SCP entirely.
    
    # Let's calculate a "Resource Weight"
    # If an activity uses 100% of a resource, it's very heavy.
    
    # Let's try:
    # score = SCP / (TF + 1)
    # score = score * (1 + 0.5 * pressure)
    
    # Let's also consider the number of successors?
    # Activities with many successors might be bottlenecks?
    # Not necessarily.
    
    # Let's try to be more aggressive with pressure for critical activities.
    # criticality = 1 / (1 + TF)
    # score = SCP * criticality * (1 + pressure)
    
    # This is slightly different from SCP / (TF+1).
    # SCP * (1/(1+TF)) vs SCP / (TF+1). Same thing.
    
    # So the formula is:
    scores = (SCP / (TF + 1.0)) * (1.0 + pressure)
    
    # Let's add a small bonus for high PCP?
    # If PCP is high, it means the activity is late in the sequence.
    # Usually we want to schedule early critical path activities first.
    # So high PCP is bad for priority?
    # Actually, if it's late, it might not be eligible yet.
    # Priority is only applied to eligible activities.
    # So PCP might not be as useful for eligible activities.
    
    # Let's check if there are any other improvements.
    # What if we use the ratio of demand to capacity for the *most constrained* resource?
    # max_pressure = max(D / C_cap, axis=1)
    # This might be more sensitive to bottlenecks.
    
    if demands is not None and capacities is not None:
        try:
            D = np.asarray(demands, dtype=float)
            C_cap = np.asarray(capacities, dtype=float)
            if D.ndim == 1:
                D = D.reshape(-1, 1)
            if C_cap.ndim == 0:
                C_cap = np.array([C_cap.item()])
            C_cap_safe = np.where(C_cap == 0, 1.0, C_cap)
            
            if D.shape[1] >= C_cap.shape[0]:
                D_sub = D[:, :len(C_cap)]
            else:
                D_sub = D
            
            ratios = D_sub / C_cap_safe
            max_pressure = np.max(ratios, axis=1)
            
            # Normalize max_pressure
            if np.max(max_pressure) > 0:
                max_pressure = max_pressure / np.max(max_pressure)
            
            # Combine sum_pressure and max_pressure?
            # max_pressure identifies the single most constraining resource.
            # Let's use max_pressure instead of sum_pressure?
            # Or average?
            
            # Let's try max_pressure.
            scores = (SCP / (TF + 1.0)) * (1.0 + max_pressure)
            
        except (ValueError, TypeError, IndexError, AttributeError):
            pass

    # Deadline guard
    if time.time() - start_time > 110:
        return scores

    return scores

# EVOLVE-BLOCK-END
