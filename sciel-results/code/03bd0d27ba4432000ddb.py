"""Improved RCPSP priority function with dynamic resource-contention-aware scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def priority(instance):
    """
    Compute priority scores for each activity using a combination of:
    1. Static CPM criticality (SCP / (TF+1))
    2. Dynamic resource contention detected via a lightweight greedy simulation
    3. Smooth soft-max resource pressure
    
    Higher scores = higher priority.
    """
    start_time = time.time()
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if n == 0:
        return np.array([])
    
    pred_list = instance.get("predecessors", [])
    if not pred_list:
        pred_list = [[] for _ in range(n)]
    
    succ_list = [[] for _ in range(n)]
    indegree = [0] * n
    for i in range(n):
        for p in pred_list[i]:
            if p is not None and 0 <= p < n:
                succ_list[p].append(i)
                indegree[i] += 1

    q = deque()
    for i in range(n):
        if indegree[i] == 0:
            q.append(i)
            
    topo_order = []
    indegree_copy = indegree[:]
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in succ_list[u]:
            indegree_copy[v] -= 1
            if indegree_copy[v] == 0:
                q.append(v)
                
    if len(topo_order) < n:
        return durations.copy()

    # Forward pass
    ES = np.zeros(n)
    EF = np.zeros(n)
    for u in topo_order:
        EF[u] = ES[u] + durations[u]
        for v in succ_list[u]:
            if EF[u] > ES[v]:
                ES[v] = EF[u]
                
    C = np.max(EF) if n > 0 else 0.0

    # Backward pass
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

    TF = np.maximum(LS - ES, 0)

    # SCP
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

    # Base CPM score
    base_scores = SCP / (TF + 1.0)

    # Resource data
    demands = instance.get("demands")
    capacities = instance.get("capacities")
    
    num_resources = 0
    D_mat = None
    C_cap = None
    if demands is not None and capacities is not None:
        try:
            D = np.asarray(demands, dtype=float)
            C_cap = np.asarray(capacities, dtype=float)
            if D.ndim == 1:
                D = D.reshape(-1, 1)
            if C_cap.ndim == 0:
                C_cap = np.array([C_cap.item()])
            num_resources = C_cap.shape[0]
            D_mat = D[:, :num_resources] if D.shape[1] >= num_resources else D
            C_cap_safe = np.where(C_cap == 0, 1.0, C_cap)
        except (ValueError, TypeError, IndexError, AttributeError):
            D_mat = None
            C_cap = None
            num_resources = 0

    # Dynamic resource contention simulation
    # Simulate a greedy schedule using current base_scores as priority,
    # track how often each activity causes resource blocking
    contention = np.zeros(n)
    if D_mat is not None and C_cap is not None and num_resources > 0:
        try:
            # Greedy simulation: schedule activities by base_scores among eligible
            remaining = set(range(n))
            in_degree_sim = indegree[:]
            scheduled_time = {}
            resource_usage = [np.zeros(num_resources)]  # list of time slots
            
            # Use a simpler approach: track which activities "blocked" others
            # by measuring resource conflicts at each scheduling step
            
            eligible = [i for i in range(n) if in_degree_sim[i] == 0]
            order = []
            
            while eligible:
                # Pick highest base_score among eligible
                best = max(eligible, key=lambda x: base_scores[x])
                order.append(best)
                remaining.discard(best)
                
                # Update successors
                for v in succ_list[best]:
                    in_degree_sim[v] -= 1
                    if in_degree_sim[v] == 0:
                        eligible.append(v)
                
                if not eligible:
                    break
                    
            # Now simulate resource usage to detect contention
            # Track when each activity starts and its resource demand
            # Contention = for each activity, how much it contributes to
            # the peak resource usage during its execution
            
            if len(order) == n:
                # Assign start times based on the greedy order
                # Use a simplified event-based simulation
                active = []  # (finish_time, activity_id)
                start_times = {}
                current_time = 0
                
                # Process in order, but respect resource constraints
                # This is a simplified greedy that may not be fully feasible,
                # but gives us contention signals
                for act in order:
                    earliest = current_time
                    for p in pred_list[act]:
                        if p in start_times:
                            p_end = start_times[p] + durations[p]
                            if p_end > earliest:
                                earliest = p_end
                    
                    # Check resource availability
                    while True:
                        # Remove finished activities
                        still_active = []
                        for (ft, a) in active:
                            if ft > earliest:
                                still_active.append((ft, a))
                        active = still_active
                        
                        # Check if resources are available
                        can_start = True
                        for r in range(num_resources):
                            demand_r = D_mat[act, r] if r < D_mat.shape[1] else 0
                            if demand_r > 0:
                                used = 0
                                for (ft, a) in active:
                                    used += D_mat[a, r] if r < D_mat.shape[1] else 0
                                cap_r = C_cap[r] if r < len(C_cap) else 1.0
                                if used + demand_r > cap_r:
                                    can_start = False
                                    # Find earliest time an activity using this resource finishes
                                    for (ft, a) in active:
                                        if (D_mat[a, r] if r < D_mat.shape[1] else 0) > 0:
                                            if ft > earliest:
                                                earliest = ft
                                            elif ft == earliest:
                                                earliest = ft
                                    break
                        
                        if can_start:
                            break
                        # If no active activities conflict, we can start
                        if not active:
                            break
                    
                    start_times[act] = earliest
                    finish = earliest + durations[act]
                    active.append((finish, act))
                    if finish > current_time:
                        current_time = finish
                
                # Compute contention: for each activity, measure how much
                # resource it uses relative to capacity, weighted by how many
                # other activities are active at the same time
                for act in range(n):
                    if act not in start_times:
                        continue
                    st = start_times[act]
                    ft = st + durations[act]
                    
                    # Count how many activities overlap with this one
                    overlap_count = 0
                    for other in range(n):
                        if other == act:
                            continue
                        if other in start_times:
                            ost = start_times[other]
                            oft = ost + durations[other]
                            if st < oft and ft > ost:
                                overlap_count += 1
                    
                    # Resource intensity
                    if D_mat is not None and D_mat.shape[1] > 0:
                        intensity = np.sum(D_mat[act] / C_cap_safe) if D_mat.shape[1] >= num_resources else np.sum(D_mat[act] / C_cap_safe[:D_mat.shape[1]])
                    else:
                        intensity = 1.0
                    
                    # Contention score: high if this activity is resource-heavy
                    # AND many other activities are active at the same time
                    contention[act] = intensity * (1.0 + 0.3 * overlap_count)
                
                # Normalize contention
                max_cont = np.max(contention)
                if max_cont > 0:
                    contention = contention / max_cont
                    
        except (ValueError, TypeError, IndexError, AttributeError, ZeroDivisionError):
            contention = np.zeros(n)

    # Static resource pressure
    static_pressure = np.zeros(n)
    if D_mat is not None and C_cap is not None and num_resources > 0:
        try:
            C_cap_safe = np.where(C_cap == 0, 1.0, C_cap)
            ratios = D_mat / C_cap_safe
            # Soft-max: smooth approximation of max
            # Use log-sum-exp for smooth max
            max_ratio = np.max(ratios, axis=1)
            if np.max(max_ratio) > 0:
                static_pressure = max_ratio / np.max(max_ratio)
        except (ValueError, TypeError, IndexError, AttributeError):
            static_pressure = np.zeros(n)

    # Combine: dynamic contention is more informative than static pressure
    # Use a weighted blend
    # When contention is available (non-zero), weight it more heavily
    has_contention = np.any(contention > 0)
    
    if has_contention:
        # Blend: 60% dynamic contention, 40% static pressure
        combined_pressure = 0.6 * contention + 0.4 * static_pressure
    else:
        combined_pressure = static_pressure

    # Final score: base CPM * (1 + pressure boost)
    # The boost is stronger for critical activities (low TF)
    # criticality_weight = 1 / (1 + TF)
    criticality_weight = 1.0 / (1.0 + TF)
    
    # Score = base * (1 + combined_pressure * criticality_weight)
    # This makes resource pressure matter more for critical activities
    scores = base_scores * (1.0 + combined_pressure * criticality_weight)

    # Deadline guard
    if time.time() - start_time > 110:
        return scores

    return scores

# EVOLVE-BLOCK-END
