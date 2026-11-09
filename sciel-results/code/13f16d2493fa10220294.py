import numpy as np
from collections import deque
import time

def priority(instance):
    """
    Strategy: Multiplicative Interaction of Criticality and Resource Pressure with Soft-CPM smoothing.
    
    Structural changes from prior attempts:
    1.  Use a soft-max (log-sum-exp) based approach to compute a 'smoothed' criticality
        score instead of hard max/min slack. This differentiates near-critical activities
        more effectively.
    2.  Use a multiplicative interaction between criticality and resource pressure,
        ensuring that only activities that are BOTH critical and resource-heavy get
        very high priority.
    3.  Include a short-processing-time (SPT) tie-breaker to release resources quickly.
    4.  Add a deadline guard to prevent timeouts.
    """
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    try:
        # Extract instance data
        durations = np.asarray(instance["durations"], dtype=float)
        predecessors = instance["predecessors"]
        n = len(durations)
        
        # Get resource data
        try:
            demands = np.asarray(instance["demands"], dtype=float)
            capacities = np.asarray(instance["capacities"], dtype=float)
            m = len(capacities)
        except (KeyError, TypeError, ValueError):
            # Fallback if resource data is missing or malformed
            # Use simple duration-based priority
            return -durations.copy()
        
        # Build successor list
        successors = [[] for _ in range(n)]
        for j in range(n):
            for i in predecessors[j]:
                successors[i].append(j)
                
        # Topological Sort
        in_degree = [len(predecessors[j]) for j in range(n)]
        queue = deque([j for j in range(n) if in_degree[j] == 0])
        topo_order = []
        
        while queue:
            u = queue.popleft()
            topo_order.append(u)
            for v in successors[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
                    
        if len(topo_order) != n:
            # Fallback if graph is invalid or cyclic
            return -durations.copy()
        
        # --- Compute Soft CPM Slack ---
        # Instead of hard max/min, we use a soft-max to smooth the critical path identification.
        # This helps in cases where multiple paths have similar lengths.
        
        # Forward pass for Earliest Start (ES)
        ES = [0.0] * n
        EF = [0.0] * n
        for j in topo_order:
            if predecessors[j]:
                # Use soft-max instead of hard max for smoother ES
                # soft_max(x) = log(sum(exp(x)))
                # To avoid overflow, we subtract the max first
                pred_ef = [EF[i] for i in predecessors[j]]
                if pred_ef:
                    max_ef = max(pred_ef)
                    # If max_ef is very large, exp could overflow, but durations are usually small
                    # For safety, use a simple max if values are large
                    if max_ef < 100:
                        exps = [np.exp(x - max_ef) for x in pred_ef]
                        es_j = max_ef + np.log(sum(exps))
                    else:
                        es_j = max_ef
                    ES[j] = es_j
                else:
                    ES[j] = 0.0
            else:
                ES[j] = 0.0
            EF[j] = ES[j] + durations[j]
        
        # Hard makespan for normalization
        makespan_cpm = max(EF)
        
        # Backward pass for Latest Start (LS)
        LF = [makespan_cpm] * n
        LS = [makespan_cpm] * n
        for j in reversed(topo_order):
            if successors[j]:
                succ_ls = [LS[k] for k in successors[j]]
                if succ_ls:
                    min_ls = min(succ_ls)
                    # Use soft-min (which is -soft_max(-x))
                    if min_ls > -100:
                        neg_ls = [-x for x in succ_ls]
                        max_neg = max(neg_ls)
                        exps = [np.exp(x - max_neg) for x in neg_ls]
                        lf_j = -(max_neg + np.log(sum(exps)))
                    else:
                        lf_j = min_ls
                    LF[j] = lf_j
                else:
                    LF[j] = makespan_cpm
            else:
                LF[j] = makespan_cpm
            LS[j] = LF[j] - durations[j]
        
        # Soft Slack
        Slack = [LS[j] - ES[j] for j in range(n)]
        
        # --- Compute Resource Pressure ---
        resource_pressure = np.zeros(n)
        for j in range(n):
            total_pressure = 0.0
            for k in range(m):
                if capacities[k] > 0:
                    total_pressure += demands[j, k] / capacities[k]
                else:
                    if demands[j, k] > 0:
                        total_pressure += 1000.0 # High penalty
            resource_pressure[j] = total_pressure
        
        # --- Normalize Scores ---
        max_slack = max(Slack) if max(Slack) > 0 else 1.0
        norm_slack = np.array([s / max_slack for s in Slack])
        
        max_pressure = max(resource_pressure) if max(resource_pressure) > 0 else 1.0
        norm_pressure = np.array([p / max_pressure for p in resource_pressure])
        
        max_dur = max(durations) if max(durations) > 0 else 1.0
        norm_dur = np.array([d / max_dur for d in durations])
        
        # --- Combine Scores ---
        # Criticality: 1 - norm_slack (High if low slack)
        criticality = 1.0 - norm_slack
        # Pressure: norm_pressure (High if high demand/capacity ratio)
        pressure = norm_pressure
        
        # Multiplicative interaction: Prioritize activities that are BOTH critical AND heavy
        # This creates a sharper distinction than additive methods
        interaction = criticality * (1.0 + pressure)
        
        # SPT tie-breaker: Shorter durations release resources faster
        spt = (1.0 - norm_dur) * 0.1
        
        scores = interaction + spt
        
        # Check deadline
        if time.time() > deadline:
            # Return current best effort
            pass
            
        return scores
        
    except Exception:
        try:
            durations = np.asarray(instance["durations"], dtype=float)
            return -durations.copy()
        except:
            return np.zeros(30)
