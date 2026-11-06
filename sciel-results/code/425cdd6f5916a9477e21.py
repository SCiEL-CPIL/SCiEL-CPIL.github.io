"""Rewritten program for Resource-Constrained Project Scheduling.

Uses a Critical Path Method (CPM) based heuristic.
Priority is determined by the activity's duration divided by its total float + 1.
This prioritizes activities that are both long and close to the critical path (low slack).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(instance):
    start_time = time.time()
    # Deadline guard: 120s limit, allow 10s margin
    deadline = start_time + 110.0
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Predecessors
    preds_raw = instance.get("predecessors", None)
    if preds_raw is None:
        preds_raw = [[] for _ in range(n)]
        
    # Normalize predecessors to lists of ints
    preds = []
    for i in range(n):
        p = preds_raw[i]
        if p is None:
            p = []
        elif isinstance(p, (list, tuple, np.ndarray)):
            p = list(p)
        else:
            # If it's a single int, wrap it? Or assume list? 
            # Usually it's a list. If it's an int, it's a single pred.
            p = [p]
        # Ensure ints
        preds.append([int(x) for x in p])
        
    # Build Successors list
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            if 0 <= p < n:
                succs[p].append(i)
                
    # Forward Pass: Calculate Earliest Start (ES) and Earliest Finish (EF)
    es = np.zeros(n, dtype=float)
    ef = np.zeros(n, dtype=float)
    
    # Simple iterative relaxation for ES (since N is small, or use topological order)
    # To be robust against order, we can iterate until no changes or use topological sort.
    # With N=30, iterative is fast enough.
    # Initialize ES to 0.
    # EF = ES + Duration
    
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if es[i] < 0: # Not visited? No, we use 0 init.
                pass
            # Calculate new ES based on predecessors
            max_pred_ef = 0.0
            for p in preds[i]:
                if ef[p] > max_pred_ef:
                    max_pred_ef = ef[p]
            new_es = max_pred_ef
            if new_es > es[i]:
                es[i] = new_es
                ef[i] = es[i] + durations[i]
                changed = True
            else:
                # Ensure EF is consistent
                ef[i] = es[i] + durations[i]
                
    # Backward Pass: Calculate Latest Finish (LF) and Latest Start (LS)
    # Project Duration is max(EF)
    project_duration = np.max(ef)
    
    lf = np.full(n, project_duration, dtype=float)
    ls = np.zeros(n, dtype=float)
    
    changed = True
    while changed:
        changed = False
        for i in range(n):
            # LF is min(LS of successors)
            min_succ_ls = project_duration
            for s in succs[i]:
                if ls[s] < min_succ_ls:
                    min_succ_ls = ls[s]
            
            new_lf = min_succ_ls
            new_ls = new_lf - durations[i]
            
            if new_lf < lf[i]:
                lf[i] = new_lf
                ls[i] = new_ls
                changed = True
            else:
                # Keep consistent
                ls[i] = lf[i] - durations[i]

    # Calculate Total Float (Slack)
    # TF = LS - ES or LF - EF
    slack = ls - es
    
    # Ensure non-negative (numerical precision)
    slack = np.maximum(slack, 0.0)
    
    # Calculate Priority
    # Heuristic: Duration / (Slack + 1)
    # This gives high priority to long tasks with low slack.
    priority_scores = durations / (slack + 1.0)
    
    # Optional: Add a small random tie-breaker? No, deterministic is better for stability.
    # But if scores are identical, the order depends on the SGS implementation.
    # Usually index order is the tie-breaker.
    
    return priority_scores
# EVOLVE-BLOCK-END
