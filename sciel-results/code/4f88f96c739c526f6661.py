"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(instance):
    """
    Computes priority scores based on Critical Path Method (CPM) slack.
    Activities with lower slack (more critical) receive higher priority scores.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    preds_raw = instance["predecessors"]
    n = len(durations)
    
    # Normalize predecessors to a list of lists of integers
    predecessors = []
    for i in range(n):
        p = preds_raw[i]
        # Convert to list of ints to ensure compatibility
        if isinstance(p, np.ndarray):
            predecessors.append([int(x) for x in p])
        else:
            predecessors.append([int(x) for x in p])
            
    # Build successor list
    successors = [[] for _ in range(n)]
    for j in range(n):
        for i in predecessors[j]:
            successors[i].append(j)
            
    # Forward Pass: Calculate Earliest Start (ES) and Earliest Finish (EF)
    ES = np.zeros(n)
    EF = np.zeros(n)
    
    # Iterate until no changes (convergence)
    # Max N iterations needed for a DAG of size N
    for _ in range(n):
        changed = False
        for j in range(n):
            if predecessors[j]:
                max_pred_ef = max(EF[i] for i in predecessors[j])
                if max_pred_ef > ES[j]:
                    ES[j] = max_pred_ef
                    EF[j] = ES[j] + durations[j]
                    changed = True
            else:
                # Root nodes: ES is 0, EF is duration
                if EF[j] < durations[j]:
                    EF[j] = durations[j]
                    changed = True
        if not changed:
            break
            
    # Project duration based on CPM
    T = max(EF)
    
    # Backward Pass: Calculate Latest Start (LS) and Latest Finish (LF)
    LF = np.zeros(n)
    LS = np.zeros(n)
    
    # Initialize leaf nodes
    for j in range(n):
        if not successors[j]:
            LF[j] = T
            LS[j] = T - durations[j]
            
    # Iterate until no changes
    for _ in range(n):
        changed = False
        for j in range(n):
            if successors[j]:
                min_succ_ls = min(LS[k] for k in successors[j])
                if min_succ_ls < LF[j]:
                    LF[j] = min_succ_ls
                    LS[j] = LF[j] - durations[j]
                    changed = True
        if not changed:
            break
            
    # Calculate Slack
    slack = LS - ES
    
    # Priority Score:
    # Higher score means higher priority.
    # Lower slack is more critical -> Higher priority.
    # So score = -slack.
    # We add a tiny fraction of duration to break ties, favoring longer tasks
    # which aligns with the baseline's intuition that long tasks matter,
    # though slack is the dominant factor.
    scores = -slack + 0.001 * durations
    
    return np.asarray(scores, dtype=float)
# EVOLVE-BLOCK-END
