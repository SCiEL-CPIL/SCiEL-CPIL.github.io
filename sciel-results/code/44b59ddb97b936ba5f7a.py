import numpy as np
from collections import defaultdict

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor graph
    # predecessors[i] is a list of indices of tasks that must precede i
    # We need successors: for each task u, which tasks v have u in predecessors[v]?
    successors = defaultdict(list)
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            
    # Compute transitive successor sum using DFS with memoization
    # score[i] = times[i] + sum(score[j] for j in immediate_successors(i)) 
    # Wait, if score[j] includes times[j] and its successors, then summing them 
    # correctly accumulates the total downstream load.
    
    memo = {}
    
    def calc_load(i):
        if i in memo:
            return memo[i]
        total = times[i]
        for j in successors[i]:
            total += calc_load(j)
        memo[i] = total
        return total
    
    scores = np.zeros(n)
    for i in range(n):
        scores[i] = calc_load(i)
        
    return scores
