import numpy as np
import time

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build successors list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            
    # Calculate out-degrees (number of successors)
    out_degrees = [len(succs) for succs in successors]
    
    # Initialize ranks
    rank = np.zeros(n, dtype=float)
    max_succ_rank = np.zeros(n, dtype=float)
    remaining_succs = out_degrees.copy()
    
    # Queue for tasks with no successors
    queue = [i for i in range(n) if out_degrees[i] == 0]
    
    # Process in reverse topological order
    while queue:
        u = queue.pop()
        # All successors of u have been processed, so rank[u] is final
        # Actually, rank[u] is set when it's pushed or processed?
        # Let's refine the logic.
        
        # When a node u is popped, it means all its successors have been processed.
        # So max_succ_rank[u] is the max rank of its successors.
        # rank[u] = times[u] + max_succ_rank[u]
        rank[u] = times[u] + max_succ_rank[u]
        
        # Now update predecessors of u
        for v in predecessors[u]:
            # u is a successor of v
            if rank[u] > max_succ_rank[v]:
                max_succ_rank[v] = rank[u]
            
            remaining_succs[v] -= 1
            if remaining_succs[v] == 0:
                queue.append(v)
                
    return rank
