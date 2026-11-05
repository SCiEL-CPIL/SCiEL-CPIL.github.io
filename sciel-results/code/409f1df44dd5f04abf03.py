import numpy as np
from collections import defaultdict, deque

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build successor list
    successors = defaultdict(list)
    for i, preds in enumerate(predecessors):
        for p in preds:
            successors[p].append(i)
    
    # Compute out-degrees
    out_degree = [0] * n
    for i in range(n):
        out_degree[i] = len(successors[i])
    
    # Topological sort (Kahn's algorithm)
    in_degree = [len(preds) for preds in predecessors]
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    # If not all nodes are in topo_order, there's a cycle. But the problem assumes a DAG.
    # Compute RPW in reverse topological order
    rpw = np.zeros(n, dtype=float)
    for i in reversed(topo_order):
        rpw[i] = times[i] + sum(rpw[s] for s in successors[i])
    
    return rpw
