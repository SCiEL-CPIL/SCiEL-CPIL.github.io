import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    N = 256
    
    # 1. Filter nodes: 2*v must not be a perfect square
    valid_nodes = []
    for v in range(1, N + 1):
        s = 2 * v
        r = int(math.isqrt(s))
        if r * r != s:
            valid_nodes.append(v)
    
    if not valid_nodes:
        return [1]
        
    # Map values to indices for faster lookup
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    M = len(valid_nodes)
    
    # 2. Build adjacency list
    # adj[i] is a set of indices j such that valid_nodes[i] + valid_nodes[j] is a square
    adj = [set() for _ in range(M)]
    
    for i in range(M):
        vi = valid_nodes[i]
        for j in range(i + 1, M):
            vj = valid_nodes[j]
            s = vi + vj
            r = int(math.isqrt(s))
            if r * r == s:
                adj[i].add(j)
                adj[j].add(i)
                
    # 3. Greedy MIS (Minimum Degree Heuristic)
    remaining = set(range(M))
    independent_set_indices = []
    
    while remaining:
        if time.time() > deadline:
            break
            
        # Find node with minimum degree in the remaining subgraph
        min_deg_node = None
        min_deg = float('inf')
        
        for u in remaining:
            deg = len(adj[u] & remaining)
            if deg < min_deg:
                min_deg = deg
                min_deg_node = u
                
        if min_deg_node is None:
            break
            
        # Add to independent set
        independent_set_indices.append(min_deg_node)
        
        # Remove node and its neighbors from remaining
        remaining.discard(min_deg_node)
        for v in list(adj[min_deg_node]):
            remaining.discard(v)
            
    # Convert back to values
    result = [valid_nodes[i] for i in independent_set_indices]
    
    if not result:
        return [valid_nodes[0]]
        
    return result
