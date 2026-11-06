import time
import random
import math

def is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def construct():
    start_time = time.time()
    # 55 seconds is a safe margin for 60s limit
    deadline = start_time + 55.0
    
    N = 256
    nodes = list(range(1, N + 1))
    
    # 1. Identify self-conflicts (2*x is a square)
    self_conflict = set()
    for x in nodes:
        if is_square(2 * x):
            self_conflict.add(x)
            
    valid_nodes = [x for x in nodes if x not in self_conflict]
    valid_set = set(valid_nodes)
    
    # 2. Build adjacency list for valid nodes
    # adj[u] contains v if u+v is a square
    adj = {x: set() for x in valid_nodes}
    for i in range(len(valid_nodes)):
        u = valid_nodes[i]
        for j in range(i + 1, len(valid_nodes)):
            v = valid_nodes[j]
            if is_square(u + v):
                adj[u].add(v)
                adj[v].add(u)
                
    # 3. Greedy Initialization (Min-Degree)
    candidates = set(valid_nodes)
    S = set()
    
    while candidates:
        # Find node with min degree in candidates
        min_deg = float('inf')
        best_node = None
        for u in candidates:
            # Degree in the induced subgraph of candidates
            deg = len(adj[u] & candidates)
            if deg < min_deg:
                min_deg = deg
                best_node = u
                
        S.add(best_node)
        # Remove best_node and its neighbors from candidates
        candidates.discard(best_node)
        for v in adj[best_node]:
            candidates.discard(v)
            
    best_S = set(S)
    best_size = len(S)
    
    # 4. Local Search (Kick and Repair)
    current_S = set(S)
    unselected = valid_set - current_S
    
    random.seed(42)
    
    while time.time() < deadline:
        if not current_S:
            break
            
        # Pick a random node to kick out
        kick_node = random.choice(list(current_S))
        
        # Create a trial set
        trial_S = set(current_S)
        trial_S.discard(kick_node)
        
        # Nodes available to add: unselected + kicked node
        available = set(unselected)
        available.add(kick_node)
        
        # Shuffle available to randomize the repair order
        avail_list = list(available)
        random.shuffle(avail_list)
        
        # Greedily add nodes from available to trial_S
        for u in avail_list:
            # Check if u can be added to trial_S
            # u conflicts with trial_S if any neighbor of u is in trial_S
            if not (adj[u] & trial_S):
                trial_S.add(u)
                
        # Evaluate the trial set
        if len(trial_S) > len(current_S):
            # Improvement found
            current_S = trial_S
            unselected = valid_set - current_S
            if len(current_S) > best_size:
                best_size = len(current_S)
                best_S = set(current_S)
        elif len(trial_S) == len(current_S):
            # Same size, but different configuration
            # Accept to allow exploration of the state space
            if trial_S != current_S:
                current_S = trial_S
                unselected = valid_set - current_S
        # If smaller, reject (implicit by not updating current_S)
        
    return list(best_S)
