import random

def construct():
    n = 256
    # 1. Build conflict graph
    adj = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj[a].add(b)
                adj[b].add(a)
                
    best_set = set()
    best_size = 0
    
    # 2. Randomized Greedy with Restarts
    for _ in range(500): # 500 restarts
        current_set = set()
        remaining = set(range(1, n + 1))
        
        # Greedy step: repeatedly pick a random available node
        while remaining:
            # Find all nodes in 'remaining' that have no neighbors in 'current_set'
            # Actually, 'remaining' is the set of nodes not yet decided.
            # A node v can be added if it has no neighbors in current_set.
            # We maintain 'available' nodes.
            
            # Optimization: Instead of checking all remaining, we can just pick one.
            # But to be efficient, let's find candidates.
            candidates = []
            for v in remaining:
                if not (adj[v] & current_set):
                    candidates.append(v)
            
            if not candidates:
                break
            
            # Pick randomly from candidates
            v = random.choice(candidates)
            current_set.add(v)
            
            # Remove v and its neighbors from remaining
            to_remove = adj[v] | {v}
            remaining -= to_remove
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return list(best_set)
