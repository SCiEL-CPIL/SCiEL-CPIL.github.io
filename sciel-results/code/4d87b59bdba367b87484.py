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
                
    def run_greedy(seed):
        random.seed(seed)
        remaining = set(range(1, n + 1))
        result = []
        
        # Pre-compute initial degrees? No, they change.
        # To speed up tie-breaking, we can assign random priorities.
        # But simple random choice among min-degree is enough.
        
        while remaining:
            min_deg = float('inf')
            candidates = []
            
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [v]
                elif deg == min_deg:
                    candidates.append(v)
            
            # Pick one candidate randomly
            chosen = random.choice(candidates)
            result.append(chosen)
            
            # Remove chosen and its neighbors
            to_remove = adj[chosen] | {chosen}
            remaining -= to_remove
            
        return result

    best_set = []
    # Run multiple times
    # 256 nodes, greedy is fast. 1000 runs is safe.
    for i in range(1000):
        current_set = run_greedy(i)
        if len(current_set) > len(best_set):
            best_set = current_set
            
    return best_set
