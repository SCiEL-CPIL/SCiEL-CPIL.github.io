import time
import random

N = 9
DEADLINE = 150.0

def construct():
    start_time = time.time()
    best_path = [0]
    
    neighbors = [[] for _ in range(1 << N)]
    for v in range(1 << N):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    for attempt in range(1000000):
        if time.time() - start_time > DEADLINE - 5.0:
            break
        
        random.seed(attempt)
        
        # Stack of (path, used, blocked, candidates, index)
        # This is getting complex. Let's stick to greedy with restarts first.
        # If that doesn't work, we can add backtracking.
        
        path = [0]
        used = {0}
        blocked = set()
        
        while True:
            cur = path[-1]
            candidates = []
            
            for v in neighbors[cur]:
                if v in used or v in blocked:
                    continue
                candidates.append(v)
            
            if not candidates:
                break
            
            new_blocked = blocked.union(neighbors[cur])
            
            scored_candidates = []
            for v in candidates:
                score = 0
                for u in neighbors[v]:
                    if u not in used and u not in new_blocked:
                        score += 1
                scored_candidates.append((score, v))
            
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            
            if len(scored_candidates) > 1 and random.random() < 0.2:
                top_k = scored_candidates[:min(3, len(scored_candidates))]
                _, v = random.choice(top_k)
            else:
                _, v = scored_candidates[0]
            
            path.append(v)
            used.add(v)
            blocked = new_blocked
            
            if len(path) > len(best_path):
                best_path = path.copy()
        
    return best_path
