import time
import random

N = 9
NUM_VERTICES = 1 << N

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute neighbors
    neighbors = [[] for _ in range(NUM_VERTICES)]
    for v in range(NUM_VERTICES):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
            
    best_path = [0, 1]
    best_len = 1
    
    def run_search(heuristic_type, random_seed):
        nonlocal best_path, best_len
        random.seed(random_seed)
        
        # Symmetry breaking: 0 -> 1
        path = [0, 1]
        used = {0, 1}
        G = set(neighbors[0])
        
        def search(path, used, G):
            nonlocal best_path, best_len
            if time.time() > deadline:
                return
            
            cur = path[-1]
            candidates = []
            
            # Precompute G_next for scoring
            G_next = G | set(neighbors[cur])
            
            for v in neighbors[cur]:
                if v in used or v in G:
                    continue
                # Score
                if heuristic_type == 0: # Max options
                    score = 0
                    for w in neighbors[v]:
                        if w in used or w == v:
                            continue
                        if w in G_next:
                            continue
                        score += 1
                elif heuristic_type == 1: # Min options
                    score = 0
                    for w in neighbors[v]:
                        if w in used or w == v:
                            continue
                        if w in G_next:
                            continue
                        score += 1
                    score = -score
                else: # Random
                    score = random.random()
                    
                candidates.append((score, v))
                
            if not candidates:
                if len(path) - 1 > best_len:
                    best_len = len(path) - 1
                    best_path = path[:]
                return
            
            # Shuffle if random
            if heuristic_type == 2:
                random.shuffle(candidates)
            else:
                candidates.sort(key=lambda x: x[0], reverse=True)
                
            for score, v in candidates:
                if time.time() > deadline:
                    return
                new_path = path + [v]
                new_used = used | {v}
                new_G = G_next # G | N(cur)
                search(new_path, new_used, new_G)
                if time.time() > deadline:
                    return

        search(path, used, G)

    # Try different heuristics and seeds
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    heuristics = [0, 1, 2]
    
    idx = 0
    while time.time() < deadline:
        h = heuristics[idx % len(heuristics)]
        s = seeds[(idx // len(heuristics)) % len(seeds)]
        run_search(h, s)
        idx += 1
        
    return best_path
