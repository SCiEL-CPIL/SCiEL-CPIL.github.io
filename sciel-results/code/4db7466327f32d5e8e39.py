import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s budget
    
    N = 512
    
    # Precompute all 4-APs in [1, N]
    aps = []
    element_aps = [[] for _ in range(N + 1)]
    
    idx = 0
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap = (a, b, c, e)
            aps.append(ap)
            element_aps[a].append(idx)
            element_aps[b].append(idx)
            element_aps[c].append(idx)
            element_aps[e].append(idx)
            idx += 1
            
    num_aps = len(aps)
    
    def run_greedy_remove(random_seed):
        random.seed(random_seed)
        in_set = [True] * (N + 1)
        deg = [0] * (N + 1)
        ap_full_count = [4] * num_aps
        full_aps_total = num_aps
        
        # Initialize deg
        for x in range(1, N + 1):
            for ap_idx in element_aps[x]:
                if ap_full_count[ap_idx] == 4:
                    deg[x] += 1
        
        while full_aps_total > 0:
            if time.time() > deadline:
                break
            
            # Find candidates with max deg
            max_deg = 0
            candidates = []
            for x in range(1, N + 1):
                if in_set[x]:
                    if deg[x] > max_deg:
                        max_deg = deg[x]
                        candidates = [x]
                    elif deg[x] == max_deg:
                        candidates.append(x)
            
            if not candidates:
                break
            
            # Pick a candidate randomly from the top ones to allow exploration
            x = random.choice(candidates)
            
            # Remove x
            in_set[x] = False
            
            # Update APs containing x
            for ap_idx in element_aps[x]:
                if ap_full_count[ap_idx] == 4:
                    full_aps_total -= 1
                    a, b, c, e = aps[ap_idx]
                    for y in (a, b, c, e):
                        if y != x:
                            deg[y] -= 1
                ap_full_count[ap_idx] -= 1
            
        return [x for x in range(1, N + 1) if in_set[x]]

    best_set = []
    best_size = 0
    
    # Run multiple iterations
    # First, a deterministic run (seed 0) to get a baseline
    s1 = run_greedy_remove(0)
    best_set = s1
    best_size = len(s1)
    
    # Then randomized runs
    seed = 1
    while time.time() < deadline:
        s = run_greedy_remove(seed)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
        seed += 1
        
    return sorted(best_set)
