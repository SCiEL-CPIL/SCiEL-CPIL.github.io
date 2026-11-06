import time
import random

N = 9

# Precompute neighbors and neighbor sets
NEIGHBORS = [[] for _ in range(1 << N)]
NEIGHBOR_SETS = [set() for _ in range(1 << N)]
for v in range(1 << N):
    for b in range(N):
        u = v ^ (1 << b)
        NEIGHBORS[v].append(u)
    NEIGHBOR_SETS[v] = set(NEIGHBORS[v])

def generate_seed():
    path = [0]
    used = {0}
    while True:
        cur = path[-1]
        nxt = None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            valid = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    valid = False
                    break
            if valid:
                nxt = v
                break
        if nxt is None:
            return path
        path.append(nxt)
        used.add(nxt)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Get baseline
    best_path = generate_seed()
    best_len = len(best_path)
    
    # Randomized Warnsdorff Search
    while time.time() < deadline:
        # Random start
        start_v = random.randint(0, (1 << N) - 1)
        path = [start_v]
        used = {start_v}
        forbidden = set()
        
        while True:
            if time.time() > deadline:
                break
                
            cur = path[-1]
            
            # Find candidates
            candidates = []
            for v in NEIGHBORS[cur]:
                if v not in used and v not in forbidden:
                    candidates.append(v)
            
            if not candidates:
                break
                
            # Determine next forbidden set (for scoring)
            # next_forbidden = forbidden | NEIGHBOR_SETS[cur]
            
            # Score candidates
            min_score = float('inf')
            best_cands = []
            for v in candidates:
                # Count extensions for v
                # Extensions are neighbors of v not in used and not in (forbidden | N(cur))
                # Note: v is in N(cur), so v is in next_forbidden.
                # But we are checking neighbors of v.
                count = 0
                for u in NEIGHBORS[v]:
                    if u not in used:
                        # Check if u is in forbidden OR u is in N(cur)
                        if u not in forbidden and u not in NEIGHBOR_SETS[cur]:
                            count += 1
                            
                if count < min_score:
                    min_score = count
                    best_cands = [v]
                elif count == min_score:
                    best_cands.append(v)
            
            # Random choice
            nxt = random.choice(best_cands)
            
            path.append(nxt)
            used.add(nxt)
            
            # Update forbidden
            # cur becomes "internal", so its neighbors are forbidden for future steps
            forbidden.update(NEIGHBOR_SETS[cur])
            
        if len(path) > best_len:
            best_len = len(path)
            best_path = path.copy()
            
    return best_path
