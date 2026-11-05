import time
import random

N = 9

def construct():
    start_time = time.time()
    time_limit = 120.0  # Safety margin before 150s kill
    
    best_path = []
    best_len = 0
    
    # Precompute neighbors for all vertices? 2^9 = 512 vertices.
    # Neighbors of v: [v ^ (1<<b) for b in range(N)]
    
    def get_neighbors(v):
        return [v ^ (1 << b) for b in range(N)]
        
    def greedy_search(start, random_tie_breaker=False):
        path = [start]
        used = {start}
        blocked = set()
        
        while True:
            cur = path[-1]
            candidates = []
            
            # Generate neighbors of cur
            for b in range(N):
                nxt = cur ^ (1 << b)
                if nxt in used:
                    continue
                if nxt in blocked:
                    continue
                candidates.append(nxt)
                
            if not candidates:
                break
            
            # Score candidates
            best_score = -1
            best_nxt = None
            best_tie_breaker_val = None
            
            # If random_tie_breaker, we might want to shuffle candidates or use random tie-break
            # But for speed, we just pick max score, and if tie, use fixed order or random.
            
            # To allow different paths, we can randomize the order of candidates if scores are equal
            # Or just use a global random seed per trial
            
            for nxt in candidates:
                # Score: number of free neighbors of nxt
                # Free neighbor: not in used, not in blocked
                # Note: blocked currently contains neighbors of path[:-1]
                # When we move to nxt, the new blocked will include neighbors of cur.
                # So for scoring, we should consider the state AFTER moving to nxt?
                # If we move to nxt, `cur` becomes `path[-2]`.
                # So `neighbors(cur)` will be added to `blocked`.
                # So a neighbor `m` of `nxt` is free if:
                # 1. m not in used
                # 2. m not in blocked (current blocked)
                # 3. m not in neighbors(cur) (because they will be blocked)
                
                # Let's check these conditions.
                score = 0
                for b2 in range(N):
                    m = nxt ^ (1 << b2)
                    if m in used:
                        continue
                    if m in blocked:
                        continue
                    # Check if m is neighbor of cur
                    # m is neighbor of cur if (m ^ cur).bit_count() == 1
                    if (m ^ cur).bit_count() == 1:
                        continue
                    score += 1
                
                if score > best_score:
                    best_score = score
                    best_nxt = nxt
                    best_tie_breaker_val = random.random() if random_tie_breaker else 0
                elif score == best_score:
                    if random_tie_breaker:
                        if random.random() > best_tie_breaker_val:
                            best_nxt = nxt
                            best_tie_breaker_val = random.random()
                    else:
                        # Fixed tie-break: lowest vertex or something
                        pass
                        
            if best_nxt is None:
                break
                
            path.append(best_nxt)
            used.add(best_nxt)
            
            # Update blocked: add neighbors of path[-2]
            if len(path) >= 2:
                prev_cur = path[-2]
                for b in range(N):
                    nb = prev_cur ^ (1 << b)
                    blocked.add(nb)
            
            # Time check inside loop?
            # It's fast, so maybe not needed per step, but per trial is good.
            
        return path

    # Run multiple trials
    # 512 possible starts
    # We can iterate through all starts, or random sample.
    # Given 150s, we can do many.
    
    # Let's try all 512 starts with fixed tie-breaking first, then random if time permits.
    
    for start in range(512):
        if time.time() - start_time > time_limit - 5:
            break
            
        p = greedy_search(start, random_tie_breaker=False)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
            
    # If time remains, do random restarts with random tie-breaking
    # This explores different branches in the greedy tree.
    
    while time.time() - start_time < time_limit - 5:
        start = random.randint(0, 511)
        p = greedy_search(start, random_tie_breaker=True)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
            
    return best_path
