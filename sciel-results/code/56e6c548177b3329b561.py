<<<<<<< SEARCH
def _construct_direct(dist, points, time_budget=280.0):
    """
    Direct construction: 
    1. Use a greedy approach with strong heuristic to minimize set size.
    2. Then local search to reduce further.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    best_size = V  # worst case
    best_set = None
    
    # Strategy: Try multiple random starts and greedy builds
    rng = random.Random(42)
    
    for trial in range(500):
        if time.time() > deadline:
            break
        
        seed = trial * 7919 + 12345
        rng = random.Random(seed)
        
        # Greedy build: always pick the point that blocks the most new points
        S = []
        U = set(range(V))
        
        while U:
            # Find all candidates that can be added
            candidates = []
            for p in U:
                can_add = True
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        a, b = S[i], S[j]
                        dab = dist[a, b]
                        dap = dist[a, p]
                        dbp = dist[b, p]
                        if dab == dap or dab == dbp or dap == dbp:
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    candidates.append(p)
            
            if not candidates:
                break
            
            # Score each candidate by how many points it would block
            best_score = -1
            best_p = None
            best_blocked = set()
            
            # Shuffle for randomization
            cand_list = candidates[:]
            rng.shuffle(cand_list)
            
            for p in cand_list:
                # Compute how many points in U would be blocked
                new_blocked = set()
                for a in S:
                    dab = dist[a, p]
                    for u in U:
                        if u == p or u in new_blocked:
                            continue
                        da = dist[a, u]
                        dp = dist[p, u]
                        if dab == da or dab == dp or da == dp:
                            new_blocked.add(u)
                
                score = len(new_blocked)
                
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_blocked = new_blocked
                elif score == best_score and rng.random() < 0.3:
                    best_p = p
                    best_blocked = new_blocked
            
            if best_p is None:
                break
            
            S.append(best_p)
            U.remove(best_p)
            U -= best_blocked
        
        # Local search: try to remove points
        S = list(S)
        for it in range(50):
            if time.time() > deadline:
                break
            improved = False
            for idx in range(len(S)):
                trial_S = S[:idx] + S[idx+1:]
                if not trial_S:
                    continue
                # Check validity
                valid = True
                for i in range(len(trial_S)):
                    for j in range(i + 1, len(trial_S)):
                        for k in trial_S:
                            if k != trial_S[i] and k != trial_S[j]:
                                a, b = trial_S[i], trial_S[j]
                                c = k
                                dab = dist[a, b]
                                dac = dist[a, c]
                                dbc = dist[b, c]
                                if dab == dac or dab == dbc or dac == dbc:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                # Check maximality
                if _is_maximal(trial_S, dist, V):
                    S = trial_S
                    improved = True
                    break
            if not improved:
                break
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)
    
    if best_set is None:
        # Fallback: just return a valid set (even if not minimal)
        return []
    
    return [points[i] for i in best_set]

def construct():
    points, dist, _ = _precompute()
    return _construct_direct(dist, points)
=======
def _construct_direct(dist, points, pair_blocked, time_budget=280.0):
    """
    Direct construction: 
    1. Use a greedy approach with strong heuristic to minimize set size.
    2. Then local search to reduce further.
    Uses precomputed pair_blocked bitmasks for O(1) conflict checks.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    best_size = V  # worst case
    best_set = None
    
    # Strategy: Try multiple random starts and greedy builds
    for trial in range(2000):
        if time.time() > deadline:
            break
        
        seed = trial * 7919 + 12345
        rng = random.Random(seed)
        
        # Greedy build: always pick the point that blocks the most new points
        S = []
        # Track blocked points as a bitmask
        blocked_mask = 0
        # Available points bitmask
        avail_mask = (1 << V) - 1
        
        while avail_mask:
            # Find all candidates that can be added (not in S, not blocked)
            # Candidates are points not in S and not in blocked_mask
            # But we also need to ensure adding them doesn't create isosceles with existing pairs
            # Actually, if a point is not blocked by any pair in S, it can be added
            
            # Get candidate bitmask
            S_mask = 0
            for s in S:
                S_mask |= (1 << s)
            cand_mask = avail_mask & ~S_mask & ~blocked_mask
            
            if not cand_mask:
                break
            
            # Get list of candidates
            candidates = []
            temp = cand_mask
            while temp:
                lsb = temp & (-temp)
                p = lsb.bit_length() - 1
                candidates.append(p)
                temp ^= lsb
            
            # Score each candidate by how many additional points it would block
            best_score = -1
            best_p = None
            best_new_blocked = 0
            
            # Shuffle for randomization
            rng.shuffle(candidates)
            
            for p in candidates:
                # Compute new blocked points: for each a in S, pair_blocked[(a,p)]
                new_blocked = 0
                for a in S:
                    new_blocked |= pair_blocked.get((a, p), 0)
                    new_blocked |= pair_blocked.get((p, a), 0)
                
                # Count how many currently available points would be newly blocked
                newly_blocked = new_blocked & avail_mask & ~blocked_mask
                score = bin(newly_blocked).count('1')
                
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_new_blocked = new_blocked
                elif score == best_score and rng.random() < 0.3:
                    best_p = p
                    best_new_blocked = new_blocked
            
            if best_p is None:
                break
            
            S.append(best_p)
            blocked_mask |= best_new_blocked
            avail_mask &= ~(1 << best_p)
        
        # Local search: try to remove points
        S = list(S)
        for it in range(100):
            if time.time() > deadline:
                break
            improved = False
            # Try removing each point
            for idx in range(len(S)):
                trial_S = S[:idx] + S[idx+1:]
                if not trial_S:
                    continue
                # Check validity using precomputed pair_blocked
                valid = _is_valid_set(trial_S, pair_blocked)
                if not valid:
                    continue
                # Check maximality
                if _is_maximal(trial_S, dist, pair_blocked, V):
                    S = trial_S
                    improved = True
                    break
            if not improved:
                break
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)
    
    if best_set is None:
        # Fallback: just return a valid set (even if not minimal)
        return []
    
    return [points[i] for i in best_set]

def construct():
    points, dist, pair_blocked = _precompute()
    return _construct_direct(dist, points, pair_blocked)
>>>>>>> REPLACE

