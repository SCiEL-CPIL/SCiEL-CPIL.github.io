import time
import random
import math
import bisect

def _score(S_bits):
    """
    S_bits is a bitset integer where bit i is 1 if i is in S.
    Returns count of distinct d > 0 such that a, a+d, a+2d in S.
    """
    count = 0
    for d in range(1, 256):
        if (S_bits & (S_bits >> d) & (S_bits >> (2 * d))) != 0:
            count += 1
    return count

def _score_fast(S_bits, active_diffs):
    """
    Faster scoring using precomputed active differences.
    active_diffs is a dict mapping each element position to the set of d values
    for which it could be part of a progression.
    """
    count = 0
    for d in range(1, 256):
        if (S_bits & (S_bits >> d) & (S_bits >> (2 * d))) != 0:
            count += 1
    return count

def _compute_diff_interactions():
    """
    Precompute which pairs of differences (d1, d2) can be simultaneously
    realized by a set of 32 elements.
    """
    # For each d, find all valid (a, a+d, a+2d) triples within [1, 512]
    diff_triples = {}
    for d in range(1, 256):
        triples = []
        for a in range(1, 512 - 2 * d + 1):
            triples.append((a, a + d, a + 2 * d))
        diff_triples[d] = triples
    return diff_triples

def _generate_initial_set(N, MAXV):
    """
    Generate an initial set biased toward creating many 3-term progressions.
    Uses a structured approach: pick a base, then add elements in patterns
    that realize small differences.
    """
    S = set()
    
    # Strategy: Use arithmetic progressions with small common differences
    # to realize many small d values simultaneously
    
    # Pick 2-3 base arithmetic progressions
    num_progressions = random.randint(2, 4)
    for p in range(num_progressions):
        base = random.randint(1, 50)
        d = random.randint(1, 8)  # Small differences
        length = random.randint(5, 12)
        for i in range(length):
            val = base + i * d
            if 1 <= val <= MAXV:
                S.add(val)
    
    # Fill remaining with random elements near existing ones
    while len(S) < N:
        if S and random.random() < 0.7:
            ref = random.choice(list(S))
            offset = random.randint(-10, 10)
            val = ref + offset
        else:
            val = random.randint(1, MAXV)
        if 1 <= val <= MAXV and val not in S:
            S.add(val)
    
    # If still not enough, fill with random
    while len(S) < N:
        val = random.randint(1, MAXV)
        if val not in S:
            S.add(val)
    
    return sorted(list(S))

def _targeted_move(current_list, current_bits, N, MAXV):
    """
    Generate a targeted move that tries to add a missing small difference.
    Returns (new_list, new_bits) or None if no good move found.
    """
    current_set = set(current_list)
    
    # Find which small differences are missing
    missing_diffs = []
    for d in range(1, 30):
        if (current_bits & (current_bits >> d) & (current_bits >> (2 * d))) == 0:
            missing_diffs.append(d)
    
    if not missing_diffs:
        return None
    
    # Try to realize one of the missing differences
    target_d = random.choice(missing_diffs[:10])  # Prefer small ones
    
    # Find positions where we could add elements to realize this d
    # We need a, a+d, a+2d all in set
    # Currently, for this d, no such triple exists
    
    # Try to find a pair (a, a+2d) or (a, a+d) that we can complete
    candidates = []
    for a in range(1, 512 - 2 * target_d + 1):
        mid = a + target_d
        end = a + 2 * target_d
        
        # Count how many of the three are already in the set
        in_set = sum(1 for x in [a, mid, end] if x in current_set)
        
        if in_set == 2:
            # We can complete this progression by adding one element
            missing = [x for x in [a, mid, end] if x not in current_set][0]
            if 1 <= missing <= MAXV:
                candidates.append((missing, a))
    
    if not candidates:
        return None
    
    missing_val, _ = random.choice(candidates)
    
    # Find an element to remove (prefer one that's not critical for other progressions)
    # Simple heuristic: remove a random element
    x_idx = random.randint(0, N - 1)
    x = current_list[x_idx]
    
    if x == missing_val:
        return None
    
    # Create new set
    new_list = current_list[:x_idx] + current_list[x_idx+1:]
    ins_idx = bisect.bisect_left(new_list, missing_val)
    new_list.insert(ins_idx, missing_val)
    new_list.sort()
    
    new_bits = 0
    for x in new_list:
        new_bits |= (1 << x)
    
    return new_list, new_bits

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    N = 32
    MAXV = 512
    
    best_score = -1
    best_list = []
    
    # Precompute difference triple information for guided search
    diff_triples = _compute_diff_interactions()
    
    # Run multiple restarts with different strategies
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        
        # Generate initial set using structured approach
        current_list = _generate_initial_set(N, MAXV)
        current_bits = 0
        for x in current_list:
            current_bits |= (1 << x)
        
        current_score = _score(current_bits)
        
        if current_score > best_score:
            best_score = current_score
            best_list = current_list.copy()
        
        # Simulated annealing with targeted moves
        T = 8.0
        T_min = 0.01
        cooling = 0.9995
        
        # Adaptive local deadline: use more time for later restarts
        time_remaining = deadline - start_time
        local_time = min(6.0, time_remaining * 0.5) if num_restarts > 1 else min(8.0, time_remaining)
        local_deadline = start_time + local_time
        
        targeted_count = 0
        max_targeted = 50  # Limit targeted moves per restart
        
        while time.time() < local_deadline:
            # Choose move type
            move_type = random.random()
            
            if move_type < 0.5:
                # 1-swap
                x_idx = random.randint(0, N - 1)
                x = current_list[x_idx]
                
                # Bias new value toward creating small-d progressions
                if random.random() < 0.6:
                    # Near an existing element with small offset
                    ref = random.choice(current_list)
                    offset = random.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                    y = ref + offset
                    if y < 1 or y > MAXV or ((current_bits >> y) & 1):
                        y = random.randint(1, MAXV)
                        while ((current_bits >> y) & 1):
                            y = random.randint(1, MAXV)
                else:
                    while True:
                        y = random.randint(1, MAXV)
                        if not ((current_bits >> y) & 1):
                            break
                
                new_bits = current_bits ^ (1 << x) ^ (1 << y)
                
                new_list = current_list[:x_idx] + current_list[x_idx+1:]
                ins_idx = bisect.bisect_left(new_list, y)
                new_list.insert(ins_idx, y)
                
            elif move_type < 0.8:
                # 2-swap
                idxs = random.sample(range(N), 2)
                x1 = current_list[idxs[0]]
                x2 = current_list[idxs[1]]
                
                y1 = y2 = 0
                while y1 == 0 or y1 == y2 or ((current_bits >> y1) & 1):
                    y1 = random.randint(1, MAXV)
                while y2 == 0 or y1 == y2 or ((current_bits >> y2) & 1):
                    y2 = random.randint(1, MAXV)
                
                new_bits = current_bits ^ (1 << x1) ^ (1 << x2) ^ (1 << y1) ^ (1 << y2)
                
                if idxs[0] > idxs[1]:
                    idxs[0], idxs[1] = idxs[1], idxs[0]
                
                new_list = current_list[:idxs[0]] + current_list[idxs[0]+1:idxs[1]] + current_list[idxs[1]+1:]
                
                bisect.insort(new_list, y1)
                bisect.insort(new_list, y2)
            else:
                # Targeted move to add a missing small difference
                if targeted_count < max_targeted:
                    result = _targeted_move(current_list, current_bits, N, MAXV)
                    if result is None:
                        # Fall back to random 1-swap
                        x_idx = random.randint(0, N - 1)
                        x = current_list[x_idx]
                        while True:
                            y = random.randint(1, MAXV)
                            if not ((current_bits >> y) & 1):
                                break
                        new_bits = current_bits ^ (1 << x) ^ (1 << y)
                        new_list = current_list[:x_idx] + current_list[x_idx+1:]
                        ins_idx = bisect.bisect_left(new_list, y)
                        new_list.insert(ins_idx, y)
                    else:
                        new_list, new_bits = result
                        targeted_count += 1
                else:
                    # Fall back to random 1-swap
                    x_idx = random.randint(0, N - 1)
                    x = current_list[x_idx]
                    while True:
                        y = random.randint(1, MAXV)
                        if not ((current_bits >> y) & 1):
                            break
                    new_bits = current_bits ^ (1 << x) ^ (1 << y)
                    new_list = current_list[:x_idx] + current_list[x_idx+1:]
                    ins_idx = bisect.bisect_left(new_list, y)
                    new_list.insert(ins_idx, y)
            
            new_score = _score(new_bits)
            
            delta = new_score - current_score
            
            if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                current_list = new_list
                current_bits = new_bits
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_list = current_list.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        # Final polish: try targeted improvements on the best solution
        if best_score > 0 and time.time() < deadline - 2.0:
            polish_list = best_list.copy()
            polish_bits = 0
            for x in polish_list:
                polish_bits |= (1 << x)
            polish_score = _score(polish_bits)
            
            polish_deadline = min(deadline - 0.5, time.time() + 3.0)
            
            for _ in range(100):
                if time.time() > polish_deadline:
                    break
                
                result = _targeted_move(polish_list, polish_bits, N, MAXV)
                if result is None:
                    break
                
                new_list, new_bits = result
                new_score = _score(new_bits)
                
                if new_score > polish_score:
                    polish_list = new_list
                    polish_bits = new_bits
                    polish_score = new_score
                    
                    if polish_score > best_score:
                        best_score = polish_score
                        best_list = polish_list.copy()
                else:
                    # Try a few random moves as well
                    x_idx = random.randint(0, N - 1)
                    x = polish_list[x_idx]
                    while True:
                        y = random.randint(1, MAXV)
                        if not ((polish_bits >> y) & 1):
                            break
                    new_bits = polish_bits ^ (1 << x) ^ (1 << y)
                    new_list = polish_list[:x_idx] + polish_list[x_idx+1:]
                    ins_idx = bisect.bisect_left(new_list, y)
                    new_list.insert(ins_idx, y)
                    new_score = _score(new_bits)
                    
                    if new_score > polish_score:
                        polish_list = new_list
                        polish_bits = new_bits
                        polish_score = new_score
                        
                        if polish_score > best_score:
                            best_score = polish_score
                            best_list = polish_list.copy()
    
    return best_list
