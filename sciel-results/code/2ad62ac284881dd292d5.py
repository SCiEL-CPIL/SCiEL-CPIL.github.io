import random
import time
import sys

N = 16
NUM_POINTS = N * N
POINTS = [(x, y) for y in range(N) for x in range(N)]

def precompute():
    dist2 = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
    for i in range(NUM_POINTS):
        xi, yi = POINTS[i]
        for j in range(NUM_POINTS):
            xj, yj = POINTS[j]
            dist2[i][j] = (xi - xj) ** 2 + (yi - yj) ** 2

    block_mask = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
    for i in range(NUM_POINTS):
        for j in range(i + 1, NUM_POINTS):
            mask = 0
            d_ij = dist2[i][j]
            for x in range(NUM_POINTS):
                if x == i or x == j:
                    continue
                d_ix = dist2[i][x]
                d_jx = dist2[j][x]
                if d_ix == d_jx or d_ix == d_ij or d_jx == d_ij:
                    mask |= (1 << x)
            block_mask[i][j] = mask
            block_mask[j][i] = mask
    return dist2, block_mask

DIST2, BLOCK_MASK = precompute()

def construct():
    start_time = time.time()
    deadline = start_time + 280
    FULL_MASK = (1 << NUM_POINTS) - 1
    
    best_set = None
    best_size = float('inf')
    
    # Greedy initial construction
    def greedy_initial(seed_val):
        random.seed(seed_val)
        S = set()
        counts = [0] * NUM_POINTS
        indices = list(range(NUM_POINTS))
        random.shuffle(indices)
        
        for p_idx in indices:
            if counts[p_idx] == 0:
                S.add(p_idx)
                for q_idx in S:
                    if q_idx == p_idx:
                        continue
                    mask = BLOCK_MASK[p_idx][q_idx]
                    m = mask
                    while m:
                        lsb = m & -m
                        x_idx = lsb.bit_length() - 1
                        counts[x_idx] += 1
                        m ^= lsb
        return S
    
    # State: S (set), blocked_mask (int), pair_counts (list)
    # blocked_mask: OR of all BLOCK_MASK[a][b] for pairs in S
    # pair_counts[x]: number of pairs (a,b) in S such that x in BLOCK_MASK[a][b]
    
    def build_state(S):
        blocked_mask = 0
        pair_counts = [0] * NUM_POINTS
        s_list = list(S)
        for i in range(len(s_list)):
            a = s_list[i]
            for j in range(i + 1, len(s_list)):
                b = s_list[j]
                mask = BLOCK_MASK[a][b]
                blocked_mask |= mask
                m = mask
                while m:
                    lsb = m & -m
                    x = lsb.bit_length() - 1
                    pair_counts[x] += 1
                    m ^= lsb
        return blocked_mask, pair_counts
    
    def check_maximal(S, blocked_mask):
        s_mask = 0
        for i in S:
            s_mask |= (1 << i)
        not_in_S = FULL_MASK ^ s_mask
        return (not_in_S & ~blocked_mask) == 0
    
    # Removal with incremental update
    def try_remove(S, p_idx, blocked_mask, pair_counts):
        """Try to remove p_idx from S. Returns (new_S, new_blocked_mask, new_pair_counts, success)"""
        # Compute new state without p_idx
        new_blocked_mask = 0
        new_pair_counts = [0] * NUM_POINTS
        s_list = list(S)
        for i in range(len(s_list)):
            a = s_list[i]
            if a == p_idx:
                continue
            for j in range(i + 1, len(s_list)):
                b = s_list[j]
                if b == p_idx:
                    continue
                mask = BLOCK_MASK[a][b]
                new_blocked_mask |= mask
                m = mask
                while m:
                    lsb = m & -m
                    x = lsb.bit_length() - 1
                    new_pair_counts[x] += 1
                    m ^= lsb
        
        # Check p_idx is blocked
        if (new_blocked_mask >> p_idx) & 1 == 0:
            return None, None, None, False
        
        # Check maximality
        new_s_mask = 0
        for i in S:
            if i != p_idx:
                new_s_mask |= (1 << i)
        not_in_new = FULL_MASK ^ new_s_mask
        if (not_in_new & ~new_blocked_mask) != 0:
            return None, None, None, False
        
        new_S = set(S)
        new_S.discard(p_idx)
        return new_S, new_blocked_mask, new_pair_counts, True
    
    # Local search with removals and swaps
    def local_search(S_init, seed_val, time_limit):
        random.seed(seed_val)
        S = set(S_init)
        blocked_mask, pair_counts = build_state(S)
        
        if not check_maximal(S, blocked_mask):
            return len(S), S
        
        current_size = len(S)
        best_size_local = current_size
        best_set_local = set(S)
        
        # Phase 1: Greedy removal
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            s_list = list(S)
            random.shuffle(s_list)
            
            for p_idx in s_list:
                if p_idx not in S:
                    continue
                
                new_S, new_blocked_mask, new_pair_counts, success = try_remove(S, p_idx, blocked_mask, pair_counts)
                if success:
                    S = new_S
                    blocked_mask = new_blocked_mask
                    pair_counts = new_pair_counts
                    current_size = len(S)
                    improved = True
                    
                    if current_size < best_size_local:
                        best_size_local = current_size
                        best_set_local = set(S)
                    
                    break
        
        # Phase 2: Swap-based local search
        # Try to find a point q not in S that can replace some p in S
        # with the result being still maximal and no-isosceles
        for outer_iter in range(100):
            if time.time() > time_limit:
                break
            
            found_improvement = False
            s_list = list(S)
            random.shuffle(s_list)
            
            for p_idx in s_list:
                if time.time() > time_limit:
                    break
                
                # Compute state without p_idx
                temp_blocked_mask = 0
                temp_pair_counts = [0] * NUM_POINTS
                s_list2 = list(S)
                for i in range(len(s_list2)):
                    a = s_list2[i]
                    if a == p_idx:
                        continue
                    for j in range(i + 1, len(s_list2)):
                        b = s_list2[j]
                        if b == p_idx:
                            continue
                        mask = BLOCK_MASK[a][b]
                        temp_blocked_mask |= mask
                        m = mask
                        while m:
                            lsb = m & -m
                            x = lsb.bit_length() - 1
                            temp_pair_counts[x] += 1
                            m ^= lsb
                
                # Points that become unblocked when p_idx is removed
                # These are candidates to add
                s_mask_without_p = 0
                for i in S:
                    if i != p_idx:
                        s_mask_without_p |= (1 << i)
                
                # Unblocked points not in S\{p_idx}
                unblocked = (~temp_blocked_mask) & FULL_MASK & (~s_mask_without_p)
                
                if unblocked == 0:
                    continue
                
                # Try each unblocked point as a candidate
                m = unblocked
                while m:
                    lsb = m & -m
                    q_idx = lsb.bit_length() - 1
                    m ^= lsb
                    
                    # Check if S\{p_idx} ∪ {q_idx} is no-isosceles
                    # For all a in S\{p_idx}, q_idx must not be in BLOCK_MASK[a][b] for any b in S\{p_idx}\{a}
                    # And for all b in S\{p_idx}, a must not be in BLOCK_MASK[q_idx][b]
                    # Simplified: for all pairs (a,b) in S\{p_idx}, q_idx not in BLOCK_MASK[a][b]
                    # And for all a in S\{p_idx}, q_idx not in BLOCK_MASK[a][q_idx]... wait
                    # The condition is: no three points form isosceles.
                    # For the new set, we need to check all triples.
                    # But since S\{p_idx} is already no-isosceles, we only need to check
                    # triples involving q_idx.
                    # For each pair (a, b) in S\{p_idx}: q_idx must not be in BLOCK_MASK[a][b]
                    # For each a in S\{p_idx}: the triple (q_idx, a, b) for b in S\{p_idx}\{a}
                    #   means: a not in BLOCK_MASK[q_idx][b] and b not in BLOCK_MASK[q_idx][a]
                    #   which is equivalent to: for each a in S\{p_idx}, for each b in S\{p_idx}\{a}:
                    #   a not in BLOCK_MASK[q_idx][b] (i.e., q_idx, b, a not isosceles with apex at q_idx or b or a)
                    #   This is the same as: for each a in S\{p_idx}, q_idx not in BLOCK_MASK[a][b] for all b != a in S\{p_idx}
                    #   Wait, BLOCK_MASK[a][b] contains x such that {a,b,x} is isosceles.
                    #   So we need: for all a != b in S\{p_idx}, q_idx not in BLOCK_MASK[a][b]
                    #   And for all a in S\{p_idx}, a not in BLOCK_MASK[q_idx][b] for all b in S\{p_idx}\{a}
                    #   But BLOCK_MASK[q_idx][b] contains x such that {q_idx, b, x} is isosceles.
                    #   So a in BLOCK_MASK[q_idx][b] means {q_idx, b, a} is isosceles.
                    #   This is the same as q_idx in BLOCK_MASK[a][b] by symmetry!
                    #   Because {a,b,x} is isosceles iff {x,a,b} is isosceles.
                    #   So BLOCK_MASK[a][b] = BLOCK_MASK[b][a] = {x : {a,b,x} is isosceles}
                    #   And a in BLOCK_MASK[q_idx][b] iff {q_idx, b, a} is isosceles iff q_idx in BLOCK_MASK[a][b]
                    #   So the condition reduces to: for all a != b in S\{p_idx}, q_idx not in BLOCK_MASK[a][b]
                    
                    valid_no_iso = True
                    for i in range(len(s_list2)):
                        a = s_list2[i]
                        if a == p_idx:
                            continue
                        for j in range(i + 1, len(s_list2)):
                            b = s_list2[j]
                            if b == p_idx:
                                continue
                            if (BLOCK_MASK[a][b] >> q_idx) & 1:
                                valid_no_iso = False
                                break
                        if not valid_no_iso:
                            break
                    
                    if not valid_no_iso:
                        continue
                    
                    # Check maximality of S\{p_idx} ∪ {q_idx}
                    # Build new blocked_mask
                    new_blocked_mask = 0
                    s_list_new = list(S)
                    s_list_new.remove(p_idx)
                    s_list_new.append(q_idx)
                    for i in range(len(s_list_new)):
                        a = s_list_new[i]
                        for j in range(i + 1, len(s_list_new)):
                            b = s_list_new[j]
                            new_blocked_mask |= BLOCK_MASK[a][b]
                    
                    new_s_mask = 0
                    for i in S:
                        if i != p_idx:
                            new_s_mask |= (1 << i)
                    new_s_mask |= (1 << q_idx)
                    not_in_new = FULL_MASK ^ new_s_mask
                    if (not_in_new & ~new_blocked_mask) != 0:
                        continue
                    
                    # Valid swap
                    S = set(S)
                    S.discard(p_idx)
                    S.add(q_idx)
                    blocked_mask = new_blocked_mask
                    current_size = len(S)
                    found_improvement = True
                    
                    if current_size < best_size_local:
                        best_size_local = current_size
                        best_set_local = set(S)
                    
                    break  # Restart outer loop
            
            if not found_improvement:
                # No swaps found, break
                break
        
        return best_size_local, best_set_local
    
    # Main loop
    restarts = 0
    while time.time() < deadline:
        S_init = greedy_initial(restarts)
        time_limit = min(deadline, start_time + 280)
        size, S_final = local_search(S_init, restarts * 7919 + 1, time_limit)
        
        if size < best_size:
            best_size = size
            best_set = S_final
        
        restarts += 1
    
    if best_set is None:
        S = set()
        for p in range(NUM_POINTS):
            valid = True
            for q in S:
                for r in S:
                    if r == q:
                        continue
                    if (BLOCK_MASK[q][r] >> p) & 1:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S.add(p)
        best_set = S
    
    return [POINTS[i] for i in best_set]

if __name__ == "__main__":
    s = construct()
    print(len(s))
