import time
import random
import math

N = 256
DEADLINE = 55.0  # Safety margin under 90s kill

# Precomputation: for each index x (0..N-1), store partner tuples.
# ap3_partners[x] contains (y, z) such that {x, y, z} form a 3-AP.
# ap4_partners[x] contains (y, z, w) such that {x, y, z, w} form a 4-AP.
ap3_partners = [[] for _ in range(N)]
ap4_partners = [[] for _ in range(N)]

def _precompute():
    # 3-APs: a, a+d, a+2d
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            ia, ib, ic = a - 1, b - 1, c - 1
            ap3_partners[ia].append((ib, ic))
            ap3_partners[ib].append((ia, ic))
            ap3_partners[ic].append((ia, ib))
            
    # 4-APs: a, a+d, a+2d, a+3d
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ia, ib, ic, ie = a - 1, b - 1, c - 1, e - 1
            ap4_partners[ia].append((ib, ic, ie))
            ap4_partners[ib].append((ia, ic, ie))
            ap4_partners[ic].append((ia, ib, ie))
            ap4_partners[ie].append((ia, ib, ic))

_precompute()

def _has_4ap(mask, x):
    """Check if adding x to mask creates a 4-AP."""
    for y, z, w in ap4_partners[x]:
        if (mask >> y) & 1 and (mask >> z) & 1 and (mask >> w) & 1:
            return True
    return False

def _delta_add(mask, x):
    """Number of new 3-APs created by adding x to mask."""
    cnt = 0
    for y, z in ap3_partners[x]:
        if (mask >> y) & 1 and (mask >> z) & 1:
            cnt += 1
    return cnt

def _delta_remove(mask, x):
    """Number of 3-APs destroyed by removing x from mask."""
    cnt = 0
    for y, z in ap3_partners[x]:
        if (mask >> y) & 1 and (mask >> z) & 1:
            cnt += 1
    return cnt

def _count_3aps(mask):
    """Total number of distinct 3-APs in mask."""
    total = 0
    for x in range(N):
        if (mask >> x) & 1:
            total += _delta_add(mask ^ (1 << x), x)
    return total // 3

def _greedy_seed():
    mask = 0
    for x in range(N):
        if not _has_4ap(mask, x):
            mask |= (1 << x)
    return mask

def _present_list(mask):
    return [i for i in range(N) if (mask >> i) & 1]

def _absent_list(mask):
    return [i for i in range(N) if not (mask >> i) & 1]

def _strict_local_search(mask, score, start_time):
    """
    Strict hill climbing. Tries all single-bit flips and swaps.
    Returns updated mask and score.
    """
    improved = True
    while improved:
        if time.time() - start_time > DEADLINE:
            break
        improved = False
        
        # 1. Try single additions
        present = _present_list(mask)
        absent = _absent_list(mask)
        
        # Additions
        for x in absent:
            if _has_4ap(mask, x):
                continue
            delta = _delta_add(mask, x)
            if delta > 0:
                mask |= (1 << x)
                score += delta
                improved = True
                break # Restart scan after change
        
        if improved:
            continue

        # 2. Try single removals (only if it helps unlock better additions, 
        #    but pure removal reduces score. We only do this if it leads to 
        #    a swap improvement immediately, or we skip pure removals in strict mode
        #    unless we are stuck and want to explore. In strict hill climbing for
        #    maximization, pure removals are never beneficial unless they are
        #    part of a swap. So we skip pure removals here.)
        
        # 3. Try swaps: Remove x (present), Add y (absent)
        # To be efficient, we sample or iterate. N=256, so N^2 is ~65k.
        # We can do a full pass of swaps.
        present = _present_list(mask)
        absent = _absent_list(mask)
        
        # Shuffle to randomize order and avoid bias
        random.shuffle(present)
        random.shuffle(absent)
        
        found_swap = False
        for x in present:
            if time.time() - start_time > DEADLINE:
                break
            mask_wo = mask ^ (1 << x)
            loss = _delta_remove(mask, x)
            
            for y in absent:
                if _has_4ap(mask_wo, y):
                    continue
                gain = _delta_add(mask_wo, y)
                delta = gain - loss
                if delta > 0:
                    mask = mask_wo | (1 << y)
                    score += delta
                    improved = True
                    found_swap = True
                    break
            if found_swap:
                break
        
    return mask, score

def _simulated_annealing(mask, score, start_time, duration=10.0):
    """
    Simulated Annealing to escape local optima.
    """
    T0 = 5.0
    Tend = 0.05
    call_start = time.time()
    end_time = call_start + duration
    
    # Pre-compute lists for efficiency? No, they change.
    
    while time.time() < end_time and time.time() - start_time < DEADLINE:
        elapsed = time.time() - call_start
        progress = elapsed / duration
        T = T0 * math.exp(-10 * progress) # Cooldown curve
        if T < Tend:
            T = Tend

        present = _present_list(mask)
        absent = _absent_list(mask)
        
        if not present:
            break
            
        # Choose move type: 50% Add, 30% Swap, 20% Remove
        r = random.random()
        
        if r < 0.5 and absent:
            # Try adding a random absent element
            x = random.choice(absent)
            if _has_4ap(mask, x):
                continue
            delta = _delta_add(mask, x)
            if delta > 0 or random.random() < math.exp(delta / T):
                mask |= (1 << x)
                score += delta
                
        elif r < 0.8 and present and absent:
            # Try swapping a random present with a random absent
            rem = random.choice(present)
            add = random.choice(absent)
            mask_wo = mask ^ (1 << rem)
            if _has_4ap(mask_wo, add):
                continue
            loss = _delta_remove(mask, rem)
            gain = _delta_add(mask_wo, add)
            delta = gain - loss
            if delta > 0 or random.random() < math.exp(delta / T):
                mask = mask_wo | (1 << add)
                score += delta
                
        else:
            # Try removing a random present element
            if len(present) > 1:
                rem = random.choice(present)
                loss = _delta_remove(mask, rem)
                # Note: loss is positive (number of APs lost), so delta = -loss
                delta = -loss
                if delta > 0 or random.random() < math.exp(delta / T):
                    mask ^= (1 << rem)
                    score += delta

    return mask, score

def construct():
    start = time.time()
    random.seed(12345)

    # Initial Solution
    mask = _greedy_seed()
    score = _count_3aps(mask)
    
    best_mask = mask
    best_score = score
    
    # Phase 1: Strict Local Search
    mask, score = _strict_local_search(mask, score, start)
    if score > best_score:
        best_score = score
        best_mask = mask

    # Phase 2: SA with Restarts
    # We run SA for short bursts, then polish, then perturb.
    
    while time.time() - start < DEADLINE:
        # SA Phase
        sa_duration = 2.0
        mask, score = _simulated_annealing(mask, score, start, duration=sa_duration)
        
        # Polish Phase
        mask, score = _strict_local_search(mask, score, start)
        
        if score > best_score:
            best_score = score
            best_mask = mask

        # Perturb Phase: Remove a few random elements to escape basin
        elems = _present_list(mask)
        if not elems:
            break
            
        # Remove 3-5 random elements
        k = min(5, max(1, len(elems) // 20))
        to_remove = random.sample(elems, k)
        for rem in to_remove:
            score -= _delta_remove(mask, rem)
            mask ^= (1 << rem)
            
        if not _present_list(mask):
            mask = _greedy_seed()
            score = _count_3aps(mask)
            
        # Occasionally restart from fresh greedy
        if random.random() < 0.1:
            fresh = _greedy_seed()
            fresh_score = _count_3aps(fresh)
            if fresh_score > score:
                mask = fresh
                score = fresh_score
                if score > best_score:
                    best_score = score
                    best_mask = mask

    # Convert best_mask to sorted list of elements (1-indexed)
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    return sorted(result)
