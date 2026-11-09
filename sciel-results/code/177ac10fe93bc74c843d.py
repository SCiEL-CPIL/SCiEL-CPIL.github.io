"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START

N = 256
MAX_D = 85 # Max difference for 4-AP: 1 + 3*85 = 256

# Precompute 4-AP forbidden witnesses: for each x, list of bitmasks of 3 elements
# such that {x} U witness forms a 4-AP.
_4ap_witnesses = {}

def _precompute_4ap_witnesses():
    global _4ap_witnesses
    for x in range(1, N+1):
        witnesses = []
        # x as pos 0: x, x+d, x+2d, x+3d
        for d in range(1, (N - x) // 3 + 1):
            e1, e2, e3 = x+d, x+2*d, x+3*d
            if e3 <= N:
                bm = (1 << (e1-1)) | (1 << (e2-1)) | (1 << (e3-1))
                witnesses.append(bm)
        # x as pos 1: x-d, x, x+d, x+2d
        for d in range(1, min(x-1, (N-x)//2) + 1):
            e0, e2, e3 = x-d, x+d, x+2*d
            if e0 >= 1 and e3 <= N:
                bm = (1 << (e0-1)) | (1 << (e2-1)) | (1 << (e3-1))
                witnesses.append(bm)
        # x as pos 2: x-2d, x-d, x, x+d
        for d in range(1, min((x-1)//2, N-x) + 1):
            e0, e1, e3 = x-2*d, x-d, x+d
            if e0 >= 1 and e3 <= N:
                bm = (1 << (e0-1)) | (1 << (e1-1)) | (1 << (e3-1))
                witnesses.append(bm)
        # x as pos 3: x-3d, x-2d, x-d, x
        for d in range(1, (x-1)//3 + 1):
            e0, e1, e2 = x-3*d, x-2*d, x-d
            if e0 >= 1:
                bm = (1 << (e0-1)) | (1 << (e1-1)) | (1 << (e2-1))
                witnesses.append(bm)
        _4ap_witnesses[x] = witnesses

_precompute_4ap_witnesses()

# Precompute 3-AP pairs for delta calculation
_mid_pairs = {}
_start_pairs = {}
_end_pairs = {}

def _precompute_3ap_pairs():
    global _mid_pairs, _start_pairs, _end_pairs
    for x in range(1, N+1):
        mid = []
        for d in range(1, min(x-1, N-x) + 1):
            a, c = x - d, x + d
            mid.append((a, c))
        _mid_pairs[x] = mid
        
        start = []
        for d in range(1, (N-x)//2 + 1):
            b, c = x + d, x + 2*d
            start.append((b, c))
        _start_pairs[x] = start
        
        end = []
        for d in range(1, (x-1)//2 + 1):
            a, b = x - 2*d, x - d
            end.append((a, b))
        _end_pairs[x] = end

_precompute_3ap_pairs()

def _count_3aps_bitmask(mask):
    """Count distinct 3-term APs in the set represented by bitmask."""
    count = 0
    for x in range(1, N+1):
        if not ((mask >> (x-1)) & 1):
            continue
        for a, c in _mid_pairs[x]:
            if ((mask >> (a-1)) & 1) and ((mask >> (c-1)) & 1):
                count += 1
    return count

def _delta_3ap(mask, x):
    """Calculate change in 3-AP count when adding x (assuming x not in mask)."""
    delta = 0
    # x as middle
    for a, c in _mid_pairs[x]:
        if ((mask >> (a-1)) & 1) and ((mask >> (c-1)) & 1):
            delta += 1
    # x as start
    for b, c in _start_pairs[x]:
        if ((mask >> (b-1)) & 1) and ((mask >> (c-1)) & 1):
            delta += 1
    # x as end
    for a, b in _end_pairs[x]:
        if ((mask >> (a-1)) & 1) and ((mask >> (b-1)) & 1):
            delta += 1
    return delta

def _creates_4ap(mask, x):
    """Check if adding x creates a 4-AP. x is NOT in mask."""
    for witness_bm in _4ap_witnesses[x]:
        if (mask & witness_bm) == witness_bm:
            return True
    return False

def _construct_greedy(order=None):
    """Greedy construction of a set with no 4-APs."""
    mask = 0
    if order is None:
        order = list(range(1, N+1))
    for x in order:
        if not _creates_4ap(mask, x):
            mask |= (1 << (x-1))
    return mask

def _local_search_sa(mask, time_limit, start_temp=5.0):
    """Simulated Annealing local search."""
    start_time = time.monotonic()
    best_mask = mask
    best_score = _count_3aps_bitmask(mask)
    current_mask = mask
    current_score = best_score
    
    elements = set()
    for i in range(N):
        if (current_mask >> i) & 1:
            elements.add(i+1)
            
    # Precompute list of all elements for random selection
    all_elements = list(range(1, N+1))
    
    T = start_temp
    cooling_rate = 0.99995
    min_T = 0.01
    
    # Use math.exp for scalar speed
    exp = math.exp
    
    while time.monotonic() < start_time + time_limit:
        r = random.random()
        
        if r < 0.4:
            # Add element
            # Pick a random element not in set
            # Efficient way: pick random from all, check if in set.
            # Since set size is ~100-150, collision rate is moderate.
            # To be faster, we could maintain a list of available elements.
            # Let's stick to simple random pick for now, N=256 is small.
            x = random.choice(all_elements)
            if x in elements:
                continue
            if _creates_4ap(current_mask, x):
                continue
            delta = _delta_3ap(current_mask, x)
            new_score = current_score + delta
            if new_score > current_score or random.random() < exp((new_score - current_score) / T):
                current_mask |= (1 << (x-1))
                current_score = new_score
                elements.add(x)
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
        elif r < 0.6:
            # Remove element
            if not elements:
                continue
            # Random choice from set
            # Converting set to list every time is O(N). 
            # Alternative: random index from 0 to N-1, check if in set.
            x = random.choice(all_elements)
            if x not in elements:
                continue
            delta = _delta_3ap(current_mask, x)
            # Removing x changes score by -delta (since delta was the contribution of x)
            # Wait, _delta_3ap calculates the gain of ADDING x to a set.
            # So removing x from a set that HAS x results in a loss of delta.
            # So new_score = current_score - delta
            new_score = current_score - delta
            if new_score > current_score or random.random() < exp((new_score - current_score) / T):
                current_mask &= ~(1 << (x-1))
                current_score = new_score
                elements.discard(x)
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
        else:
            # Swap
            if not elements:
                continue
            x_remove = random.choice(all_elements)
            if x_remove not in elements:
                continue
            x_add = random.choice(all_elements)
            if x_add == x_remove or x_add in elements:
                continue
            
            temp_mask = current_mask & ~(1 << (x_remove-1))
            if _creates_4ap(temp_mask, x_add):
                continue
            
            # Delta for removing x_remove
            delta_rem = _delta_3ap(current_mask, x_remove)
            # Delta for adding x_add to temp_mask
            delta_add = _delta_3ap(temp_mask, x_add)
            
            new_score = current_score - delta_rem + delta_add
            
            if new_score > current_score or random.random() < exp((new_score - current_score) / T):
                current_mask = temp_mask | (1 << (x_add-1))
                current_score = new_score
                elements.discard(x_remove)
                elements.add(x_add)
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
        
        T = max(min_T, T * cooling_rate)

    return best_mask, best_score

def _deterministic_polish(mask, score):
    """Try to improve the set by adding any valid element that increases score."""
    improved = True
    while improved:
        improved = False
        for x in range(1, N+1):
            if (mask >> (x-1)) & 1:
                continue
            if _creates_4ap(mask, x):
                continue
            delta = _delta_3ap(mask, x)
            if delta > 0:
                mask |= (1 << (x-1))
                score += delta
                improved = True
    return mask, score

def construct():
    start_time = time.monotonic()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    best_overall_mask = 0
    best_overall_score = 0
    
    # Phase 1: Generate diverse initial solutions
    initial_candidates = []
    
    # 1. Standard Greedy
    m = _construct_greedy()
    s = _count_3aps_bitmask(m)
    initial_candidates.append((m, s))
    
    # 2. Random Order Greedies (5 iterations)
    for _ in range(5):
        if time.monotonic() > deadline - 20: break
        order = list(range(1, N+1))
        random.shuffle(order)
        m = _construct_greedy(order)
        s = _count_3aps_bitmask(m)
        initial_candidates.append((m, s))
        
    # 3. "Density" Greedy: Prioritize elements with high potential 3-APs?
    # Simple heuristic: Sort by number of possible 3-AP partners?
    # This is complex to precompute dynamically. Let's stick to random shuffles.
    
    # Select top 3 initial candidates for SA
    initial_candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = initial_candidates[:3]
    
    # Phase 2: Simulated Annealing on top candidates
    # Split remaining time among top candidates
    remaining_time = deadline - time.monotonic()
    if remaining_time > 5:
        # Allocate time: 50% to best, 30% to second, 20% to third
        times = [0.5 * remaining_time, 0.3 * remaining_time, 0.2 * remaining_time]
        
        for i, (m, s) in enumerate(top_candidates):
            if i >= len(times): break
            if time.monotonic() > deadline - 2: break
            
            t_alloc = times[i]
            if t_alloc < 1: continue
            
            # Run SA
            sa_mask, sa_score = _local_search_sa(m, t_alloc)
            
            # Polish
            pol_mask, pol_score = _deterministic_polish(sa_mask, sa_score)
            
            if pol_score > best_overall_score:
                best_overall_score = pol_score
                best_overall_mask = pol_mask
    
    # Fallback if no SA ran or failed
    if best_overall_mask == 0:
        best_overall_mask, best_overall_score = top_candidates[0]
        best_overall_mask, best_overall_score = _deterministic_polish(best_overall_mask, best_overall_score)

    result = [i+1 for i in range(N) if (best_overall_mask >> i) & 1]
    
    if not result:
        result = [1]
        
    return result

# EVOLVE-BLOCK-END
