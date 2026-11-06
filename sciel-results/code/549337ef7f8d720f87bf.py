# EVOLVE-BLOCK-START
import numpy as np
import time
import random

N = 256
MAX_D = N // 3

def _thue_morse_fast(n):
    """Generate Thue-Morse sequence: t(i) = popcount(i) % 2."""
    c = np.zeros(n, dtype=np.int8)
    for i in range(n):
        c[i] = bin(i).count('1') % 2
    return c

def _precompute_aps():
    """
    Precompute all 4-term APs and for each position, the APs it belongs to.
    """
    aps = []
    pos_to_aps = [[] for _ in range(N)]
    ap_index = 0
    for d in range(1, MAX_D + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            p0 = a - 1
            p1 = a + d - 1
            p2 = a + 2 * d - 1
            p3 = a + 3 * d - 1
            aps.append((a, d))
            pos_to_aps[p0].append(ap_index)
            pos_to_aps[p1].append(ap_index)
            pos_to_aps[p2].append(ap_index)
            pos_to_aps[p3].append(ap_index)
            ap_index += 1
    return aps, pos_to_aps

APS, POS_TO_APS = _precompute_aps()
NUM_APS = len(APS)

def _count_mono4(c):
    """Count monochromatic 4-term APs."""
    count = 0
    for a, d in APS:
        p0 = a - 1
        p1 = a + d - 1
        p2 = a + 2 * d - 1
        p3 = a + 3 * d - 1
        if c[p0] == c[p1] == c[p2] == c[p3]:
            count += 1
    return count

def _delta_flip(c, p, current_count):
    """Calculate change in mono-AP count if we flip position p."""
    old_val = c[p]
    new_val = 1 - old_val
    delta = 0
    
    for ap_idx in POS_TO_APS[p]:
        a, d = APS[ap_idx]
        p0 = a - 1
        p1 = a + d - 1
        p2 = a + 2 * d - 1
        p3 = a + 3 * d - 1
        
        v0 = c[p0]
        v1 = c[p1]
        v2 = c[p2]
        v3 = c[p3]
        
        was_mono = (v0 == v1 and v1 == v2 and v2 == v3)
        
        # Determine new values
        if p == p0:
            nv0, nv1, nv2, nv3 = new_val, v1, v2, v3
        elif p == p1:
            nv0, nv1, nv2, nv3 = v0, new_val, v2, v3
        elif p == p2:
            nv0, nv1, nv2, nv3 = v0, v1, new_val, v3
        else:
            nv0, nv1, nv2, nv3 = v0, v1, v2, new_val
            
        is_mono_after = (nv0 == nv1 and nv1 == nv2 and nv2 == nv3)
        
        if was_mono and not is_mono_after:
            delta -= 1
        elif not was_mono and is_mono_after:
            delta += 1
            
    return current_count + delta

def _local_search_fast(c, time_limit):
    """Fast local search using delta updates."""
    start = time.time()
    best_c = c.copy()
    best_count = _count_mono4(best_c)
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        order = list(range(N))
        random.shuffle(order)
        
        for p in order:
            if (time.time() - start) >= time_limit:
                break
            new_count = _delta_flip(best_c, p, best_count)
            if new_count < best_count:
                best_c[p] = 1 - best_c[p]
                best_count = new_count
                improved = True
    
    return best_c, best_count

def construct():
    deadline = time.time() + 140
    
    # Start with Thue-Morse
    best_c = _thue_morse_fast(N)
    best_count = _count_mono4(best_c)
    
    # Phase 1: Intensive local search on best
    remaining = deadline - time.time()
    if remaining > 5:
        best_c, best_count = _local_search_fast(best_c, min(remaining - 5, 20))
    
    # Phase 2: Diversification via Random Restarts and Crossover
    # Maintain a small population
    pop = [(best_c.copy(), best_count)]
    
    while time.time() < deadline - 5:
        # Generate new candidate
        if random.random() < 0.7:
            # Mutate best
            parent = pop[0][0]
            c_new = parent.copy()
            num_flips = random.randint(2, 8)
            for _ in range(num_flips):
                p = random.randint(0, N - 1)
                c_new[p] = 1 - c_new[p]
        else:
            # Crossover or Random
            if len(pop) > 1 and random.random() < 0.5:
                p1, p2 = random.sample(range(len(pop)), 2)
                c1 = pop[p1][0]
                c2 = pop[p2][0]
                # Uniform crossover
                mask = np.random.randint(0, 2, N, dtype=np.int8)
                c_new = np.where(mask == 1, c1, c2).astype(np.int8)
            else:
                c_new = np.random.randint(0, 2, N, dtype=np.int8)
        
        # Local search on new candidate
        # Give it a short budget
        t_limit = min(1.5, deadline - time.time() - 5)
        if t_limit <= 0: break
        c_local, count_local = _local_search_fast(c_new, t_limit)
        
        # Update population
        if count_local < best_count:
            best_count = count_local
            best_c = c_local
            pop = [(best_c.copy(), best_count)]
        else:
            # Keep top 5
            pop.append((c_local.copy(), count_local))
            pop.sort(key=lambda x: x[1])
            if len(pop) > 5:
                pop = pop[:5]
                
    # Final polish
    remaining = deadline - time.time()
    if remaining > 1:
        best_c, best_count = _local_search_fast(best_c, remaining - 1)

    result = [int(x) for x in best_c]
    assert len(result) == 256
    assert all(x in (0, 1) for x in result)
    return result
# EVOLVE-BLOCK-END
