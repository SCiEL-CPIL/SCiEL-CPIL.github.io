import random
import time
import numpy as np

N = 128
APS = []
POS_APS = [[] for _ in range(N)]
NUM_APS = 0

def _init_data():
    global APS, POS_APS, NUM_APS
    # 0-indexed positions 0..127
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            idx = NUM_APS
            APS.append((a, b, c, e))
            POS_APS[a].append(idx)
            POS_APS[b].append(idx)
            POS_APS[c].append(idx)
            POS_APS[e].append(idx)
            NUM_APS += 1

_init_data()

def _local_search(k, seed, time_budget):
    rng = random.Random(seed)
    start = time.time()
    colors = [rng.randrange(k) for _ in range(N)]
    ap_hist = []
    ap_distinct = []
    for (a, b, c, e) in APS:
        h = [0] * k
        h[colors[a]] += 1
        h[colors[b]] += 1
        h[colors[c]] += 1
        h[colors[e]] += 1
        ap_hist.append(h)
        ap_distinct.append(sum(1 for x in h if x))
    total_bad = sum(1 for d in ap_distinct if d < 3)
    best_colors = colors[:]
    best_bad = total_bad
    bad_aps = [i for i in range(NUM_APS) if ap_distinct[i] < 3]
    step = 0
    while time.time() - start < time_budget:
        if total_bad == 0:
            return colors
        step += 1
        if step % 500 == 0:
            bad_aps = [i for i in range(NUM_APS) if ap_distinct[i] < 3]
            if total_bad != len(bad_aps):
                total_bad = len(bad_aps)
        if not bad_aps:
            bad_aps = [i for i in range(NUM_APS) if ap_distinct[i] < 3]
            total_bad = len(bad_aps)
            if not bad_aps:
                return colors
        ap = bad_aps[rng.randrange(len(bad_aps))]
        a, b, c, e = APS[ap]
        i = (a, b, c, e)[rng.randrange(4)]
        old = colors[i]
        new = rng.randrange(k)
        if new == old:
            new = (new + 1) % k
        delta = 0
        for ap_idx in POS_APS[i]:
            h = ap_hist[ap_idx]
            D = ap_distinct[ap_idx]
            ho = h[old]
            hn = h[new]
            dD = 0
            if ho == 1:
                dD -= 1
            if hn == 0:
                dD += 1
            Dn = D + dD
            delta += (1 if Dn < 3 else 0) - (1 if D < 3 else 0)
        if delta < 0:
            accept = True
        elif delta == 0:
            accept = rng.random() < 0.25
        else:
            accept = rng.random() < 0.08
        if not accept:
            continue
        colors[i] = new
        for ap_idx in POS_APS[i]:
            h = ap_hist[ap_idx]
            D = ap_distinct[ap_idx]
            h[old] -= 1
            if h[old] == 0:
                D -= 1
            h[new] += 1
            if h[new] == 1:
                D += 1
            ap_distinct[ap_idx] = D
        total_bad += delta
        if total_bad < best_bad:
            best_bad = total_bad
            best_colors = colors[:]
    return best_colors

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    best_colors = None
    best_num_colors = N + 1
    
    # Try to find a solution with few colors
    # Start with small k, increase if no solution found
    # We allocate time to each k attempt
    # If k is too small, we might not find a solution, so we give it less time
    # If k is larger, it's easier to find a solution, but we want to minimize colors used
    
    # Heuristic: The minimum number of colors for 4-APs is likely small.
    # Let's try k=4, 5, 6, 7...
    
    # We will run a loop that increases k and allocates time.
    # If we find a valid solution for k, we record the distinct colors used.
    # We continue to try higher k only if we haven't found a solution with very few colors yet?
    # Actually, if we find a valid solution with k=4 using 4 colors, that's likely optimal or close.
    # If we find a valid solution with k=5 using 5 colors, it's worse.
    # So we should prioritize finding *any* valid solution with small k.
    
    # Strategy:
    # 1. Try k=4 for 20s. If valid, record.
    # 2. Try k=5 for 20s. If valid, record.
    # 3. Try k=6 for 20s. If valid, record.
    # 4. If time remains, try to improve the best found by running local search on the best found solution with a larger k (to allow moves) but tracking distinct colors?
    #    The provided _local_search minimizes bad APs. It doesn't minimize distinct colors directly.
    #    However, if we start with a solution that uses few colors, and we allow moves to any of k colors, we might increase the distinct colors.
    #    To minimize distinct colors, we should restrict `new` color selection to existing colors or a small set.
    
    # Let's stick to the provided _local_search design.
    # It returns a coloring. We evaluate it.
    
    k_attempts = [4, 5, 6, 7, 8, 10, 12]
    
    for k in k_attempts:
        if time.time() > deadline - 10.0:
            break
            
        # Allocate time based on k? Or fixed slices?
        # Let's give 10s to each k attempt initially, or scale it.
        # If we find a solution with k=4, we might stop early if we are confident?
        # But maybe k=4 is impossible?
        
        time_slice = min(15.0, deadline - time.time() - 5.0)
        if time_slice <= 0:
            break
            
        seed = int(time.time() * 1000) % (2**32)
        
        res = _local_search(k, seed, time_slice)
        
        # Evaluate result
        # Check if valid
        bad_count = 0
        distinct_colors = set(res)
        
        # Fast validity check
        # We can reuse the histogram logic or just re-check APs
        # For N=128, re-checking ~2600 APs is fast.
        valid = True
        for (a, b, c, e) in APS:
            ca, cb, cc, ce = res[a], res[b], res[c], res[e]
            if len({ca, cb, cc, ce}) < 3:
                valid = False
                break
        
        if valid:
            num_used = len(distinct_colors)
            if num_used < best_num_colors:
                best_num_colors = num_used
                best_colors = res[:]
        
        # If we found a solution with 4 colors, it's likely optimal (min is at least 3, but 3 might be impossible for 4-APs in 128? 
        # Actually, 3 colors might work. Let's add k=3 to attempts?
        # The problem says "Minimize the number of labels". 
        # If k=3 is possible, we should try it.
        
    # If we didn't find any valid solution, we need to return *something*.
    # But the task implies a valid solution exists.
    # If best_colors is None, we must run a longer search with larger k to ensure validity.
    if best_colors is None:
        # Fallback: Run with a large k to ensure we find *a* valid solution
        time_slice = max(1.0, deadline - time.time() - 1.0)
        if time_slice > 0:
            res = _local_search(128, 42, time_slice)
            # Verify
            valid = True
            for (a, b, c, e) in APS:
                ca, cb, cc, ce = res[a], res[b], res[c], res[e]
                if len({ca, cb, cc, ce}) < 3:
                    valid = False
                    break
            if valid:
                best_colors = res
            else:
                # Should not happen if search is long enough, but just in case
                # Return a random coloring? No, must be valid.
                # Try once more with more time?
                pass

    if best_colors is None:
        # Emergency: Generate a random coloring and hope? Or force validity?
        # This should not occur.
        return [0] * N

    return best_colors
