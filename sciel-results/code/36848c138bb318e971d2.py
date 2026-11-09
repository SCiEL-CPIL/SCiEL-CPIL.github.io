import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    deadline = 55.0  # seconds safety margin

    # Precompute all APs: (start, step) with start in [0, n-1], step >= 1
    # AP indices: start, start+step, start+2*step, ... < n
    # We need to track sum for each AP.
    # To make incremental updates fast, for each index i, store list of (ap_id, sign_in_ap)
    # Actually, each AP has a fixed set of indices. When we flip x[i], the AP sum changes by -2*x[i]_old = ±2.
    
    # Build AP list
    aps = []  # list of lists of indices
    index_to_aps = [[] for _ in range(n)]  # for each index, list of ap_ids
    
    ap_id = 0
    for step in range(1, n + 1):
        for start in range(0, n):
            # Only consider APs with at least 2 elements? No, the task says "including one".
            # But singletons: sum is just x[i], |x[i]| = 1. So max is at least 1.
            # For step >= 1, start in [0, n-1], the AP is start, start+step, ...
            # But if step > start, then only one element.
            # Let's include all APs with length >= 1.
            # Actually, for efficiency, let's include all.
            idxs = list(range(start, n, step))
            if not idxs:
                continue
            for idx in idxs:
                index_to_aps[idx].append(ap_id)
            aps.append(idxs)
            ap_id += 1
    
    num_aps = len(aps)
    
    # Initialize signs
    rng = random.Random(12345)
    signs = [rng.choice([-1, 1]) for _ in range(n)]
    
    # Compute initial AP sums
    ap_sums = [0] * num_aps
    for i in range(n):
        for ap_id_i in index_to_aps[i]:
            ap_sums[ap_id_i] += signs[i]
    
    # Find current max abs sum
    def current_max():
        return max(abs(s) for s in ap_sums)
    
    best_signs = signs[:]
    best_max = current_max()
    
    # Hill climbing with SA
    T = 5.0
    cooling = 0.9995
    
    while time.time() - start_time < deadline:
        # Pick a random index to flip
        i = rng.randrange(n)
        old_sign = signs[i]
        new_sign = -old_sign
        
        # Update AP sums
        delta = 2 * new_sign  # change in sum = new_sign - old_sign = 2*new_sign
        # Actually: new_sum = old_sum + (new_sign - old_sign) = old_sum + 2*new_sign
        # Since new_sign = -old_sign, new_sign - old_sign = -2*old_sign = 2*new_sign
        for ap_id_i in index_to_aps[i]:
            ap_sums[ap_id_i] += delta
        
        new_max = current_max()
        
        # Accept if better, or with probability exp(-(new_max - best_max)/T) if worse
        if new_max <= best_max:
            # Accept
            signs[i] = new_sign
            best_signs = signs[:]
            best_max = new_max
        else:
            # SA acceptance
            diff = new_max - best_max
            if rng.random() < math.exp(-diff / T):
                signs[i] = new_sign
            else:
                # Revert
                for ap_id_i in index_to_aps[i]:
                    ap_sums[ap_id_i] -= delta
                signs[i] = old_sign
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    return best_signs
