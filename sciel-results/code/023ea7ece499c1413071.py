import time
import random
import numpy as np

def _next_prime(n):
    """Find the next prime number greater than or equal to n."""
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(np.sqrt(num)) + 1):
            if num % i == 0:
                return False
        return True
    
    while True:
        if is_prime(n):
            return n
        n += 1

def _legendre_symbol(a, p):
    """Compute the Legendre symbol (a/p)."""
    if a % p == 0:
        return 0
    return pow(a, (p - 1) // 2, p)

def _generate_legendre_sequence(n, p):
    """Generate a Legendre sequence of length n using prime p."""
    seq = []
    for i in range(1, n + 1):
        ls = _legendre_symbol(i, p)
        if ls == 0:
            seq.append(1) # Should not happen if p > n and i < p
        elif ls == 1:
            seq.append(1)
        else:
            seq.append(-1)
    return seq

def _build_ap_structure(n):
    """
    Pre-compute all arithmetic progressions.
    Returns:
        ap_indices: list of arrays of indices
        index_to_aps: list of lists of (ap_id, pos)
    """
    ap_indices = []
    index_to_aps = [[] for _ in range(n)]
    
    ap_id = 0
    for step in range(1, n + 1):
        for start in range(n):
            indices = list(range(start, n, step))
            if len(indices) == 0:
                continue
            
            ap_indices.append(np.array(indices, dtype=np.int32))
            
            for pos, idx in enumerate(indices):
                index_to_aps[idx].append((ap_id, pos))
            
            ap_id += 1
            
    return ap_indices, index_to_aps

def _init_prefix_sums(signs, ap_indices):
    """Compute initial sums for all APs."""
    signs_arr = np.array(signs, dtype=np.int64)
    ap_sums = np.zeros(len(ap_indices), dtype=np.int64)
    for i, indices in enumerate(ap_indices):
        ap_sums[i] = np.sum(signs_arr[indices])
    return ap_sums

def _get_max_disc(ap_sums):
    """Get max absolute sum and count of APs achieving it."""
    abs_sums = np.abs(ap_sums)
    max_val = int(np.max(abs_sums))
    count_max = int(np.sum(abs_sums == max_val))
    return max_val, count_max

def _eval_flip(signs, ap_sums, index_to_aps, i, old_val, new_val, current_max, count_max):
    """
    Evaluate the effect of flipping index i.
    Returns:
        new_max, new_count_max, delta_ap_sums (list of (ap_id, delta))
    """
    delta = new_val - old_val
    affected_aps = index_to_aps[i]
    
    old_abs_max = current_max
    new_max = current_max
    new_count_max = count_max
    
    # We need to update the count of maxes carefully.
    # If an AP had abs_sum == old_max, and its new abs_sum != old_max, count decreases.
    # If an AP had abs_sum < old_max, and its new abs_sum == new_max (which might be > old_max or == old_max), count increases.
    
    # This is tricky to do in one pass without storing old values.
    # Let's just compute the new abs sums for affected APs and update the global stats.
    
    # Optimization: If the flip doesn't change the max, we can accept faster.
    # But we need to know the new max.
    
    # Let's store the changes
    changes = []
    for ap_id, pos in affected_aps:
        old_s = int(ap_sums[ap_id])
        new_s = old_s + delta
        
        old_abs = abs(old_s)
        new_abs = abs(new_s)
        
        changes.append((ap_id, delta, old_abs, new_abs))
        
        if new_abs > new_max:
            new_max = new_abs
    
    # Now update count_max
    # If new_max > old_max:
    #   count_max = number of affected APs with new_abs == new_max
    #   (Unaffected APs have abs <= old_max < new_max, so they don't contribute)
    if new_max > old_max:
        new_count_max = 0
        for ap_id, delta, old_abs, new_abs in changes:
            if new_abs == new_max:
                new_count_max += 1
    else:
        # new_max <= old_max
        # If new_max < old_max:
        #   We need to check if any UNaffected AP has abs == old_max.
        #   If count_max was > number of affected APs with old_abs == old_max, then new_max remains old_max.
        #   This requires knowing how many affected APs had old_abs == old_max.
        
        affected_old_max_count = 0
        for ap_id, delta, old_abs, new_abs in changes:
            if old_abs == old_max:
                affected_old_max_count += 1
        
        if new_max < old_max:
            # If there are other APs (unaffected) with abs == old_max, the max doesn't drop.
            # We don't track the count of "second max" or similar.
            # So we must conservatively say new_max = old_max if count_max > affected_old_max_count.
            if count_max > affected_old_max_count:
                new_max = old_max
                # Count for new_max (which is old_max) includes unaffected ones + affected ones that still have old_max?
                # No, if new_max is forced to old_max, we need the count of APs with abs == old_max.
                # Unaffected: count_max - affected_old_max_count
                # Affected: those with new_abs == old_max
                new_count_max = (count_max - affected_old_max_count)
                for ap_id, delta, old_abs, new_abs in changes:
                    if new_abs == old_max:
                        new_count_max += 1
            else:
                # All maxes were in affected APs, and none of them remain at old_max (since new_max < old_max implies no affected AP has new_abs == old_max? No, new_max is the max of new_abs. If new_max < old_max, then no affected AP has new_abs == old_max).
                # So new_count_max is 0? No, new_max is the new global max.
                # We already found new_max = max(new_abs).
                # We need count of new_abs == new_max.
                new_count_max = 0
                for ap_id, delta, old_abs, new_abs in changes:
                    if new_abs == new_max:
                        new_count_max += 1
        else: # new_max == old_max
            # Max didn't change.
            # Count changes by: - (affected APs with old_abs==old_max and new_abs!=old_max) + (affected APs with old_abs!=old_max and new_abs==old_max)
            # Simpler: new_count_max = count_max - affected_old_max_count + (affected APs with new_abs == old_max)
            new_count_max = count_max - affected_old_max_count
            for ap_id, delta, old_abs, new_abs in changes:
                if new_abs == old_max:
                    new_count_max += 1
    
    return new_max, new_count_max, changes

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55
    
    rng = random.Random(42)
    
    # Pre-compute AP structure
    ap_indices, index_to_aps = _build_ap_structure(n)
    num_aps = len(ap_indices)
    
    # 1. Legendre Initialization
    p = _next_prime(n)
    signs = _generate_legendre_sequence(n, p)
    
    # 2. Compute initial sums and stats
    ap_sums = _init_prefix_sums(signs, ap_indices)
    current_max, count_max = _get_max_disc(ap_sums)
    
    best_signs = list(signs)
    best_max = current_max
    best_count_max = count_max
    
    # 3. Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    no_improve = 0
    max_no_improve = 10000
    
    while time.time() < deadline - 2 and T > T_min and no_improve < max_no_improve:
        i = rng.randrange(n)
        old_val = signs[i]
        new_val = -old_val
        
        new_max, new_count_max, changes = _eval_flip(signs, ap_sums, index_to_aps, i, old_val, new_val, current_max, count_max)
        
        # Acceptance Criteria
        accept = False
        if new_max < current_max:
            accept = True
        elif new_max == current_max:
            # Prefer moves that increase the count of maxes? No, usually we want to reduce the "pressure".
            # Actually, if max is same, it's neutral. Accept with prob 0.5 or 1.0 to allow exploration.
            accept = rng.random() < 0.5
        else:
            # Worse
            delta = new_max - current_max
            accept = rng.random() < np.exp(-delta / max(T, 1e-10))
        
        if accept:
            # Apply changes
            for ap_id, delta, old_abs, new_abs in changes:
                ap_sums[ap_id] += delta
            signs[i] = new_val
            current_max = new_max
            count_max = new_count_max
            
            if current_max < best_max:
                best_max = current_max
                best_signs = list(signs)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        T *= cooling
    
    return best_signs

if __name__ == "__main__":
    import time
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s, Result length: {len(result)}")
    
    # Verify
    ap_indices, _ = _build_ap_structure(256)
    max_disc = 0
    for indices in ap_indices:
        s = sum(result[i] for i in indices)
        if abs(s) > max_disc:
            max_disc = abs(s)
    print(f"Max Discrepancy: {max_disc}")
