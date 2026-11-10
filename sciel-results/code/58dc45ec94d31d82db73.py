import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s buffer
    
    n = 128
    target_lo = 128
    target_hi = 256
    num_targets = target_hi - target_lo + 1  # 129
    
    # Precompute diagonal mask
    # diag_mask[i] is 1 if target t=128+i is even and t//2 is in [0,128]
    diag_mask = np.zeros(num_targets, dtype=np.int32)
    for i in range(num_targets):
        t = target_lo + i
        if t % 2 == 0:
            half_t = t // 2
            if 0 <= half_t <= n:
                diag_mask[i] = 1

    def get_counts(freq):
        conv = np.convolve(freq, freq, mode='full')
        # conv index t corresponds to sum t
        # We need sums from 128 to 256
        ordered = conv[target_lo:target_hi+1]
        # Calculate diagonal contributions
        # For each target t, if t is even, check if freq[t//2] is 1
        # We can vectorize this
        # Indices for t//2:
        # t = 128 + i
        # t//2 = 64 + i//2 (for even t)
        # Let's just use a loop or vectorized indexing
        diag_vals = np.zeros(num_targets, dtype=np.int32)
        # Vectorized way:
        # Create an array of indices for t//2 for all t
        t_vals = np.arange(target_lo, target_hi + 1)
        half_indices = t_vals // 2
        # Only valid if t is even
        mask = (t_vals % 2 == 0) & (half_indices <= n) & (half_indices >= 0)
        valid_indices = half_indices[mask]
        diag_vals[mask] = freq[valid_indices]
        
        counts = (ordered + diag_vals) // 2
        return counts

    def evaluate(A):
        freq = np.zeros(n + 1, dtype=np.int32)
        for x in A:
            freq[x] = 1
        counts = get_counts(freq)
        min_count = np.min(counts)
        max_count = np.max(counts)
        return min_count, max_count, counts

    def local_search(A_init, max_iters=50000):
        A = sorted(set(A_init))
        if not A:
            return A, float('inf')
        
        freq = np.zeros(n + 1, dtype=np.int32)
        for x in A:
            freq[x] = 1
            
        min_c, max_c, counts = evaluate(A)
        
        if min_c < 1:
             # Try to fix gaps greedily
             # This is a bit complex, so maybe just restart if initial is bad?
             # Or try adding elements that cover gaps.
             # For simplicity, if initial is bad, we might skip or try a few adds.
             # Let's try to add elements to cover gaps.
             for _ in range(100):
                 gaps = np.where(counts == 0)[0]
                 if len(gaps) == 0: break
                 # Pick a random gap
                 g = gaps[random.randint(0, len(gaps)-1)]
                 target_t = target_lo + g
                 # Find a, b such that a+b=target_t and (a,b) not in A
                 # Just pick random a in [0, n], b = target_t - a
                 # If b in [0, n] and (a,b) pair is new?
                 # Actually, just add a random element and see if it helps?
                 # This is inefficient. Better to start with valid sets.
                 pass
        
        cur_peak = max_c
        cur_min = min_c
        A_set = set(A)
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
                
            if cur_min < 1:
                # Invalid state, try to repair
                # Find a target with count 0
                zero_targets = np.where(counts == 0)[0]
                if len(zero_targets) == 0:
                    continue
                t_idx = zero_targets[random.randint(0, len(zero_targets)-1)]
                t_val = target_lo + t_idx
                # Try to add an element x such that x + y = t_val for some y in A
                # Or x + x = t_val
                found_fix = False
                for _ in range(10):
                    x = random.randint(0, n)
                    if x in A_set: continue
                    y = t_val - x
                    if 0 <= y <= n:
                        # Check if adding x helps
                        freq[x] = 1
                        new_counts = get_counts(freq)
                        new_min = np.min(new_counts)
                        if new_min > cur_min or (new_min == cur_min and np.sum(new_counts>0) > np.sum(counts>0)):
                            # Accept
                            A_set.add(x)
                            A.append(x)
                            counts = new_counts
                            cur_min = new_min
                            cur_peak = np.max(counts)
                            found_fix = True
                            break
                        else:
                            freq[x] = 0
                if not found_fix:
                    # Try removing a random element to change state?
                    # Or just break?
                    # Let's try a random swap
                    if A:
                        a_out = A[random.randint(0, len(A)-1)]
                        a_in = random.randint(0, n)
                        if a_in not in A_set:
                            freq[a_out] = 0
                            freq[a_in] = 1
                            new_counts = get_counts(freq)
                            new_min = np.min(new_counts)
                            if new_min > cur_min:
                                A_set.discard(a_out)
                                A_set.add(a_in)
                                A[A.index(a_out)] = a_in
                                A.sort()
                                counts = new_counts
                                cur_min = new_min
                                cur_peak = np.max(counts)
                            else:
                                freq[a_out] = 1
                                freq[a_in] = 0
                continue

            # Valid state (cur_min >= 1). Try to reduce cur_peak.
            # Find target with max count
            max_targets = np.where(counts == cur_peak)[0]
            if len(max_targets) == 0:
                break
            
            t_idx = max_targets[random.randint(0, len(max_targets)-1)]
            t_val = target_lo + t_idx
            
            # List pairs (a,b) in A with a+b = t_val
            # a <= b
            pairs = []
            for a in A:
                b = t_val - a
                if b < a: continue
                if b > n: break
                if b in A_set:
                    pairs.append((a, b))
            
            if not pairs:
                # Should not happen if count > 0
                continue
                
            # Try to remove one element from a random pair
            a, b = pairs[random.randint(0, len(pairs)-1)]
            
            # Option 1: Remove a
            if a != b: # If a==b, removing a removes the pair
                freq[a] = 0
                new_counts = get_counts(freq)
                new_min = np.min(new_counts)
                new_peak = np.max(new_counts)
                if new_min >= 1:
                    if new_peak < cur_peak:
                        A_set.discard(a)
                        A.remove(a)
                        counts = new_counts
                        cur_peak = new_peak
                        cur_min = new_min
                        continue
                    elif new_peak == cur_peak:
                        # Keep if it reduces set size?
                        # Or just accept to escape local minima?
                        # Let's accept if it doesn't increase peak
                        A_set.discard(a)
                        A.remove(a)
                        counts = new_counts
                        cur_min = new_min
                        # cur_peak unchanged
                        continue
                else:
                    freq[a] = 1
            
            # Option 2: Remove b (if different)
            if a != b:
                freq[b] = 0
                new_counts = get_counts(freq)
                new_min = np.min(new_counts)
                new_peak = np.max(new_counts)
                if new_min >= 1:
                    if new_peak < cur_peak:
                        A_set.discard(b)
                        A.remove(b)
                        counts = new_counts
                        cur_peak = new_peak
                        cur_min = new_min
                        continue
                    elif new_peak == cur_peak:
                        A_set.discard(b)
                        A.remove(b)
                        counts = new_counts
                        cur_min = new_min
                        continue
                else:
                    freq[b] = 1
            
            # Option 3: Swap a with x
            x = random.randint(0, n)
            if x not in A_set and x != a:
                freq[a] = 0
                freq[x] = 1
                new_counts = get_counts(freq)
                new_min = np.min(new_counts)
                new_peak = np.max(new_counts)
                if new_min >= 1:
                    if new_peak < cur_peak:
                        A_set.discard(a)
                        A_set.add(x)
                        A[A.index(a)] = x
                        A.sort()
                        counts = new_counts
                        cur_peak = new_peak
                        cur_min = new_min
                        continue
                    elif new_peak == cur_peak and random.random() < 0.1:
                        # Simulated annealing style acceptance
                        A_set.discard(a)
                        A_set.add(x)
                        A[A.index(a)] = x
                        A.sort()
                        counts = new_counts
                        cur_min = new_min
                        continue
                else:
                    freq[a] = 1
                    freq[x] = 0
            
            # Option 4: Swap b with x (if a!=b)
            if a != b:
                x = random.randint(0, n)
                if x not in A_set and x != b:
                    freq[b] = 0
                    freq[x] = 1
                    new_counts = get_counts(freq)
                    new_min = np.min(new_counts)
                    new_peak = np.max(new_counts)
                    if new_min >= 1:
                        if new_peak < cur_peak:
                            A_set.discard(b)
                            A_set.add(x)
                            A[A.index(b)] = x
                            A.sort()
                            counts = new_counts
                            cur_peak = new_peak
                            cur_min = new_min
                            continue
                        elif new_peak == cur_peak and random.random() < 0.1:
                            A_set.discard(b)
                            A_set.add(x)
                            A[A.index(b)] = x
                            A.sort()
                            counts = new_counts
                            cur_min = new_min
                            continue
                    else:
                        freq[b] = 1
                        freq[x] = 0
                        
        return A, cur_peak

    best_A = None
    best_peak = float('inf')
    
    random.seed(42)
    
    # Try various sizes
    for k in range(15, 45, 2):
        if time.time() > deadline - 5:
            break
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            A_rand = sorted(random.sample(range(n + 1), k))
            A_ls, peak_ls = local_search(A_rand)
            if peak_ls < best_peak:
                # Verify
                freq = np.zeros(n + 1, dtype=np.int32)
                for x in A_ls:
                    freq[x] = 1
                counts = get_counts(freq)
                if np.min(counts) >= 1:
                    best_A = A_ls
                    best_peak = peak_ls
                    # print(f"New best: k={len(A_ls)}, peak={best_peak}")
    
    # If we didn't find a good one, fallback to a dense set
    if best_A is None:
        best_A = list(range(64, 129))
        
    return list(best_A)
