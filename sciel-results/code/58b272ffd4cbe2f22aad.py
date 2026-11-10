import numpy as np
import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    # Precompute all APs and their index lists
    ap_lists = []
    for step in range(1, n):
        for start in range(step):
            seq = list(range(start, n, step))
            if len(seq) >= 2:
                ap_lists.append(seq)
    
    m = len(ap_lists)
    
    # Build incidence: for each index, which APs contain it
    index_to_aps = [[] for _ in range(n)]
    for ap_id, seq in enumerate(ap_lists):
        for idx in seq:
            index_to_aps[idx].append(ap_id)
    
    # Convert to numpy arrays for speed
    ap_sums = np.zeros(m, dtype=np.int32)
    
    # Precompute for each index the list of AP ids it belongs to as numpy array
    index_to_aps_np = [np.array(aps, dtype=np.int32) for aps in index_to_aps]
    
    # --- Initialization: Thue-Morse based + random restarts ---
    def thue_morse(n):
        """Generate Thue-Morse sequence (0/1), map to -1/+1."""
        signs = []
        for i in range(n):
            # Thue-Morse: parity of number of 1s in binary representation
            t = bin(i).count('1') % 2
            signs.append(1 if t == 0 else -1)
        return np.array(signs, dtype=np.int8)
    
    def legendre_mod_p(n, p):
        """Legendre symbol sequence mod p."""
        signs = np.zeros(n, dtype=np.int8)
        for i in range(n):
            val = (i + 1) % p
            if val == 0:
                signs[i] = 1
            else:
                ls = pow(val, (p - 1) // 2, p)
                signs[i] = 1 if ls == 1 else -1
        return signs
    
    def compute_ap_sums(signs):
        """Compute all AP sums from scratch."""
        sums = np.zeros(m, dtype=np.int32)
        for ap_id, seq in enumerate(ap_lists):
            s = 0
            for idx in seq:
                s += signs[idx]
            sums[ap_id] = s
        return sums
    
    def get_max_abs(sums):
        return int(np.max(np.abs(sums)))
    
    # Try several initializations
    rng = random.Random(12345)
    initializations = [
        thue_morse(n),
        legendre_mod_p(n, 257),
        legendre_mod_p(n, 127),
        legendre_mod_p(n, 61),
        np.array([rng.choice((-1, 1)) for _ in range(n)], dtype=np.int8),
        np.array([rng.choice((-1, 1)) for _ in range(n)], dtype=np.int8),
    ]
    
    best_signs = None
    best_max = float('inf')
    
    for init_signs in initializations:
        sums = compute_ap_sums(init_signs)
        mx = get_max_abs(sums)
        if mx < best_max:
            best_max = mx
            best_signs = init_signs.copy()
    
    # --- Local Search with Targeted Worst-AP Focus ---
    signs = best_signs.copy()
    ap_sums = compute_ap_sums(signs)
    current_max = best_max
    best_signs = signs.copy()
    
    def flip_index(i):
        """Flip sign at index i and update ap_sums. Returns old sign."""
        old_sign = signs[i]
        new_sign = -old_sign
        signs[i] = new_sign
        delta = 2 * new_sign
        for ap_id in index_to_aps_np[i]:
            ap_sums[ap_id] += delta
        return old_sign
    
    def revert_flip(i, old_sign):
        """Revert a flip at index i."""
        new_sign = -old_sign
        signs[i] = old_sign
        delta = 2 * new_sign
        for ap_id in index_to_aps_np[i]:
            ap_sums[ap_id] -= delta
    
    # Simulated annealing with targeted search
    temperature = 4.0
    cooling = 0.95
    no_improve = 0
    max_no_improve = 20
    best_overall_max = current_max
    best_overall_signs = signs.copy()
    
    while time.time() < deadline - 2:
        # Get worst APs
        abs_sums = np.abs(ap_sums)
        k = min(40, m)
        worst = np.argpartition(-abs_sums, k)[:k]
        
        # Collect indices that appear in worst APs
        candidate_indices = set()
        for ap_id in worst:
            for idx in ap_lists[ap_id]:
                candidate_indices.add(idx)
        
        candidates = list(candidate_indices)
        if not candidates:
            candidates = list(range(n))
        rng.shuffle(candidates)
        
        improved = False
        
        for i in candidates:
            if time.time() > deadline - 2:
                break
            
            old_sign = signs[i]
            new_sign = -old_sign
            delta = 2 * new_sign
            
            # Tentatively flip
            for ap_id in index_to_aps_np[i]:
                ap_sums[ap_id] += delta
            signs[i] = new_sign
            
            new_max = get_max_abs(ap_sums)
            
            if new_max < current_max:
                current_max = new_max
                improved = True
            elif new_max == current_max:
                if rng.random() < 0.2:
                    improved = True
                else:
                    revert_flip(i, old_sign)
            else:
                delta_max = new_max - current_max
                if rng.random() < np.exp(-delta_max / temperature):
                    current_max = new_max
                    improved = True
                else:
                    revert_flip(i, old_sign)
        
        if current_max < best_overall_max:
            best_overall_max = current_max
            best_overall_signs = signs.copy()
            no_improve = 0
        elif improved:
            no_improve = 0
        else:
            no_improve += 1
            temperature *= cooling
            
            if no_improve > max_no_improve:
                if best_overall_max > 3:
                    # Restart from best with perturbation
                    signs = best_overall_signs.copy()
                    num_perturb = rng.randint(6, 15)
                    for _ in range(num_perturb):
                        idx = rng.randint(0, n - 1)
                        signs[idx] = -signs[idx]
                    ap_sums = compute_ap_sums(signs)
                    current_max = get_max_abs(ap_sums)
                    temperature = 3.5
                    no_improve = 0
                else:
                    break
    
    return best_overall_signs.tolist()
