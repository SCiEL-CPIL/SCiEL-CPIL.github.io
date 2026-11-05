import time
import math
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 256

    # Build all APs with length >= 2
    aps = []
    ap_membership = [[] for _ in range(n)]
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                break
            sub_indices = list(range(r, n, d))
            if len(sub_indices) >= 2:
                ap_idx = len(aps)
                aps.append(sub_indices)
                for idx in sub_indices:
                    ap_membership[idx].append(ap_idx)

    def _update_ap(ap_idx, signs):
        """Compute max and min subarray sum for AP ap_idx."""
        ap = aps[ap_idx]
        vals = np.array([signs[i] for i in ap], dtype=np.int32)
        cum = np.cumsum(vals)
        cum_ext = np.concatenate(([0], cum))
        min_prefix = np.minimum.accumulate(cum_ext)
        max_prefix = np.maximum.accumulate(cum_ext)
        max_sub = int(np.max(cum_ext[1:] - min_prefix[:-1]))
        min_sub = int(np.min(cum_ext[1:] - max_prefix[:-1]))
        return max_sub, min_sub

    def _compute_all(signs):
        """Compute max/min subarray sums for all APs."""
        max_sums = np.zeros(len(aps), dtype=np.int32)
        min_sums = np.zeros(len(aps), dtype=np.int32)
        for i in range(len(aps)):
            max_sums[i], min_sums[i] = _update_ap(i, signs)
        return max_sums, min_sums

    def _get_disc(max_sums, min_sums):
        if len(max_sums) == 0:
            return 1
        return max(int(np.max(max_sums)), -int(np.min(min_sums)), 1)

    def _eval_full(signs):
        max_sums, min_sums = _compute_all(signs)
        return _get_disc(max_sums, min_sums)

    def _find_worst_ap_indices(max_sums, min_sums, k=5):
        """Find indices of the k worst APs by absolute discrepancy."""
        abs_disc = np.maximum(max_sums, -min_sums)
        if len(abs_disc) == 0:
            return []
        k = min(k, len(abs_disc))
        worst_idx = np.argsort(-abs_disc)[:k]
        return list(worst_idx)

    def _find_best_flip_in_ap(ap_idx, signs, max_sums, min_sums):
        """Find the sign flip within AP ap_idx that most reduces its discrepancy."""
        ap = aps[ap_idx]
        current_max = max_sums[ap_idx]
        current_min = min_sums[ap_idx]
        current_worst = max(current_max, -current_min)
        
        best_flip_pos = -1
        best_new_worst = current_worst + 1
        
        for pos in ap:
            old_val = signs[pos]
            signs[pos] = -old_val
            new_max, new_min = _update_ap(ap_idx, signs)
            new_worst = max(new_max, -new_min)
            signs[pos] = old_val
            
            if new_worst < best_new_worst:
                best_new_worst = new_worst
                best_flip_pos = pos
        
        return best_flip_pos, best_new_worst

    # --- Phase 1: Generate diverse candidates ---
    candidates = []

    # Thue-Morse
    tm = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    candidates.append(tm)

    # Alternating
    alt = [1 if i % 2 == 0 else -1 for i in range(n)]
    candidates.append(alt)

    # Block patterns with different sizes
    for block_size in [2, 4, 8, 16]:
        direct = [0] * n
        for i in range(n):
            block = (i // block_size) % 2
            pos = i % block_size
            if block == 0:
                direct[i] = 1 if pos < block_size // 2 else -1
            else:
                direct[i] = -1 if pos < block_size // 2 else 1
        candidates.append(direct)

    # Multiple random
    for s in range(40):
        r = random.Random(s)
        candidates.append([r.choice((-1, 1)) for _ in range(n)])

    # Random with bias toward balance
    for s in range(20):
        r = random.Random(100 + s)
        signs = [r.choice((-1, 1)) for _ in range(n)]
        while sum(signs) > 0:
            idx = r.randint(0, n - 1)
            if signs[idx] == 1:
                signs[idx] = -1
                break
        candidates.append(signs)

    # --- Phase 2: Evaluate candidates, pick best ---
    best_signs = candidates[0][:]
    best_disc = float('inf')

    for cand in candidates:
        if time.time() > deadline - 30:
            break
        d = _eval_full(cand)
        if d < best_disc:
            best_disc = d
            best_signs = cand[:]

    # --- Phase 2.5: Try MILP for a few iterations ---
    if time.time() < deadline - 25:
        try:
            # Use MILP to minimize max AP discrepancy
            # Variables: x[0..n-1] in {-1, 1}, t >= 0
            # Minimize t
            # Subject to: for each AP, for each subarray, sum(signs in subarray) <= t
            #            for each AP, for each subarray, -sum(signs in subarray) <= t
            
            # This is too many constraints. Instead, use a simpler formulation:
            # For each AP, bound the max and min cumulative sums
            
            # Let's use a different approach: iterative constraint addition
            # Start with a good candidate, then add constraints for worst APs
            
            milp_signs = None
            milp_disc = float('inf')
            
            # Try MILP with a limited set of constraints (top worst APs)
            for milp_iter in range(3):
                if time.time() > deadline - 15:
                    break
                
                # Get current worst APs
                if milp_signs is None:
                    current_signs = best_signs
                else:
                    current_signs = milp_signs
                
                max_sums, min_sums = _compute_all(current_signs)
                current_disc = _get_disc(max_sums, min_sums)
                
                if current_disc <= 2:
                    break
                
                # Select top K worst APs to constrain
                K = 50
                abs_disc = np.maximum(max_sums, -min_sums)
                worst_idx = np.argsort(-abs_disc)[:K]
                
                # Build MILP: minimize t
                # x_i in {-1, 1} for i in 0..n-1
                # t >= 0
                # For each selected AP, for each subarray: sum <= t and -sum <= t
                
                # Number of variables: n + 1
                c = np.zeros(n + 1)
                c[n] = 1.0  # minimize t
                
                # Bounds: x_i in [-1, 1], t in [0, inf)
                lb = np.full(n + 1, -1.0)
                ub = np.full(n + 1, 1.0)
                ub[n] = float('inf')
                
                # Integrality: x_i must be integer
                integrality = np.ones(n + 1)
                integrality[n] = 0  # t is continuous
                
                # Constraints: for each selected AP and subarray
                # We'll build the constraint matrix
                # Each row: coefficients for x_0..x_{n-1}, t
                # For sum(subarray) <= t: sum(x_i for i in subarray) - t <= 0
                # For -sum(subarray) <= t: -sum(x_i for i in subarray) - t <= 0
                
                A_rows = []
                b_ub = []
                
                for ap_idx in worst_idx:
                    ap = aps[ap_idx]
                    # For each subarray of this AP
                    m = len(ap)
                    for start in range(m):
                        for end in range(start + 1, m + 1):
                            sub = ap[start:end]
                            # sum(x_i for i in sub) - t <= 0
                            row = np.zeros(n + 1)
                            for pos in sub:
                                row[pos] = 1.0
                            row[n] = -1.0
                            A_rows.append(row)
                            b_ub.append(0.0)
                            # -sum(x_i for i in sub) - t <= 0
                            row2 = np.zeros(n + 1)
                            for pos in sub:
                                row2[pos] = -1.0
                            row2[n] = -1.0
                            A_rows.append(row2)
                            b_ub.append(0.0)
                
                if len(A_rows) == 0:
                    break
                
                A = np.array(A_rows)
                b_ub_arr = np.array(b_ub)
                
                # Also add constraint: sum(x_i) = 0 (balance)
                # This helps but may be too restrictive. Let's add it as a soft constraint.
                # Actually, let's not add it to keep the problem simpler.
                
                constraints = LinearConstraint(A, -np.inf, b_ub_arr)
                bounds = Bounds(lb, ub)
                
                result = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
                
                if result.status == 0:
                    x = result.x[:n]
                    # Convert to signs
                    milp_signs = [1 if xi >= 0 else -1 for xi in x]
                    milp_disc = _eval_full(milp_signs)
                    
                    if milp_disc < best_disc:
                        best_disc = milp_disc
                        best_signs = milp_signs[:]
                
                if time.time() > deadline - 15:
                    break
            
            # If MILP found something good, use it as starting point
            if milp_signs is not None and milp_disc < best_disc:
                best_signs = milp_signs[:]
                best_disc = milp_disc
                
        except Exception:
            pass  # Fall back to heuristic if MILP fails

    # --- Phase 3: Targeted + SA hybrid ---
    signs = best_signs[:]
    max_sums, min_sums = _compute_all(signs)
    current_disc = _get_disc(max_sums, min_sums)

    rng = random.Random(12345)
    no_improve = 0
    max_no_improve = 500
    temp = 3.0
    cool = 0.996
    targeted_count = 0
    random_count = 0

    while time.time() < deadline - 8 and no_improve < max_no_improve:
        use_targeted = (targeted_count % 3 == 0) or (current_disc > 4)
        
        if use_targeted and len(aps) > 0:
            worst_aps = _find_worst_ap_indices(max_sums, min_sums, k=3)
            flip_made = False
            
            for ap_idx in worst_aps:
                flip_pos, new_worst = _find_best_flip_in_ap(ap_idx, signs, max_sums, min_sums)
                
                if flip_pos >= 0:
                    old_val = signs[flip_pos]
                    signs[flip_pos] = -old_val
                    
                    for a_idx in ap_membership[flip_pos]:
                        max_sums[a_idx], min_sums[a_idx] = _update_ap(a_idx, signs)
                    
                    new_disc = _get_disc(max_sums, min_sums)
                    
                    if new_disc < current_disc:
                        current_disc = new_disc
                        no_improve = 0
                        targeted_count += 1
                        if current_disc < best_disc:
                            best_disc = current_disc
                            best_signs = signs[:]
                    else:
                        signs[flip_pos] = old_val
                        for a_idx in ap_membership[flip_pos]:
                            max_sums[a_idx], min_sums[a_idx] = _update_ap(a_idx, signs)
                        no_improve += 1
                    
                    flip_made = True
                    break
            
            if not flip_made:
                idx = rng.randint(0, n - 1)
                old_val = signs[idx]
                signs[idx] = -old_val
                
                for ap_idx in ap_membership[idx]:
                    max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)
                
                new_disc = _get_disc(max_sums, min_sums)
                
                if new_disc < current_disc:
                    current_disc = new_disc
                    no_improve = 0
                    random_count += 1
                    if current_disc < best_disc:
                        best_disc = current_disc
                        best_signs = signs[:]
                else:
                    delta = new_disc - current_disc
                    if delta > 0 and temp > 0.1:
                        if rng.random() < math.exp(-delta / temp):
                            current_disc = new_disc
                            no_improve = 1
                        else:
                            signs[idx] = old_val
                            for ap_idx in ap_membership[idx]:
                                max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)
                            no_improve += 1
                    else:
                        signs[idx] = old_val
                        for ap_idx in ap_membership[idx]:
                            max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)
                        no_improve += 1
        else:
            idx = rng.randint(0, n - 1)
            old_val = signs[idx]
            signs[idx] = -old_val
            
            for ap_idx in ap_membership[idx]:
                max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)
            
            new_disc = _get_disc(max_sums, min_sums)
            
            if new_disc < current_disc:
                current_disc = new_disc
                no_improve = 0
                random_count += 1
                if current_disc < best_disc:
                    best_disc = current_disc
                    best_signs = signs[:]
            else:
                delta = new_disc - current_disc
                if delta > 0 and temp > 0.1:
                    if rng.random() < math.exp(-delta / temp):
                        current_disc = new_disc
                        no_improve = 1
                    else:
                        signs[idx] = old_val
                        for ap_idx in ap_membership[idx]:
                            max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)
                        no_improve += 1
                else:
                    signs[idx] = old_val
                    for ap_idx in ap_membership[idx]:
                        max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)
                    no_improve += 1

        temp *= cool

    # --- Phase 4: Strict targeted hill climbing ---
    if time.time() < deadline - 5:
        signs = best_signs[:]
        max_sums, min_sums = _compute_all(signs)
        current_disc = _get_disc(max_sums, min_sums)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            worst_aps = _find_worst_ap_indices(max_sums, min_sums, k=5)
            for ap_idx in worst_aps:
                flip_pos, new_worst = _find_best_flip_in_ap(ap_idx, signs, max_sums, min_sums)
                if flip_pos >= 0:
                    old_val = signs[flip_pos]
                    signs[flip_pos] = -old_val
                    for a_idx in ap_membership[flip_pos]:
                        max_sums[a_idx], min_sums[a_idx] = _update_ap(a_idx, signs)
                    new_disc = _get_disc(max_sums, min_sums)
                    if new_disc < current_disc:
                        current_disc = new_disc
                        improved = True
                        if current_disc < best_disc:
                            best_disc = current_disc
                            best_signs = signs[:]
                    else:
                        signs[flip_pos] = old_val
                        for a_idx in ap_membership[flip_pos]:
                            max_sums[a_idx], min_sums[a_idx] = _update_ap(a_idx, signs)
            
            for idx in range(n):
                if not improved and time.time() > deadline - 5:
                    break
                old_val = signs[idx]
                signs[idx] = -old_val
                for ap_idx in ap_membership[idx]:
                    max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)
                new_disc = _get_disc(max_sums, min_sums)
                if new_disc < current_disc:
                    current_disc = new_disc
                    improved = True
                    if current_disc < best_disc:
                        best_disc = current_disc
                        best_signs = signs[:]
                else:
                    signs[idx] = old_val
                    for ap_idx in ap_membership[idx]:
                        max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)

    # --- Phase 5: Additional restarts ---
    if time.time() < deadline - 10:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            signs = best_signs[:]
            rng2 = random.Random(9999 + restart)
            for _ in range(10):
                idx = rng2.randint(0, n - 1)
                signs[idx] = -signs[idx]

            max_sums, min_sums = _compute_all(signs)
            current_disc = _get_disc(max_sums, min_sums)

            temp = 2.0
            cool = 0.995
            no_improve = 0
            max_no_improve = 200
            t_count = 0

            while time.time() < deadline - 5 and no_improve < max_no_improve:
                use_targeted = (t_count % 3 == 0) or (current_disc > 4)
                
                if use_targeted and len(aps) > 0:
                    worst_aps = _find_worst_ap_indices(max_sums, min_sums, k=3)
                    flip_made = False
                    for ap_idx in worst_aps:
                        flip_pos, new_worst = _find_best_flip_in_ap(ap_idx, signs, max_sums, min_sums)
                        if flip_pos >= 0:
                            old_val = signs[flip_pos]
                            signs[flip_pos] = -old_val
                            for a_idx in ap_membership[flip_pos]:
                                max_sums[a_idx], min_sums[a_idx] = _update_ap(a_idx, signs)
                            new_disc = _get_disc(max_sums, min_sums)
                            if new_disc < current_disc:
                                current_disc = new_disc
                                no_improve = 0
                                t_count += 1
                                if current_disc < best_disc:
                                    best_disc = current_disc
                                    best_signs = signs[:]
                            else:
                                signs[flip_pos] = old_val
                                for a_idx in ap_membership[flip_pos]:
                                    max_sums[a_idx], min_sums[a_idx] = _update_ap(a_idx, signs)
                                no_improve += 1
                            flip_made = True
                            break
                    if not flip_made:
                        idx = rng2.randint(0, n - 1)
                        old_val = signs[idx]
                        signs[idx] = -old_val
                        for ap_idx in ap_membership[idx]:
                            max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)
                        new_disc = _get_disc(max_sums, min_sums)
                        if new_disc < current_disc:
                            current_disc = new_disc
                            no_improve = 0
                            if current_disc < best_disc:
                                best_disc = current_disc
                                best_signs = signs[:]
                        else:
                            delta = new_disc - current_disc
                            if delta > 0 and temp > 0.1:
                                if rng2.random() < math.exp(-delta / temp):
                                    current_disc = new_disc
                                    no_improve = 1
                                else:
                                    signs[idx] = old_val
                                    for ap_idx in ap_membership[idx]:
                                        max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)
                                    no_improve += 1
                            else:
                                signs[idx] = old_val
                                for ap_idx in ap_membership[idx]:
                                    max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)
                                no_improve += 1
                else:
                    idx = rng2.randint(0, n - 1)
                    old_val = signs[idx]
                    signs[idx] = -old_val
                    for ap_idx in ap_membership[idx]:
                        max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)
                    new_disc = _get_disc(max_sums, min_sums)
                    if new_disc < current_disc:
                        current_disc = new_disc
                        no_improve = 0
                        if current_disc < best_disc:
                            best_disc = current_disc
                            best_signs = signs[:]
                    else:
                        delta = new_disc - current_disc
                        if delta > 0 and temp > 0.1:
                            if rng2.random() < math.exp(-delta / temp):
                                current_disc = new_disc
                                no_improve = 1
                            else:
                                signs[idx] = old_val
                                for ap_idx in ap_membership[idx]:
                                    max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)
                                no_improve += 1
                        else:
                            signs[idx] = old_val
                            for ap_idx in ap_membership[idx]:
                                max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)
                            no_improve += 1
                temp *= cool

    # --- Phase 6: Final polish ---
    if time.time() < deadline - 2:
        signs = best_signs[:]
        max_sums, min_sums = _compute_all(signs)
        current_disc = _get_disc(max_sums, min_sums)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            worst_aps = _find_worst_ap_indices(max_sums, min_sums, k=3)
            for ap_idx in worst_aps:
                flip_pos, new_worst = _find_best_flip_in_ap(ap_idx, signs, max_sums, min_sums)
                if flip_pos >= 0:
                    old_val = signs[flip_pos]
                    signs[flip_pos] = -old_val
                    for a_idx in ap_membership[flip_pos]:
                        max_sums[a_idx], min_sums[a_idx] = _update_ap(a_idx, signs)
                    new_disc = _get_disc(max_sums, min_sums)
                    if new_disc < current_disc:
                        current_disc = new_disc
                        improved = True
                        if current_disc < best_disc:
                            best_disc = current_disc
                            best_signs = signs[:]
                    else:
                        signs[flip_pos] = old_val
                        for a_idx in ap_membership[flip_pos]:
                            max_sums[a_idx], min_sums[a_idx] = _update_ap(a_idx, signs)
            for idx in range(n):
                if not improved and time.time() > deadline - 2:
                    break
                old_val = signs[idx]
                signs[idx] = -old_val
                for ap_idx in ap_membership[idx]:
                    max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)
                new_disc = _get_disc(max_sums, min_sums)
                if new_disc < current_disc:
                    current_disc = new_disc
                    improved = True
                    if current_disc < best_disc:
                        best_disc = current_disc
                        best_signs = signs[:]
                else:
                    signs[idx] = old_val
                    for ap_idx in ap_membership[idx]:
                        max_sums[ap_idx], min_sums[ap_idx] = _update_ap(ap_idx, signs)

    return best_signs
