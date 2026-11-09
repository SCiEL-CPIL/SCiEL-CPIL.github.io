import time
import random
import numpy as np
from math import log
from scipy.fft import fft, ifft

def _compute_score(U):
    """Compute score using FFT for fast sumset/difference set cardinality."""
    U = sorted(set(U))
    n = len(U)
    if n == 0 or U[0] != 0:
        return 0.0
    max_u = U[-1]
    if max_u == 0:
        return 1.0
    
    # Range of sums: 0 to 2*max_u
    # Range of diffs: -max_u to max_u
    sum_range = 2 * max_u + 1
    diff_range = 2 * max_u + 1
    
    # FFT size (next power of 2 for efficiency)
    def next_pow2(n):
        return 1 << (n - 1).bit_length()
    
    fft_size = next_pow2(max(sum_range, diff_range))
    
    # Indicator vector for U
    ind = np.zeros(fft_size, dtype=np.float64)
    for u in U:
        ind[u] = 1.0
    
    # Sumset: convolution of ind with ind
    # (ind * ind)(k) = number of pairs (a,b) in U×U with a+b=k
    conv_sum = ifft(fft(ind) * fft(ind)).real
    len_sum = int(np.sum(conv_sum > 0.5))
    
    # Difference set: convolve ind with reversed ind
    # (ind * rev_ind)(k) gives differences
    rev_ind = np.zeros(fft_size, dtype=np.float64)
    for u in U:
        rev_ind[fft_size - u] = 1.0
    # Actually, difference set: a - b for a,b in U
    # This is conv of ind with ind_reversed where ind_reversed[i] = ind[-i mod N]
    # Better: use the fact that diffset = {a-b : a,b in U}
    # conv(ind, ind_flipped) where ind_flipped[i] = ind[-i]
    
    # For differences: create indicator where position i represents value i
    # diff at position k means there exist a,b in U with a-b=k
    # This is cross-correlation, which is conv with reversed
    ind_rev = np.zeros(fft_size, dtype=np.float64)
    for u in U:
        ind_rev[fft_size - u] = 1.0  # This handles negative indices via wraparound
    
    conv_diff = ifft(fft(ind) * fft(ind_rev)).real
    # The result is shifted: conv_diff[k] corresponds to difference (k - fft_size + max_u) or similar
    # Actually let's be more careful.
    
    # Alternative: just compute directly for small sets, use FFT for larger
    if n <= 50:
        U_arr = np.array(U, dtype=np.int64)
        sums = U_arr[:, None] + U_arr[None, :]
        len_sum = len(np.unique(sums))
        diffs = U_arr[:, None] - U_arr[None, :]
        len_diff = len(np.unique(diffs))
    else:
        # Use FFT properly
        # Sumset: positions 0..2*max_u
        conv_sum = ifft(fft(ind) * fft(ind)).real
        len_sum = int(np.sum(conv_sum[:sum_range] > 0.5))
        
        # Diffset: positions -max_u..max_u
        # Create indicator for -U
        ind_neg = np.zeros(fft_size, dtype=np.float64)
        for u in U:
            pos = fft_size - u if u > 0 else 0
            ind_neg[pos] = 1.0
        
        conv_diff = ifft(fft(ind) * fft(ind_neg)).real
        # The convolution gives values at positions representing a + (-b) = a - b
        # Position k in result corresponds to value k (with wraparound handling)
        # We need to count positions where conv_diff > 0.5
        # The valid range is from -max_u to max_u, which maps to positions
        # (fft_size - max_u) to (fft_size + max_u) mod fft_size
        
        # Actually, let me just count all non-zero in the relevant range
        # Positions 0 to fft_size-1, valid diffs are -max_u to max_u
        # In the FFT output, diff d appears at position d mod fft_size
        # So for d in [-max_u, max_u], position is d if d>=0, else fft_size+d
        
        valid_positions = np.concatenate([
            np.arange(0, max_u + 1),
            np.arange(fft_size - max_u, fft_size)
        ])
        len_diff = int(np.sum(conv_diff[valid_positions] > 0.5))
    
    if len_sum == 0:
        return 0.0
        
    ratio = len_diff / len_sum
    if ratio <= 0:
        return 0.0
        
    denominator = log(2 * max_u + 1)
    if denominator == 0:
        return 1.0
        
    return 1 + log(ratio) / denominator

def _compute_score_fast(U_set):
    """Fast score computation for a set (already sorted, contains 0)."""
    n = len(U_set)
    if n < 2:
        return 1.0
    max_u = U_set[-1]
    if max_u == 0:
        return 1.0
    
    U_arr = np.array(U_set, dtype=np.int64)
    
    # For very small sets, direct computation is faster
    if n <= 30:
        sums = U_arr[:, None] + U_arr[None, :]
        len_sum = len(np.unique(sums))
        diffs = U_arr[:, None] - U_arr[None, :]
        len_diff = len(np.unique(diffs))
    else:
        # FFT-based
        max_u = U_set[-1]
        sum_range = 2 * max_u + 1
        
        def next_pow2(n):
            return 1 << (n - 1).bit_length()
        
        fft_size = next_pow2(sum_range)
        
        ind = np.zeros(fft_size, dtype=np.float64)
        for u in U_set:
            ind[u] = 1.0
        
        conv_sum = ifft(fft(ind) * fft(ind)).real
        len_sum = int(np.sum(conv_sum[:sum_range] > 0.5))
        
        # For diffs
        ind_neg = np.zeros(fft_size, dtype=np.float64)
        for u in U_set:
            if u == 0:
                ind_neg[0] = 1.0
            else:
                ind_neg[fft_size - u] = 1.0
        
        conv_diff = ifft(fft(ind) * fft(ind_neg)).real
        valid_pos = np.concatenate([np.arange(0, max_u + 1), np.arange(fft_size - max_u, fft_size)])
        len_diff = int(np.sum(conv_diff[valid_pos] > 0.5))
    
    if len_sum == 0:
        return 0.0
    ratio = len_diff / len_sum
    if ratio <= 0:
        return 0.0
    denominator = log(2 * max_u + 1)
    if denominator == 0:
        return 1.0
    return 1 + log(ratio) / denominator

def _greedy_construct(max_val, target_size, rng):
    """Greedy construction trying to maximize score."""
    U = [0]
    
    for _ in range(target_size - 1):
        best_elem = None
        best_score = _compute_score(U)
        
        # Try candidate elements
        candidates = set()
        # Random candidates
        for _ in range(20):
            candidates.add(rng.randint(1, max_val))
        # Near existing elements
        for u in U:
            for d in range(1, 5):
                if u + d <= max_val:
                    candidates.add(u + d)
                if u - d >= 1:
                    candidates.add(u - d)
        # Powers
        for base in [2, 3]:
            val = 1
            while val <= max_val:
                candidates.add(val)
                val *= base
        
        for c in candidates:
            if c in U:
                continue
            new_U = sorted(U + [c])
            s = _compute_score(new_U)
            if s > best_score:
                best_score = s
                best_elem = c
        
        if best_elem is None:
            # Pick random
            best_elem = rng.randint(1, max_val)
            while best_elem in U:
                best_elem = rng.randint(1, max_val)
        
        U.append(best_elem)
    
    return sorted(U)

def _local_search(U, max_val, timeout, rng):
    """Local search with multiple move types."""
    start_time = time.time()
    current = sorted(set(U))
    if 0 not in current:
        current.insert(0, 0)
    current_score = _compute_score(current)
    best = list(current)
    best_score = current_score
    
    # Track best per neighborhood
    no_improve = 0
    
    while time.time() - start_time < timeout:
        improved = False
        
        # Try all single-element modifications
        for i in range(len(current)):
            if current[i] == 0:
                continue
            
            # Try removing
            if len(current) > 2:
                new_U = current[:i] + current[i+1:]
                s = _compute_score(new_U)
                if s > current_score:
                    current = new_U
                    current_score = s
                    improved = True
                    no_improve = 0
                    break
            
            # Try shifting
            for delta in [-3, -2, -1, 1, 2, 3, 5, 10]:
                new_val = current[i] + delta
                if new_val < 1 or new_val > max_val:
                    continue
                if new_val in current:
                    continue
                new_U = current.copy()
                new_U[i] = new_val
                new_U = sorted(new_U)
                s = _compute_score(new_U)
                if s > current_score:
                    current = new_U
                    current_score = s
                    improved = True
                    no_improve = 0
                    break
            
            if improved:
                break
        
        if not improved:
            # Try adding
            if len(current) < 25:
                best_add = None
                best_add_score = current_score
                for _ in range(30):
                    c = rng.randint(1, max_val)
                    if c in current:
                        continue
                    new_U = sorted(current + [c])
                    s = _compute_score(new_U)
                    if s > best_add_score:
                        best_add_score = s
                        best_add = c
                if best_add is not None:
                    current = sorted(current + [best_add])
                    current_score = best_add_score
                    improved = True
                    no_improve = 0
        
        if improved:
            if current_score > best_score:
                best_score = current_score
                best = list(current)
        else:
            no_improve += 1
            if no_improve > 5:
                # Random restart
                current = sorted(set(current + [rng.randint(1, max_val)]))
                current_score = _compute_score(current)
                no_improve = 0
        
        if time.time() - start_time > timeout:
            break
    
    return best, best_score

def _sa_search(initial_set, max_val, timeout, rng):
    """Simulated Annealing with fast evaluations."""
    start_time = time.time()
    
    current = sorted(set(initial_set))
    if 0 not in current:
        current.insert(0, 0)
    current_score = _compute_score(current)
    
    best = list(current)
    best_score = current_score
    
    T = 0.3
    T_min = 1e-4
    cooling = 0.999
    
    step_count = 0
    
    while time.time() - start_time < timeout:
        if T < T_min:
            break
        
        # Generate neighbor
        move_type = rng.random()
        new_set = list(current)
        
        if move_type < 0.35:
            # Shift an element
            if len(new_set) <= 1:
                continue
            non_zero_indices = [i for i, x in enumerate(new_set) if x != 0]
            if not non_zero_indices:
                continue
            idx = non_zero_indices[rng.randint(0, len(non_zero_indices) - 1)]
            old_val = new_set[idx]
            
            # Larger range for shifts
            delta = rng.randint(-20, 20)
            new_val = old_val + delta
            
            if new_val < 1:
                new_val = rng.randint(1, max_val)
            elif new_val > max_val:
                new_val = rng.randint(1, max_val)
            
            if new_val in new_set and new_val != old_val:
                # Find nearby free value
                found = False
                for d in range(1, 10):
                    for v in [new_val + d, new_val - d]:
                        if 1 <= v <= max_val and v not in new_set:
                            new_val = v
                            found = True
                            break
                    if found:
                        break
                if not found:
                    continue
            
            new_set[idx] = new_val
        elif move_type < 0.55:
            # Add element
            if len(new_set) > 30:
                continue
            new_elem = rng.randint(1, max_val)
            if new_elem in new_set:
                continue
            new_set.append(new_elem)
        elif move_type < 0.75:
            # Remove element
            non_zero = [x for x in new_set if x != 0]
            if not non_zero:
                continue
            rem = non_zero[rng.randint(0, len(non_zero) - 1)]
            new_set.remove(rem)
        else:
            # Larger perturbation
            if rng.random() < 0.5:
                # Swap two elements
                if len(new_set) >= 3:
                    non_zero = [i for i, x in enumerate(new_set) if x != 0]
                    if len(non_zero) >= 2:
                        i, j = rng.sample(non_zero, 2)
                        new_set[i], new_set[j] = new_set[j], new_set[i]
                else:
                    continue
            else:
                # Scale
                factor = rng.choice([2, 3, 0.5, 0.33])
                new_set = [0] + [int(x * factor) for x in new_set if x != 0]
                new_set = sorted(set([x for x in new_set if 1 <= x <= max_val]))
                if 0 not in new_set:
                    new_set.insert(0, 0)
                if len(new_set) < 2:
                    new_set.append(1)
        
        new_set = sorted(set(new_set))
        if 0 not in new_set or len(new_set) < 2:
            continue
        
        new_score = _compute_score(new_set)
        step_count += 1
        
        # Acceptance
        if new_score >= current_score:
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = list(current)
        else:
            delta_s = current_score - new_score
            if delta_s < 0:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-delta_s / T)
            
            if rng.random() < accept_prob:
                current = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
        
        T *= cooling
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    rng = random.Random(42)
    
    # Start with good initial sets
    initial_sets = []
    
    # Known good structures
    initial_sets.append([0, 1, 3, 7, 15])  # 2^k - 1
    initial_sets.append([0, 1, 3, 9, 27])  # 3^k
    initial_sets.append([0, 1, 4, 9, 16, 25])  # squares
    initial_sets.append([0, 1, 3, 7, 13, 21])  # near quadratic
    initial_sets.append([0, 1, 2, 4, 8, 16])  # powers of 2
    
    # Greedy constructions
    for max_v in [50, 100, 200]:
        for size in [5, 8, 10]:
            g = _greedy_construct(max_v, size, rng)
            initial_sets.append(g)
    
    # Random sets
    for _ in range(5):
        size = rng.randint(4, 10)
        s = {0}
        while len(s) < size:
            s.add(rng.randint(1, 150))
        initial_sets.append(sorted(s))
    
    # Deduplicate and shuffle
    seen = set()
    unique_sets = []
    for s in initial_sets:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique_sets.append(s)
    
    rng.shuffle(unique_sets)
    
    best_result = [0, 1, 3, 7]
    best_score = _compute_score(best_result)
    
    # Phase 1: SA on each initial set
    for init in unique_sets:
        if time.time() > deadline - 10:
            break
        
        max_val = max(max(init) * 3, 150)
        remaining = deadline - time.time()
        if remaining < 3.0:
            break
        sa_timeout = min(8.0, remaining * 0.25)
        
        result, score = _sa_search(init, max_val, sa_timeout, rng)
        
        if score > best_score:
            best_score = score
            best_result = result
    
    # Phase 2: Local search refinement on best
    if time.time() < deadline - 5:
        max_val_refine = max(max(best_result) * 2, 200)
        remaining = deadline - time.time()
        result, score = _local_search(best_result, max_val_refine, min(15.0, remaining - 2), rng)
        if score > best_score:
            best_score = score
            best_result = result
    
    # Phase 3: Another SA round on the best
    if time.time() < deadline - 3:
        max_val_refine = max(max(best_result) * 2, 200)
        remaining = deadline - time.time()
        result, score = _sa_search(best_result, max_val_refine, min(12.0, remaining - 1), rng)
        if score > best_score:
            best_score = score
            best_result = result
    
    # Final check
    final_score = _compute_score(best_result)
    if final_score < 1.0:
        best_result = [0, 1, 3, 7]
    
    return best_result
