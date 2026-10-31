import time
import random
import math
import numpy as np

_START = time.time()
_DEADLINE = _START + 145.0  # Safety margin vs 150 s kill

def _score_from_set(u):
    """
    Compute the score for a set of integers u using FFT for efficiency.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    """
    if not u or len(u) < 2:
        return 0.0
    m = max(u)
    if m < 1:
        return 0.0
    
    u_arr = np.array(u, dtype=np.int64)
    n = len(u_arr)
    
    # FFT size needs to be at least 2 * max_val + 1 to avoid circular convolution artifacts
    # For sums: range is [0, 2m]
    # For diffs: range is [-m, m], shifted to [0, 2m]
    size = 1
    while size < 2 * m + 1:
        size *= 2
        
    # Create indicator vector
    f = np.zeros(size, dtype=np.float64)
    f[u_arr] = 1.0
    
    # FFT
    F = np.fft.rfft(f)
    
    # Sumset: U + U
    # Convolution of f with f
    sum_fft = F * F
    sum_counts = np.fft.irfft(sum_fft, n=size)
    # The sum counts are at indices 0 to 2m
    # Unique sums are indices where count > 0
    # Since f is indicator, counts are integers (mostly). 
    # Due to floating point, we check > 0.5
    unique_sums = np.sum(sum_counts[:2*m+1] > 0.5)
    
    # Difference set: U - U
    # Convolution of f with reverse(f)
    # Let g[i] = f[-i mod size]. Then (f * g)[k] = sum f[j] g[k-j] = sum f[j] f[j-k]
    # This gives counts of differences j - (j-k) = k? 
    # Actually, standard difference set calculation via FFT:
    # If we compute f * reverse(f), the peak is at the center.
    # Let's use the property that |U-U| can be found from autocorrelation.
    # Autocorrelation R[k] = sum f[i] f[i+k]. This is equivalent to counts of differences.
    # Wait, the number of distinct differences is the support of the autocorrelation function.
    
    # Let's compute autocorrelation properly.
    # R = ifft(fft(f) * conj(fft(f)))
    # R[k] = sum_i f[i+k] f[i]
    # This counts pairs (i, i+k) in U, i.e., difference k.
    # The indices k range from -m to m. In the FFT output, negative indices wrap around.
    auto_fft = F * np.conj(F)
    auto_counts = np.fft.irfft(auto_fft, n=size)
    
    # The distinct differences correspond to indices where auto_counts > 0.5
    # The range of differences is [-m, m].
    # In the circular convolution of size 'size', the positive differences 1..m are at indices 1..m.
    # The negative differences -1..-m are at indices size-1..size-m.
    # The zero difference is at index 0.
    
    # We need to count the number of distinct differences.
    # Indices 0 to m cover diffs 0, 1, ..., m
    # Indices size-m to size-1 cover diffs -m, ..., -1
    
    # Let's extract the relevant parts
    # Part 1: 0 to m
    pos_diffs = auto_counts[:m+1]
    # Part 2: size-m to size-1
    neg_diffs = auto_counts[size-m:size]
    
    # Combine and count non-zeros
    # Note: auto_counts[0] is always n (pairs i,i).
    # We just need the count of indices in [0, m] U [size-m, size-1] where value > 0.5
    distinct_diffs = np.sum(pos_diffs > 0.5) + np.sum(neg_diffs > 0.5)
    
    if unique_sums == 0 or distinct_diffs == 0:
        return 0.0
        
    ratio = distinct_diffs / float(unique_sums)
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * m + 1)

def _construct_greedy_golomb(n):
    """
    Construct a Golomb ruler-like set of size n.
    Greedy approach: start with 0, then add the smallest integer that doesn't create duplicate differences.
    This maximizes |U-U| for a given max(U), which helps the ratio.
    """
    if n <= 0:
        return []
    if n == 1:
        return [0]
    
    u = [0]
    diffs = set([0])
    
    for _ in range(1, n):
        candidate = u[-1] + 1
        while True:
            # Check if adding 'candidate' creates duplicate differences
            valid = True
            new_diffs_to_check = set()
            for x in u:
                d = abs(candidate - x)
                new_diffs_to_check.add(d)
            
            # If any of these new diffs already exist in 'diffs' (excluding 0? No, diffs includes 0)
            # Wait, Golomb ruler definition: all pairwise differences are distinct.
            # Since we are adding a new point, we only need to check differences between the new point and existing points.
            # And we need to ensure these new differences are not in the existing set of differences.
            # Note: The difference 0 is only for x=x, which is not a pair of distinct elements in the set difference context usually, 
            # but U-U includes 0. However, for Golomb property, we care about distinct non-zero differences.
            # Actually, if we want to maximize |U-U|, we want ALL differences to be unique.
            # The set of differences of a Golomb ruler has size n(n-1)/2 + 1 (including 0).
            
            if any(d in diffs for d in new_diffs_to_check):
                candidate += 1
                valid = False
                continue
            else:
                # It's valid
                u.append(candidate)
                diffs.update(new_diffs_to_check)
                break
                
    return u

def _local_search(u, max_steps=10000, timeout=10.0):
    """
    Perform local search on the set u.
    Moves:
    1. Replace an element with a nearby value.
    2. Add a new element.
    3. Remove an element.
    """
    u = sorted(set(u))
    if len(u) < 2:
        u = [0, 1]
        
    current_score = _score_from_set(u)
    best_u = u[:]
    best_score = current_score
    
    start_time = time.time()
    
    for step in range(max_steps):
        if time.time() > _DEADLINE - 5.0 or time.time() > start_time + timeout:
            break
            
        move_type = random.choice(['replace', 'replace', 'add', 'remove'])
        new_u = u[:]
        
        if move_type == 'replace':
            if len(u) < 2:
                continue
            idx = random.randint(1, len(u) - 1)  # Don't replace 0
            old_val = u[idx]
            # Try a few nearby values
            # Use a wider search range for structure exploration
            search_range = max(1, int(math.sqrt(old_val)))
            for _ in range(5):
                delta = random.randint(-search_range, search_range)
                new_val = old_val + delta
                if new_val < 0:
                    new_val = 0
                if new_val in u:
                    continue
                test_u = u[:]
                test_u[idx] = new_val
                test_u = sorted(set(test_u))
                if len(test_u) != len(u):
                    continue
                s = _score_from_set(test_u)
                if s > current_score:
                    u = test_u
                    current_score = s
                    if current_score > best_score:
                        best_score = current_score
                        best_u = u[:]
                    break
        elif move_type == 'add':
            if len(u) > 100:  # Cap size for performance, FFT handles large ranges well but n^2 memory in score is avoided
                continue
            max_val = max(u)
            # Try adding a value in a random gap or extending the range
            if random.random() < 0.7:
                # Extend range significantly to potentially lower log denominator impact? 
                # No, log(2m+1) grows slowly. We want to increase |U-U|/|U+U|.
                # Extending range adds large unique differences.
                new_val = max_val + random.randint(1, max(1, max_val // 5))
            else:
                # Fill a gap
                if len(u) < 2:
                    continue
                idx = random.randint(0, len(u)-2)
                low = u[idx]
                high = u[idx+1]
                if high - low > 1:
                    new_val = random.randint(low + 1, high - 1)
                else:
                    continue
                    
            if new_val in u or new_val < 0:
                continue
            test_u = sorted(u + [new_val])
            s = _score_from_set(test_u)
            if s > current_score:
                u = test_u
                current_score = s
                if current_score > best_score:
                    best_score = current_score
                    best_u = u[:]
        elif move_type == 'remove':
            if len(u) < 3:
                continue
            idx = random.randint(1, len(u) - 1)  # Don't remove 0
            test_u = u[:]
            test_u.pop(idx)
            s = _score_from_set(test_u)
            if s > current_score:
                u = test_u
                current_score = s
                if current_score > best_score:
                    best_score = current_score
                    best_u = u[:]
                    
    return best_u, best_score

def construct():
    global _DEADLINE
    
    random.seed(42)
    np.random.seed(42)
    
    best_u = [0, 1]
    best_s = 0.0
    
    # Phase 1: Generate diverse initial candidates
    # 1. Greedy Golomb Rulers (Structurally optimal for diff/sum ratio)
    # 2. Geometric Progressions (Spread out)
    # 3. Random sets
    
    candidates = []
    
    # Golomb Rulers of various sizes
    for n in [10, 20, 30, 40, 50, 60, 80, 100]:
        if time.time() > _DEADLINE - 30.0:
            break
        try:
            u = _construct_greedy_golomb(n)
            if u:
                candidates.append(u)
        except Exception:
            pass
            
    # Geometric Progressions
    for base in [1.5, 2.0, 2.5, 3.0, 4.0]:
        for n in [10, 20, 30, 40]:
            u = [0]
            curr = 1
            for i in range(1, n):
                curr = int(curr * base)
                u.append(curr)
            candidates.append(u)
            
    # Random Structured Sets
    for n in [20, 50, 100]:
        for mv in [1000, 5000, 10000]:
            if time.time() > _DEADLINE - 20.0:
                break
            u = [0]
            curr = 0
            growth = random.uniform(1.2, 2.0)
            for i in range(1, n):
                noise = random.uniform(0.9, 1.5)
                next_val = int(curr * growth * noise)
                if next_val <= curr:
                    next_val = curr + 1
                if next_val > mv:
                    next_val = curr + 1
                    if next_val > mv:
                        break
                u.append(next_val)
                curr = next_val
            while len(u) < n:
                next_val = u[-1] + random.randint(1, max(1, mv // n))
                if next_val > mv:
                    next_val = u[-1] + 1
                if next_val not in u:
                    u.append(next_val)
            candidates.append(sorted(set(u)))
            
    # Evaluate candidates
    for u in candidates:
        if time.time() > _DEADLINE - 10.0:
            break
        s = _score_from_set(u)
        if s > best_s:
            best_s = s
            best_u = u[:]
            
    # Phase 2: Local search from the best candidate and a few others
    # Sort candidates by score to prioritize
    scored_candidates = []
    for u in candidates:
        s = _score_from_set(u)
        scored_candidates.append((s, u))
    scored_candidates.sort(reverse=True)
    
    # Run local search on the top 5 candidates (we can afford more due to FFT speed)
    for i, (init_s, init_u) in enumerate(scored_candidates[:5]):
        if time.time() > _DEADLINE - 5.0:
            break
        if init_s < best_s * 0.9:  # Skip if too far behind
            continue
            
        u_new, s_new = _local_search(init_u, max_steps=5000, timeout=10.0)
        if s_new > best_s:
            best_s = s_new
            best_u = u_new[:]
            
    # Phase 3: Final polish on the best result
    if time.time() < _DEADLINE - 2.0:
        u_new, s_new = _local_search(best_u, max_steps=2000, timeout=5.0)
        if s_new > best_s:
            best_s = s_new
            best_u = u_new[:]
            
    # Final sanity check
    best_u = sorted(set(int(x) for x in best_u))
    if 0 not in best_u:
        best_u.append(0)
    if len(best_u) < 2:
        best_u = [0, 1]
        
    return best_u
