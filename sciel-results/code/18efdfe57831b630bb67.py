"""Rewritten program: FFT-accelerated local search for sumset/difference-set exponent."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    """Construct a finite set U of nonnegative integers containing 0 to maximize
    1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s kill

    def score_set(U):
        """Compute the score for a set U (list of nonnegative ints containing 0)."""
        if not U or 0 not in U:
            return 0.0
        U = list(set(U))
        U.sort()
        n = len(U)
        if n < 2:
            return 0.0
        max_u = U[-1]
        if max_u == 0:
            return 0.0
        # Compute |U+U| and |U-U| using FFT for efficiency
        # U+U: elements range from 0 to 2*max_u
        # U-U: elements range from -max_u to max_u
        M = 2 * max_u + 1
        # Use a size that's a power of 2 for FFT efficiency
        fft_size = 1
        while fft_size < 2 * M:
            fft_size *= 2
        
        # Indicator vector for U
        f = np.zeros(fft_size)
        for u in U:
            f[u] = 1.0
        
        # FFT
        F = np.fft.rfft(f)
        F2 = F * F
        
        # Inverse FFT to get convolution
        conv = np.fft.irfft(F2, n=fft_size)
        
        # |U+U| = number of elements with conv > 0 in range [0, 2*max_u]
        # Due to numerical issues, use a threshold
        threshold = 0.5
        sum_set_size = np.sum(conv[:M + 1] > threshold)
        
        # |U-U|: this is the autocorrelation, which is the same as |U+U| for the 
        # difference set when we consider |a-b| for a,b in U. Actually, U-U includes
        # both positive and negative differences. Let's compute it properly.
        # The autocorrelation gives |{a-b : a,b in U}| for non-negative differences.
        # Actually, the circular convolution approach: for difference set, we need
        # to compute the number of distinct values of a-b for a,b in U.
        # a-b ranges from -max_u to max_u.
        # The autocorrelation r[k] = sum_x f[x]*f[x+k] gives the count of pairs with
        # difference k. So |U-U| = number of k in [-max_u, max_u] where r[k] > 0.
        # In our FFT, conv[i] for i in [0, fft_size) gives the circular convolution.
        # The autocorrelation is conv[i] for i in [0, max_u] (positive diffs) and
        # conv[fft_size - i] for i in [1, max_u] (negative diffs).
        
        # Actually, let me reconsider. The convolution f*f gives (f*f)[k] = sum_i f[i]*f[k-i]
        # which counts pairs (i, k-i) both in U, i.e., pairs with sum k. So conv[k] for
        # k in [0, 2*max_u] gives the sumset counts.
        
        # For the difference set, I need the autocorrelation: sum_i f[i]*f[i+k].
        # This is equivalent to convolving f with a flipped version of f.
        
        # Let me compute the difference set size separately.
        g = np.zeros(fft_size)
        for u in U:
            g[fft_size - u if u > 0 else 0] = 1.0
        # Actually, for autocorrelation, we flip f: g[k] = f[-k] = f[fft_size - k] for k>0, g[0]=f[0]
        # But it's easier to just compute it directly for small n, or use another FFT.
        
        # For efficiency, let's compute |U-U| using the fact that the autocorrelation
        # R[k] = sum_x f[x] f[x+k] can be computed via FFT of f and conjugate of FFT(f).
        # But since f is real and non-negative, let's just do:
        # The number of distinct differences |a-b| for a,b in U.
        # Note that a-b can be negative. The set U-U = {a-b : a,b in U}.
        # |U-U| = number of distinct values.
        
        # Alternative: for the difference set, use the autocorrelation.
        # R[k] = sum_x f[x]*f[x+k] for k >= 0 gives counts of pairs with difference k.
        # R[-k] = R[k] by symmetry.
        # So |U-U| = 1 + 2 * (number of k in [1, max_u] with R[k] > 0)
        
        # Compute autocorrelation via FFT:
        # autocorr = ifft(fft(f) * conj(fft(f)))
        # But we already have F = fft(f), so:
        auto = np.fft.irfft(F * np.conj(F), n=fft_size)
        
        # auto[k] for k in [0, max_u] gives R[k]
        # The number of distinct positive differences:
        pos_diffs = np.sum(auto[1:max_u + 1] > threshold)
        diff_set_size = 1 + 2 * pos_diffs
        
        # Score: 1 + log(|U-U|/|U+U|) / log(2*max_u + 1)
        if sum_set_size <= 0:
            return 0.0
        
        ratio = diff_set_size / sum_set_size
        if ratio <= 0:
            return 0.0
        
        log_base = np.log(2 * max_u + 1)
        if log_base <= 0:
            return 0.0
        
        return 1.0 + np.log(ratio) / log_base

    def score_set_direct(U):
        """Direct computation for small sets (fallback)."""
        U = list(set(U))
        U.sort()
        n = len(U)
        if n < 2:
            return 0.0
        max_u = U[-1]
        if max_u == 0:
            return 0.0
        
        sum_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
        
        diff_set = set()
        for a in U:
            for b in U:
                diff_set.add(a - b)
        
        ratio = len(diff_set) / len(sum_set)
        if ratio <= 0:
            return 0.0
        log_base = np.log(2 * max_u + 1)
        if log_base <= 0:
            return 0.0
        return 1.0 + np.log(ratio) / log_base

    def score_set_auto(U):
        """Choose between FFT and direct based on set size."""
        U = list(set(U))
        max_u = max(U) if U else 0
        n = len(U)
        if n <= 30 and max_u <= 500:
            return score_set_direct(U)
        else:
            return score_set(U)

    # Initial set: start with a known good small set
    # The paper mentions constructions of size 2003 and 54265.
    # Start with a geometric progression-like set.
    # A set like {0, 1, 3, 9, 27, ...} (powers of 3) has good properties.
    # Let's start with a small set and grow it.
    
    # Start with the seed
    best_set = [0, 1, 3]
    best_score = score_set_auto(best_set)
    
    # Try a few initial candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 9],
        [0, 1, 3, 9, 27],
        [0, 1, 3, 9, 27, 81],
        [0, 1, 2, 4, 8, 16],
        [0, 1, 3, 6, 10, 15],
    ]
    
    for cand in candidates:
        s = score_set_auto(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # Local search: add, remove, or swap elements
    random.seed(42)
    
    # Determine the maximum value to consider
    max_val = 200  # Start modest
    
    # Local search loop
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        if random.random() < 0.3:
            # Add a new element
            new_elem = random.randint(1, max_val)
            if new_elem not in best_set:
                candidate = best_set + [new_elem]
                candidate.sort()
                s = score_set_auto(candidate)
                if s > best_score:
                    best_score = s
                    best_set = candidate
                    no_improve_count = 0
                    # Expand max_val if we added a large element
                    if new_elem > max_val * 0.8:
                        max_val = int(max_val * 1.3)
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        elif random.random() < 0.5:
            # Remove an element (keep 0)
            if len(best_set) > 2:
                removable = [x for x in best_set if x != 0]
                if removable:
                    remove_elem = random.choice(removable)
                    candidate = [x for x in best_set if x != remove_elem]
                    s = score_set_auto(candidate)
                    if s > best_score:
                        best_score = s
                        best_set = candidate
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        else:
            # Swap: remove one, add another
            if len(best_set) > 2:
                removable = [x for x in best_set if x != 0]
                if removable:
                    remove_elem = random.choice(removable)
                    add_elem = random.randint(1, max_val)
                    if add_elem not in best_set:
                        candidate = [x for x in best_set if x != remove_elem]
                        candidate.append(add_elem)
                        candidate.sort()
                        s = score_set_auto(candidate)
                        if s > best_score:
                            best_score = s
                            best_set = candidate
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        # Periodically try to expand the range
        if no_improve_count % 500 == 0 and max_val < 5000:
            max_val = int(max_val * 1.2)
    
    # Final polish: try adding elements near existing ones
    if time.time() < deadline:
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            # Try adding elements in gaps
            best_set_sorted = sorted(best_set)
            for i in range(len(best_set_sorted) - 1):
                gap = best_set_sorted[i+1] - best_set_sorted[i]
                if gap > 2:
                    mid = best_set_sorted[i] + random.randint(1, gap - 1)
                    if mid not in best_set:
                        candidate = best_set + [mid]
                        candidate.sort()
                        s = score_set_auto(candidate)
                        if s > best_score:
                            best_score = s
                            best_set = candidate
                            break
    
    # Ensure 0 is in the set
    if 0 not in best_set:
        best_set.append(0)
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
