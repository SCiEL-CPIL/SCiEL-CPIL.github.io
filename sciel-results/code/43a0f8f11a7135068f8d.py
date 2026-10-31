import math
import time
import random
import numpy as np

def construct():
    START_TIME = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s kill

    def elapsed():
        return time.time() - START_TIME

    def remaining():
        return DEADLINE - elapsed() - 3  # 3s safety margin

    def compute_score(u):
        """Compute the score for a set u."""
        u = sorted(set(u))
        if not u or 0 not in u:
            return -1.0
        max_u = max(u)
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0

        n = len(u)
        if n <= 20:
            arr = np.array(u, dtype=np.int64)
            sums = arr[:, None] + arr[None, :]
            diffs = arr[:, None] - arr[None, :]
            sumset_size = len(np.unique(sums))
            diffset_size = len(np.unique(diffs))
        else:
            # FFT-based approach
            max_val = max_u
            n_fft = 1
            while n_fft < 2 * (2 * max_val + 1):
                n_fft *= 2

            indicator = np.zeros(n_fft, dtype=np.float64)
            for val in u:
                indicator[val] = 1.0

            ft = np.fft.fft(indicator)
            sum_conv = np.fft.ifft(ft * ft).real
            sumset_size = np.sum(np.round(sum_conv) > 0.5)

            diff_conv = np.fft.ifft(ft * np.conj(ft)).real
            half = n_fft // 2
            positive_lags = np.sum(np.round(diff_conv[:half+1]) > 0.5)
            negative_lags = np.sum(np.round(diff_conv[half+1:]) > 0.5)
            diffset_size = positive_lags + negative_lags

        r = diffset_size / sumset_size
        return 1 + math.log(r) / math.log(m)

    def is_golomb(u):
        """Check if u is a Golomb ruler (all pairwise differences are distinct)."""
        u = sorted(u)
        diffs = set()
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                d = u[j] - u[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    # Bitset-based marginal gain computation
    class BitSetTracker:
        def __init__(self, u, max_val):
            """Initialize with a set u and track sum/diff sets using bitsets."""
            self.u = sorted(u)
            self.max_val = max_val
            # Bitset for sums: index s means s is in U+U
            # Max sum is 2*max_val, so we need bits 0..2*max_val
            self.sum_bits = 0
            # Bitset for diffs: index (d + max_val) means d is in U-U
            # Diffs range from -max_val to max_val, so we need bits 0..2*max_val
            self.diff_bits = 0
            
            for a in self.u:
                for b in self.u:
                    s = a + b
                    self.sum_bits |= (1 << s)
                    d = a - b + max_val  # offset to make non-negative
                    self.diff_bits |= (1 << d)
            
            self.sum_size = self.sum_bits.bit_count()
            self.diff_size = self.diff_bits.bit_count()

        def marginal_gain(self, x):
            """Compute the change in sum_size and diff_size if we add x to U.
            Returns (new_sum_size, new_diff_size)."""
            new_sum_bits = self.sum_bits
            new_diff_bits = self.diff_bits
            
            for a in self.u:
                # New sums: x + a
                s = x + a
                new_sum_bits |= (1 << s)
                # New diffs: x - a and a - x
                d1 = x - a + self.max_val
                d2 = a - x + self.max_val
                new_diff_bits |= (1 << d1)
                new_diff_bits |= (1 << d2)
            
            # Also need x + x = 2x for sums (already covered if a=x, but x not in u yet)
            s = 2 * x
            new_sum_bits |= (1 << s)
            # x - x = 0 for diffs
            d0 = 0 + self.max_val
            new_diff_bits |= (1 << d0)
            
            return new_sum_bits.bit_count(), new_diff_bits.bit_count()

        def add_element(self, x):
            """Actually add x to the set and update bitsets."""
            self.u.append(x)
            self.u.sort()
            
            for a in self.u[:-1]:  # all previous elements
                s = x + a
                self.sum_bits |= (1 << s)
                d1 = x - a + self.max_val
                d2 = a - x + self.max_val
                self.diff_bits |= (1 << d1)
                self.diff_bits |= (1 << d2)
            
            s = 2 * x
            self.sum_bits |= (1 << s)
            d0 = 0 + self.max_val
            self.diff_bits |= (1 << d0)
            
            self.sum_size = self.sum_bits.bit_count()
            self.diff_size = self.diff_bits.bit_count()

    def fast_greedy_grow(base_u, time_limit, max_size=30, max_val=200):
        """Greedy growth using bitset marginal gains."""
        start_time = time.time()
        best_u = list(base_u)
        best_score_local = compute_score(best_u)
        
        # Initialize tracker
        current_max = max(best_u)
        tracker = BitSetTracker(best_u, max_val)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            n = len(best_u)
            if n >= max_size:
                break
            
            best_candidate = None
            best_new_score = best_score_local
            
            # Try adding each candidate
            search_hi = min(max_val, current_max + max(20, n * 5))
            candidates = list(range(1, search_hi + 1))
            random.shuffle(candidates)
            
            for new_val in candidates:
                if (time.time() - start_time) > time_limit:
                    break
                if new_val in best_u:
                    continue
                
                new_sum_size, new_diff_size = tracker.marginal_gain(new_val)
                
                # Compute new max
                new_max = max(current_max, new_val)
                m = 2 * new_max + 1
                if m <= 1:
                    continue
                
                r = new_diff_size / new_sum_size
                s = 1 + math.log(r) / math.log(m)
                
                if s > best_new_score + 1e-9:
                    best_new_score = s
                    best_candidate = new_val
            
            if best_candidate is not None:
                best_u.append(best_candidate)
                best_u.sort()
                tracker.add_element(best_candidate)
                current_max = max(best_u)
                best_score_local = best_new_score
                improved = True
        
        return best_u, best_score_local

    def bitset_local_search(base_u, time_limit, max_val=300):
        """Local search using bitset marginal gains for replacement moves."""
        start_time = time.time()
        best_u = list(base_u)
        best_score_local = compute_score(best_u)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            n = len(best_u)
            current_max = max(best_u)
            
            indices_to_try = list(range(1, n))  # skip index 0 (must be 0)
            random.shuffle(indices_to_try)
            
            for idx in indices_to_try:
                if (time.time() - start_time) > time_limit:
                    break
                
                old_val = best_u[idx]
                
                # Try replacing old_val with new_val
                window = max(15, n * 4)
                lo = max(1, old_val - window)
                hi = min(max_val, old_val + window)
                
                best_replacement = None
                best_new_score = best_score_local
                
                for new_val in range(lo, hi + 1):
                    if (time.time() - start_time) > time_limit:
                        break
                    if new_val == old_val or new_val in best_u:
                        continue
                    
                    # Compute score with replacement
                    candidate = sorted([x if x != old_val else new_val for x in best_u])
                    
                    # Quick Golomb check
                    if not is_golomb(candidate):
                        continue
                    
                    s = compute_score(candidate)
                    if s > best_new_score + 1e-9:
                        best_new_score = s
                        best_replacement = new_val
                
                if best_replacement is not None:
                    best_u[idx] = best_replacement
                    best_u.sort()
                    best_score_local = best_new_score
                    improved = True
                    break  # Restart with new best
        
        return best_u, best_score_local

    # Known good Golomb rulers (optimal or near-optimal)
    known_rulers = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 23, 29, 37, 40],
        [0, 1, 4, 10, 18, 23, 29, 37, 47, 52],
        [0, 1, 4, 10, 18, 23, 29, 37, 47, 58, 64],
        [0, 1, 4, 10, 18, 23, 29, 37, 47, 58, 64, 75],
    ]

    best_set = [0, 1, 3]
    best_score = compute_score(best_set)

    for u in known_rulers:
        if not is_golomb(u):
            continue
        s = compute_score(u)
        if s > best_score:
            best_score = s
            best_set = list(u)

    # Phase 1: Greedy growth with bitset marginal gains
    if remaining() > 30:
        t1 = min(remaining() - 10, 50)
        grow_u, grow_score = fast_greedy_grow(best_set, t1, max_size=25, max_val=200)
        if grow_score > best_score:
            best_set = grow_u
            best_score = grow_score

    # Phase 2: Local search
    if remaining() > 20:
        t2 = min(remaining() - 5, 40)
        local_u, local_score = bitset_local_search(best_set, t2, max_val=300)
        if local_score > best_score:
            best_set = local_u
            best_score = local_score

    # Phase 3: More greedy growth from improved set
    if remaining() > 15:
        t3 = min(remaining() - 3, 30)
        grow2_u, grow2_score = fast_greedy_grow(best_set, t3, max_size=30, max_val=300)
        if grow2_score > best_score:
            best_set = grow2_u
            best_score = grow2_score

    # Phase 4: Final local search
    if remaining() > 10:
        t4 = remaining() - 2
        final_u, final_score = bitset_local_search(best_set, t4, max_val=400)
        if final_score > best_score:
            best_set = final_u
            best_score = final_score

    return best_set
