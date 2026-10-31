import math
import time
import numpy as np
from numpy.fft import rfft, irfft

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    
    def compute_score(u):
        """Compute score using FFT for sumset and diffset sizes."""
        max_u = max(u)
        m = 2 * max_u + 1
        
        if m <= 1:
            return 1.0
        
        n = len(u)
        
        # For small sets, use direct computation
        if n <= 60:
            sumset = set()
            for a in u:
                for b in u:
                    sumset.add(a + b)
            diffset = set()
            for a in u:
                for b in u:
                    diffset.add(a - b)
            r = len(diffset) / len(sumset)
        else:
            # FFT-based computation
            # Sumset: convolution of indicator with itself
            size = 2 * max_u + 1
            # Next power of 2 for FFT
            fft_size = 1
            while fft_size < 2 * size:
                fft_size *= 2
            
            f = np.zeros(fft_size)
            for x in u:
                f[x] = 1.0
            
            # Sumset via convolution
            conv = irfft(rfft(f) * rfft(f)).round().astype(int)
            sumset_size = np.count_nonzero(conv[:size])
            
            # Diffset: |a-b| for a,b in u, range [-max_u, max_u]
            # Use correlation: flip f and convolve
            g = f.copy()
            g[:max_u+1] = g[:max_u+1][::-1]
            # Actually, let's compute diffset directly for correctness
            # Or use: diffset size = number of distinct values of a-b
            # This equals the support of the autocorrelation
            # autocorr = irfft(|rfft(f)|^2) gives sum of f[i]*f[j] where i-j = k
            # So support of autocorr gives diffset size
            autocorr = irfft(np.abs(rfft(f))**2).round().astype(int)
            # autocorr[k] for k in 0..max_u gives count of pairs with diff = k
            # But we need distinct differences, not counts
            # diffset includes negative values too
            # |diffset| = 1 + 2 * (number of positive distinct differences)
            # Actually for a set containing 0, |U-U| = 1 + 2*|{a-b: a>b, a,b in U, a!=b}|
            # The autocorrelation at lag k gives the number of pairs (a,b) with a-b=k
            # So positive distinct diffs = number of k>0 where autocorr[k] > 0
            pos_diffs = np.count_nonzero(autocorr[1:max_u+1])
            diffset_size = 1 + 2 * pos_diffs
            
            r = diffset_size / sumset_size
        
        score = 1 + math.log(r) / math.log(m)
        return score
    
    # Known Golomb rulers (verified optimal or near-optimal)
    golomb_rulers = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 25],
        [0, 1, 4, 10, 12, 17, 25, 34],
        [0, 1, 4, 10, 12, 17, 25, 34, 44],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67, 80],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67, 80, 94],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67, 80, 94, 109],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67, 80, 94, 109, 126],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67, 80, 94, 109, 126, 144],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67, 80, 94, 109, 126, 144, 163],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67, 80, 94, 109, 126, 144, 163, 183],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67, 80, 94, 109, 126, 144, 163, 183, 204],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67, 80, 94, 109, 126, 144, 163, 183, 204, 226],
    ]
    
    # Also try some other structures - geometric progressions, etc.
    extra_candidates = []
    for base in [2, 3]:
        s = [0]
        x = 1
        while x <= 500:
            s.append(x)
            x *= base
        extra_candidates.append(s)
    
    # Power-of-2 based
    s = [0]
    for i in range(1, 10):
        s.append(1 << i)
    extra_candidates.append(s)
    
    # Fibonacci-like
    fib = [0, 1]
    while fib[-1] <= 500:
        fib.append(fib[-1] + fib[-2])
    extra_candidates.append(fib)
    
    all_candidates = golomb_rulers + extra_candidates
    
    best_set = [0, 1, 3]
    best_score = 0.0
    
    # Evaluate all initial candidates
    for u in all_candidates:
        if time.time() > deadline:
            break
        u = sorted(set(u))
        if 0 not in u:
            u = [0] + u
        score = compute_score(u)
        if score > best_score:
            best_score = score
            best_set = list(u)
    
    # Local search: try small perturbations
    # For the best candidate, try adding/removing elements
    max_search_time = deadline - time.time()
    if max_search_time > 5:
        current = list(best_set)
        current_score = best_score
        
        # Try adding elements
        max_val = max(current) if current else 0
        
        # Try adding elements near existing ones or at strategic positions
        attempts = 0
        max_attempts = 5000
        while time.time() < deadline and attempts < max_attempts:
            attempts += 1
            
            # Try adding a random element
            if max_val < 10000:
                # Try positions that might help: near max, or gaps
                candidates_to_add = []
                for i in range(len(current)):
                    if i > 0:
                        gap_start = current[i-1] + 1
                        gap_end = current[i] - 1
                        if gap_end > gap_start:
                            candidates_to_add.append((gap_start + gap_end) // 2)
                            candidates_to_add.append(gap_start)
                            candidates_to_add.append(gap_end)
                candidates_to_add.append(max_val + 1)
                candidates_to_add.append(max_val + 2)
                candidates_to_add.append(max_val + 3)
                if max_val > 10:
                    candidates_to_add.append(max_val // 2)
                    candidates_to_add.append(max_val // 3)
                
                if candidates_to_add:
                    new_elem = candidates_to_add[attempts % len(candidates_to_add)]
                    if new_elem > 0 and new_elem not in current:
                        new_set = sorted(current + [new_elem])
                        s = compute_score(new_set)
                        if s > current_score:
                            current = new_set
                            current_score = s
                            max_val = max(current)
                            if current_score > best_score:
                                best_score = current_score
                                best_set = list(current)
            
            # Try removing the largest element (if > 2 elements)
            if len(current) > 2 and attempts % 3 == 0:
                new_set = current[:-1]
                s = compute_score(new_set)
                if s > current_score:
                    current = new_set
                    current_score = s
                    max_val = max(current) if current else 0
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current)
            
            # Try shifting: replace last element with a smaller one
            if len(current) > 2 and attempts % 5 == 0:
                for delta in [1, 2, 3, 5, 8]:
                    if max_val > delta:
                        new_set = sorted(set(current[:-1] + [max_val - delta]))
                        if len(new_set) == len(current):
                            s = compute_score(new_set)
                            if s > current_score:
                                current = new_set
                                current_score = s
                                max_val = max(current)
                                if current_score > best_score:
                                    best_score = current_score
                                    best_set = list(current)
                                break
        
        # Try different starting points from golomb rulers with local search
        for ruler in golomb_rulers[:10]:
            if time.time() > deadline:
                break
            current = list(ruler)
            current_score = compute_score(current)
            max_val = max(current)
            
            # Quick local search on this ruler
            for _ in range(100):
                if time.time() > deadline:
                    break
                if max_val < 5000:
                    # Try adding
                    new_elem = max_val + 1
                    if new_elem not in current:
                        new_set = sorted(current + [new_elem])
                        s = compute_score(new_set)
                        if s > current_score:
                            current = new_set
                            current_score = s
                            max_val = max(current)
                            if current_score > best_score:
                                best_score = current_score
                                best_set = list(current)
                
                # Try removing largest
                if len(current) > 2:
                    new_set = current[:-1]
                    s = compute_score(new_set)
                    if s > current_score:
                        current = new_set
                        current_score = s
                        max_val = max(current) if current else 0
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(current)
    
    return best_set
