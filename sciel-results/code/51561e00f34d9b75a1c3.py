"""Rewritten program using FFT for fast sumset/difference-set computation."""
import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We aim to finish by 145s.
    deadline = start_time + 145.0
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    def compute_score(U):
        """
        Computes the score using FFT for sumset and difference set sizes.
        U: list of non-negative integers containing 0.
        """
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1.0
        max_u = max(U)
        if max_u == 0:
            return 0.0
        
        # Create indicator vector
        # Size needs to be at least 2*max_u + 1 for sums (0..2*max_u)
        # For differences, range is -max_u..max_u, so size 2*max_u+1 is enough for absolute values?
        # Actually, diff set contains values in [-max_u, max_u].
        # Sum set contains values in [0, 2*max_u].
        
        # We can use one FFT to get both?
        # Convolution of A with A gives sumset counts.
        # Autocorrelation of A with A gives difference set counts (shifted).
        
        # Let's do convolution for sums.
        n = len(U)
        # Next power of 2 for FFT efficiency
        size = 1
        while size < 2 * max_u + 1:
            size *= 2
            
        # Indicator vector
        vec = np.zeros(size)
        for x in U:
            vec[x] = 1.0
            
        # Sumset: Convolution of vec with vec
        # The sumset size is the number of non-zero entries in the convolution.
        # Note: Convolution of real vectors is real.
        fft_vec = np.fft.rfft(vec)
        conv = np.fft.irfft(fft_vec * fft_vec, n=size)
        
        # The sumset is supported on [0, 2*max_u].
        # Due to floating point errors, we threshold.
        # Threshold should be > 0.5 to count as present.
        sum_counts = conv[:2*max_u + 1]
        n_sum = np.sum(sum_counts > 0.5)
        
        # Difference set: Autocorrelation
        # Autocorrelation R[k] = sum_x A[x] A[x+k].
        # This gives counts for differences d such that a-b=d.
        # The difference set size is the number of distinct differences.
        # We can compute autocorrelation via FFT: ifft(fft(A) * conj(fft(A)))
        # The result is circular autocorrelation. We need linear.
        # Since vec is zero-padded to size >= 2*max_u+1, the linear autocorrelation
        # is embedded in the circular one.
        
        ac = np.fft.irfft(fft_vec * np.conj(fft_vec), n=size)
        
        # The autocorrelation ac[k] corresponds to sum_x A[x] A[x+k] for k>=0?
        # Actually, for real signals, ac[k] = sum_x A[x] A[x+k].
        # The differences d = a-b can be negative.
        # The set of differences is symmetric.
        # ac[k] counts pairs (a,b) such that a-b = -k? Or a-b=k?
        # Let's check: ac[k] = sum_x A[x] A[x+k]. Let y=x+k. sum_y A[y-k] A[y].
        # This is correlation at lag k.
        # If we want count of d = a-b, we look at lags.
        # The number of distinct differences is the number of non-zero lags (including 0) in the autocorrelation.
        # Since the support of A is [0, max_u], the differences are in [-max_u, max_u].
        # The autocorrelation computed by IFFT of |FFT(A)|^2 gives the circular autocorrelation.
        # For zero-padded A, ac[k] for k=0..size-1 contains the linear autocorrelation.
        # Specifically, ac[k] for k=0..max_u corresponds to lags 0..max_u.
        # ac[size-k] corresponds to lags -k..-1.
        
        # So we check non-zero entries in ac[0..max_u] and ac[size-max_u..size-1].
        # But since the set is symmetric, we just need to count distinct absolute differences.
        # Actually, the question asks for |U-U|, which is the size of the set {a-b | a,b in U}.
        # This set is symmetric and contains 0.
        # So |U-U| = 1 + 2 * (number of positive differences).
        
        # Let's count non-zero entries in ac[1..max_u].
        # Threshold for floating point.
        pos_diffs = ac[1:max_u+1]
        n_pos_diffs = np.sum(pos_diffs > 0.5)
        n_diff = 1 + 2 * n_pos_diffs
        
        if n_sum == 0:
            return -1.0
            
        base = 2 * max_u + 1
        if base <= 1:
            return 0.0
        
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if max(cand) == 0:
            return
        
        # Optional: prune if max_u is too large for current best?
        # Score is 1 + log(ratio)/log(2*max_u+1).
        # If max_u is very large, the denominator is large.
        # But ratio can be large for Golomb rulers.
        
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
            # print(f"New best: {s:.4f} for set size {len(cand)}, max {max(cand)}")

    # Phase 1: Test some known good structured sets
    # Small sets
    for i in range(1, 10):
        for j in range(i+1, 15):
            for k in range(j+1, 20):
                try_set([0, i, j, k])
                
    # Golomb Rulers (small)
    # Known optimal Golomb rulers
    grs = [
        [0, 1, 4, 10], # length 10
        [0, 1, 4, 10, 18], # length 18
        [0, 1, 4, 10, 18, 23], # length 23
        [0, 1, 4, 10, 18, 23, 31], # length 31
        [0, 1, 4, 10, 18, 23, 31, 37], # length 37
        [0, 1, 4, 10, 18, 23, 31, 37, 41], # length 41
    ]
    for gr in grs:
        try_set(gr)
        
    # Phase 2: Randomized Local Search
    # We will run multiple iterations of local search starting from different seeds.
    # Seeds: Random sets, Golomb rulers, Arithmetic progressions.
    
    num_starts = 20
    max_steps_per_start = 100
    
    # Generate seeds
    seeds = []
    
    # 1. Random sets of various sizes
    for size in [4, 5, 6, 8, 10, 15]:
        for _ in range(5):
            # Random max_u
            max_u = random.randint(10, 100)
            elems = random.sample(range(1, max_u + 1), size - 1)
            seeds.append(sorted([0] + elems))
            
    # 2. Golomb rulers
    seeds.extend(grs)
    
    # 3. Powers of 2
    for base in [2, 3, 4]:
        p = [0, 1]
        while p[-1] * base < 100:
            p.append(p[-1] * base)
        seeds.append(p)
        
    # 4. Arithmetic progressions
    for d in range(1, 10):
        ap = list(range(0, 50, d))
        seeds.append(ap)
        
    # Shuffle seeds
    random.shuffle(seeds)
    
    for seed in seeds:
        if time.time() > deadline - 5.0:
            break
            
        current = list(seed)
        try_set(current)
        
        # Local search
        no_improve = 0
        max_no_improve = 20
        
        while no_improve < max_no_improve and time.time() < deadline - 5.0:
            improved = False
            
            # 1. Try adding an element
            if len(current) < 20: # Limit size to keep FFT fast
                max_cur = max(current)
                # Candidates: near existing, or extending range
                candidates = set()
                for x in current:
                    for dx in [-3, -2, -1, 1, 2, 3]:
                        if x + dx > 0 and x + dx not in current:
                            candidates.add(x + dx)
                # Extend range
                for ext in range(1, 10):
                    candidates.add(max_cur + ext)
                
                # Also try random elements in range
                for _ in range(5):
                    candidates.add(random.randint(1, max_cur + 20))
                
                for new_el in sorted(candidates):
                    if new_el in current:
                        continue
                    cand = sorted(current + [new_el])
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        break
                    elif s > compute_score(current) - 0.01: # Accept slightly worse to escape local minima?
                        # Keep it simple: only accept improvements for now.
                        pass
                        
            # 2. Try replacing an element
            if len(current) >= 3:
                for i in range(1, len(current)): # Don't replace 0
                    orig = current[i]
                    # Try nearby values
                    for new_el in range(max(1, orig - 5), orig + 6):
                        if new_el in current:
                            continue
                        cand = list(current)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand:
                            continue
                        s = compute_score(cand)
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                            current = list(cand)
                            improved = True
                            break
                    if improved:
                        break
                
            # 3. Try removing an element
            if len(current) > 3:
                for i in range(1, len(current)):
                    cand = [x for j, x in enumerate(current) if j != i]
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        break
                        
            if not improved:
                no_improve += 1
            else:
                no_improve = 0
                
    return list(best_set)
