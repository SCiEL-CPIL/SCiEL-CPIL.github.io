"""
Sums and differences of finite sets - spectral optimization approach.

Strategy: Use FFT to compute |U+U| and |U-U| in O(M log M) time,
enabling stochastic local search over large sets (n > 50).
"""
import time
import math
import random
import numpy as np

def _compute_score(set_U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    if not set_U or len(set_U) == 0:
        return 0.0
    max_u = max(set_U)
    if max_u == 0:
        return 0.0
    # |U-U|: difference set
    # For a set U, |U-U| = number of distinct values of a-b for a,b in U
    # Using FFT: autocorrelation of indicator
    n = len(set_U)
    max_val = max_u
    # Sumset: U+U has values in [0, 2*max_val]
    # Diffset: U-U has values in [-max_val, max_val]
    
    # Build indicator array for sumset computation
    size_sum = 2 * max_val + 1
    ind = np.zeros(size_sum, dtype=np.float64)
    for x in set_U:
        ind[x] = 1.0
    
    # Sumset: convolution of ind with itself
    # |U+U| = number of positions where (ind * ind)[k] > 0
    # Use FFT
    fft_size = 1
    while fft_size < 2 * size_sum:
        fft_size *= 2
    
    F = np.fft.rfft(ind, n=fft_size)
    conv = np.fft.irfft(F * F, n=fft_size)
    # conv[k] = sum_{i} ind[i] * ind[k-i] for i where both are in [0, size_sum-1]
    # The sumset U+U = {a+b : a,b in U}
    # conv[k] > 0 iff k is in U+U
    sumset_size = np.sum(conv > 0.5)  # threshold to avoid numerical issues
    
    # Diffset: U-U = {a-b : a,b in U}
    # |U-U| = number of distinct a-b
    # This is the support of the cross-correlation
    # cross_corr[k] = sum_i ind[i] * ind[i+k] for k >= 0
    # For k < 0, it's sum_i ind[i] * ind[i-|k|]
    # Actually, |U-U| can be computed as: for each d in [-max, max], check if exists a,b in U with a-b=d
    # Equivalently, the number of d such that sum_i ind[i]*ind[i+d] > 0 (for d>=0) plus symmetric
    
    # Use the autocorrelation
    # autocorr[d] = sum_i ind[i] * ind[i+d] for d >= 0
    # |U-U| = 1 + 2 * sum_{d=1}^{max} [autocorr[d] > 0]
    # Wait, that's not quite right. Let me think again.
    # U-U = {a-b : a,b in U}. For d >= 0, d in U-U iff exists a,b with a-b=d, i.e., exists b with b and b+d in U.
    # So autocorr[d] = sum_b ind[b]*ind[b+d] > 0 iff d in U-U (for d >= 0)
    # Similarly for d < 0, -d in U-U iff d in U-U by symmetry (since if a-b=d then b-a=-d)
    # So |U-U| = 1 + 2 * #{d in 1..max : autocorr[d] > 0}
    
    # autocorr via FFT
    # autocorr[d] = sum_i ind[i] * ind[i+d]
    # This is the inverse FFT of |F|^2
    acorr = np.fft.irfft(F * np.conj(F), n=fft_size)
    # acorr[d] for d >= 0 gives sum_i ind[i] * ind[i+d] (with wraparound, but our array is zero-padded)
    # Actually acorr from irfft(|F|^2) gives the circular autocorrelation
    # Since we zero-padded, for d < fft_size - size_sum + 1, it's the linear autocorrelation
    
    # The linear autocorrelation: acorr_lin[d] = sum_{i=0}^{size_sum-1-d} ind[i]*ind[i+d] for d >= 0
    # From the FFT result, acorr[d] for 0 <= d < size_sum gives us this (since fft_size >= 2*size_sum)
    
    positive_d = acorr[1:max_val+1]
    num_positive = np.sum(positive_d > 0.5)
    diffset_size = 1 + 2 * num_positive
    
    ratio = diffset_size / max(sumset_size, 1)
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * max_val + 1)
    if denom <= 0:
        return 0.0
    score = 1.0 + math.log(ratio) / denom
    return score


def _fast_score(set_U):
    """Faster score computation using direct sets for small U, FFT for large U."""
    if not set_U:
        return 0.0
    max_u = max(set_U)
    if max_u == 0:
        return 0.0
    
    n = len(set_U)
    # For small sets, direct computation is faster
    if n <= 60:
        U = set(set_U)
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        sumset_size = len(sumset)
        diffset_size = len(diffset)
    else:
        # FFT-based
        size_sum = 2 * max_u + 1
        ind = np.zeros(size_sum, dtype=np.float64)
        for x in set_U:
            ind[x] = 1.0
        
        fft_size = 1
        while fft_size < 2 * size_sum:
            fft_size *= 2
        
        F = np.fft.rfft(ind, n=fft_size)
        conv = np.fft.irfft(F * F, n=fft_size)
        sumset_size = np.sum(conv > 0.5)
        
        acorr = np.fft.irfft(F * np.conj(F), n=fft_size)
        positive_d = acorr[1:max_u+1]
        num_positive = np.sum(positive_d > 0.5)
        diffset_size = 1 + 2 * num_positive
    
    ratio = diffset_size / max(sumset_size, 1)
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom


def _build_geometric(n, ratio):
    """Build a set from a geometric-like structure with n elements."""
    if n <= 1:
        return [0]
    # Try different base structures
    # Option 1: powers of 2 scaled
    # Option 2: arithmetic progression with gaps
    # Start with a simple construction and let search refine
    return list(range(n))


def construct():
    """Construct a finite set U of nonnegative integers containing 0 to maximize the score."""
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Initial set: start with a reasonable size
    # The paper mentions sizes 2003 and 54265, but we need to find good configurations
    # Start with a moderate size and grow
    
    random.seed(42)
    
    best_set = [0, 1, 3]
    best_score = _fast_score(best_set)
    
    # Try a few initial configurations
    for n_init in [10, 20, 50, 100, 200]:
        if time.time() > deadline - 20:
            break
        # Random set with 0
        if n_init == 1:
            candidate = [0]
        else:
            max_val = n_init * 3
            candidate = [0] + random.sample(range(1, max_val + 1), min(n_init - 1, max_val))
            candidate.sort()
        score = _fast_score(candidate)
        if score > best_score:
            best_score = score
            best_set = list(candidate)
    
    # Stochastic local search
    current_set = list(best_set)
    current_score = best_score
    
    max_iters = 50000
    iters = 0
    
    while time.time() < deadline - 5:
        iters += 1
        
        # Choose mutation strategy
        strategy = random.random()
        
        if strategy < 0.3:
            # Add a random element
            max_u = max(current_set) if current_set else 0
            max_val = max_u * 2 + 100
            new_elem = random.randint(0, max_val)
            if new_elem not in current_set:
                candidate = list(current_set) + [new_elem]
                candidate.sort()
        elif strategy < 0.6:
            # Remove a random element (keep 0)
            if len(current_set) > 2:
                idx = random.randint(1, len(current_set) - 1)  # Don't remove 0
                candidate = list(current_set)
                candidate.pop(idx)
            else:
                candidate = list(current_set)
        elif strategy < 0.8:
            # Replace a random element (not 0)
            if len(current_set) > 2:
                idx = random.randint(1, len(current_set) - 1)
                max_u = max(current_set)
                max_val = max_u * 2 + 100
                new_elem = random.randint(0, max_val)
                candidate = list(current_set)
                candidate[idx] = new_elem
                candidate.sort()
                # Remove duplicates
                seen = set()
                deduped = []
                for x in candidate:
                    if x not in seen:
                        seen.add(x)
                        deduped.append(x)
                candidate = deduped
                if 0 not in candidate:
                    candidate.insert(0, 0)
            else:
                candidate = list(current_set)
        else:
            # Local perturbation: shift some elements
            candidate = list(current_set)
            for i in range(1, min(len(candidate), 5)):
                if random.random() < 0.5:
                    candidate[i] += random.randint(-10, 10)
                    candidate[i] = max(0, candidate[i])
            candidate.sort()
            # Remove duplicates
            seen = set()
            deduped = []
            for x in candidate:
                if x not in seen:
                    seen.add(x)
                    deduped.append(x)
            candidate = deduped
            if 0 not in candidate:
                candidate.insert(0, 0)
        
        if not candidate or 0 not in candidate:
            continue
        
        new_score = _fast_score(candidate)
        
        # Accept if better, or with small probability (simulated annealing style)
        if new_score > current_score:
            current_set = candidate
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_set = list(candidate)
        elif random.random() < 0.01:
            # Occasionally accept worse solutions to escape local optima
            current_set = candidate
            current_score = new_score
    
    return best_set


# Test
if __name__ == "__main__":
    U = construct()
    print(f"Set size: {len(U)}")
    print(f"Max element: {max(U)}")
    print(f"Score: {_fast_score(U)}")
