import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    def compute_score_fft(U):
        """
        Computes the score using FFT for speed.
        Score = 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)
        """
        U = sorted(set(U))
        if not U or 0 not in U or len(U) < 2:
            return 0.0
        
        max_u = max(U)
        if max_u == 0:
            return 0.0
            
        # FFT size needs to be at least 2*max_u + 1 for linear convolution
        size = 1
        while size < 2 * max_u + 1:
            size *= 2
        
        # Create boolean array for U
        arr = np.zeros(size, dtype=np.float64)
        for x in U:
            arr[x] = 1.0
            
        # FFT of U
        fft_u = np.fft.fft(arr)
        
        # Sumset: U+U
        sumset_fft = fft_u * fft_u
        sumset_conv = np.fft.ifft(sumset_fft).real
        n_sum = np.sum(sumset_conv > 1e-6)
        
        # Difference set: U-U (Autocorrelation)
        corr_fft = fft_u * np.conj(fft_u)
        corr = np.fft.ifft(corr_fft).real
        n_diff = np.sum(corr > 1e-6)
        
        if n_sum == 0 or n_diff == 0:
            return 0.0
            
        base = 2 * max_u + 1
        if base <= 1:
            return 0.0
            
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
            
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def compute_score_naive(U):
        """Fallback for small sets or verification"""
        U = sorted(set(U))
        if not U or 0 not in U or len(U) < 2:
            return 0.0
        max_u = max(U)
        if max_u == 0:
            return 0.0
            
        sumset = set()
        diffset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
                
        n_sum = len(sumset)
        n_diff = len(diffset)
        
        if n_sum == 0:
            return 0.0
            
        base = 2 * max_u + 1
        if base <= 1:
            return 0.0
            
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
            
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def get_score(U):
        if max(U) > 100:
            return compute_score_fft(U)
        else:
            return compute_score_naive(U)

    # Strategy: Search for optimal Digit Set D
    # Score contribution of D: log(|D-D|/|D+D|) / log(max(D+D)+1)
    # We want to maximize this.
    
    best_digit_set = None
    best_digit_score = 0.0
    best_digit_ratio = 1.0
    
    # Search space for D:
    # D must contain 0.
    # Size of D up to 10.
    # Max element of D up to 20.
    
    # Pre-compute scores for small D
    # We will iterate through combinations of elements
    
    max_d_size = 8
    max_d_val = 15
    
    # Generate candidate D sets
    # To be efficient, we can just iterate over all subsets of {0, 1, ..., max_d_val} of size <= max_d_size
    # But 2^15 is 32768, which is manageable.
    
    all_elements = list(range(max_d_val + 1))
    
    # Use itertools.combinations? Or just recursive generation?
    # Let's just iterate through all masks of length max_d_val+1 with popcount <= max_d_size
    # And ensure 0 is included.
    
    candidates = []
    
    # Bitmask iteration is fast in Python if we do it smartly, or just use combinations
    from itertools import combinations
    
    # Include 0 always
    # Choose k-1 elements from 1..max_d_val
    for k in range(2, max_d_size + 1):
        if time.time() > deadline - 20:
            break
        for comb in combinations(range(1, max_d_val + 1), k - 1):
            D = [0] + list(comb)
            D.sort()
            
            # Quick prune: if max(D) is very large, log(max) grows, ratio might drop
            # Calculate exact score for D
            # |D+D| and |D-D|
            sumset = set()
            diffset = set()
            for a in D:
                for b in D:
                    sumset.add(a + b)
                    diffset.add(a - b)
            
            n_sum = len(sumset)
            n_diff = len(diffset)
            
            if n_sum == 0:
                continue
                
            ratio = n_diff / n_sum
            if ratio <= 1:
                continue
                
            max_sum = max(sumset)
            base_d = max_sum + 1
            if base_d <= 1:
                continue
            
            # Asymptotic score component
            # The actual score for U constructed from D with k_digits will approach:
            # 1 + log(ratio) / log(base_d)
            # Let's use this as the proxy for ranking D
            proxy_score = 1 + math.log(ratio) / math.log(base_d)
            
            if proxy_score > best_digit_score:
                best_digit_score = proxy_score
                best_digit_set = D
                best_digit_ratio = ratio
                # Store base for construction
                best_base = base_d

    # If no good D found, fallback
    if best_digit_set is None:
        best_digit_set = [0, 1, 3]
        best_base = 7
        best_digit_score = get_score(best_digit_set)

    # Construct U using the best D
    # U = { sum d_i B^i }
    # We want to make U large enough to be in the asymptotic regime, but not too large for FFT.
    # Max U approx max(D) * B^k.
    # We have ~100s left.
    # FFT of size 2*max(U) takes time proportional to N log N.
    # Let's try to fit k such that max(U) is around 10^5 to 10^6.
    
    D = best_digit_set
    B = best_base
    max_d = max(D)
    
    # Estimate max U for k digits
    # max_U_k = max_d * (B^k - 1) / (B - 1)
    
    # Let's find max k such that max_U < 500,000 (safe for FFT in < 1s)
    # If B is small, k can be larger.
    
    k_digits = 1
    current_max_u = max_d
    while True:
        next_max_u = current_max_u * B + max_d
        if next_max_u > 500000:
            break
        current_max_u = next_max_u
        k_digits += 1
        if k_digits > 20: # Cap
            break
            
    # Construct U
    U = [0]
    # Generate all combinations of digits
    # Size of U is |D|^k_digits
    # If |D|^k_digits is too large (> 10000), construction might be slow or memory heavy.
    # But |D| is small (<=8), so 8^5 = 32768. That's fine.
    
    # Iterative construction
    current_U = [0]
    for _ in range(k_digits):
        next_U = []
        for u in current_U:
            for d in D:
                next_U.append(u * B + d)
        current_U = next_U
        
    U = current_U
    
    # Verify and Score
    if 0 not in U:
        U.append(0)
    U = sorted(set(U))
    
    final_score = get_score(U)
    
    # If the asymptotic score is not reached (e.g. k=1), we might try to adjust k
    # But given the search for D was based on asymptotic score, this should be good.
    
    # Optional: Local search on U?
    # The structure is rigid. Changing one element breaks the product structure.
    # It's better to stick with the structural solution.
    
    # However, we can try a few variations of D if time permits?
    # The search above was exhaustive for small D.
    
    # Check time
    if time.time() < deadline - 10:
        # Try to see if a slightly different B helps?
        # B = max(D+D) + 1 is the minimum B to avoid carries.
        # Larger B increases denominator log(B) but doesn't change numerator log(ratio).
        # So minimum B is best.
        
        # What if we use a non-prime B? Doesn't matter for the set definition, only for carries.
        
        # Let's just return the constructed U.
        pass

    return U

if __name__ == "__main__":
    # Simple test
    res = construct()
    print(res)
