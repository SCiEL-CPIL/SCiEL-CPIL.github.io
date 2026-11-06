import random
import time
import numpy as np

def construct():
    """
    Returns a list of 32 distinct integers in 1..512 maximizing the number
    of distinct d > 0 such that a, a+d, a+2d are in the set.
    """
    
    def exact_score(S):
        """
        Calculate the number of distinct differences d > 0 
        such that a, a+d, a+2d are in S for some a.
        """
        S_set = set(S)
        count = 0
        for d in range(1, 256):
            found = False
            for a in S_set:
                if a + d in S_set and a + 2*d in S_set:
                    found = True
                    break
            if found:
                count += 1
        return count

    def fast_score(S):
        """
        Approximate score based on Total Number of 3-APs using FFT.
        Higher total 3-APs generally correlates with more distinct differences.
        """
        # Use a length that avoids wrap-around for sums a+c up to 1024
        # Max a+c is 512+512=1024. 
        # If we use L=2048, indices 0..2047 are safe.
        L = 2048
        x = np.zeros(L)
        for val in S:
            if 1 <= val <= 512:
                x[val-1] = 1 # 0-indexed for numpy, but values are 1..512
                # Actually, let's just use the value as index if we shift?
                # Let's map 1..512 to 0..511
                # x[i] = 1 if i+1 in S
        
        # Re-do mapping carefully
        x = np.zeros(L)
        for val in S:
            x[val-1] = 1.0
            
        X = np.fft.fft(x)
        # Convolution C[k] = sum x[i] x[k-i]
        C = np.fft.ifft(X * X).real
        
        # Total 3-APs = sum over b in S of C[2b]
        # b is in 1..512, so index b-1.
        # 2b is in 2..1024.
        # C index for sum 2b is 2b.
        # Wait, C[k] corresponds to sum = k.
        # If x[i] is 1 for value i+1, then x[i]x[j] contributes to sum (i+1)+(j+1) = i+j+2.
        # So C[i+j] in the convolution of the shifted array?
        
        # Let's stick to standard:
        # Let y[k] = 1 if k in S.
        # We want sum_{a,c} y[a] y[c] where a+c = 2b.
        # Convolution Z = y * y.
        # Z[s] = sum_{a} y[a] y[s-a].
        # So Z[2b] is the count.
        
        # My x array is indexed 0..511 for values 1..512.
        # So x[i] = y[i+1].
        # Convolution of x: W[k] = sum_i x[i] x[k-i] = sum_i y[i+1] y[k-i+1].
        # Let a = i+1, c = k-i+1. a+c = k+2.
        # So W[k] counts pairs (a,c) with a+c = k+2.
        # We want a+c = 2b.
        # So k+2 = 2b => k = 2b - 2.
        # So we need to sum W[2b - 2] for all b in S.
        
        total_aps = 0
        for b in S:
            idx = 2*b - 2
            if 0 <= idx < L:
                total_aps += C[idx]
                
        return total_aps

    # Initial solution
    # Try a few random starts and pick the best by fast_score
    best_initial = None
    best_fast = -1
    
    for _ in range(20):
        candidate = random.sample(range(1, 513), 32)
        s = fast_score(candidate)
        if s > best_fast:
            best_fast = s
            best_initial = candidate
    
    # Also try a structured start: spread out
    c_struct = []
    # Take 32 points evenly spaced?
    # 1, 16, 31, ...
    # Or just random is fine for SA.
    
    current_set = best_initial[:]
    current_fast = best_fast
    
    best_exact_set = current_set[:]
    best_exact_score = exact_score(current_set)
    
    start_time = time.time()
    time_limit = 115.0
    
    temperature = 50.0 # Scale for fast score (which is larger)
    cooling = 0.9995
    min_temp = 1.0
    
    iteration = 0
    
    while time.time() - start_time < time_limit:
        # Pick a random index to modify
        idx = random.randint(0, 31)
        old_val = current_set[idx]
        
        # Generate a new value
        new_val = random.randint(1, 512)
        
        # Ensure distinctness
        if new_val in current_set:
            continue
            
        current_set[idx] = new_val
        
        new_fast = fast_score(current_set)
        
        # Acceptance
        delta = new_fast - current_fast
        if delta > 0:
            current_fast = new_fast
            # Check exact score occasionally
            if iteration % 100 == 0:
                exact_s = exact_score(current_set)
                if exact_s > best_exact_score:
                    best_exact_score = exact_s
                    best_exact_set = current_set[:]
        else:
            if temperature > min_temp:
                # Avoid underflow
                if delta > -1000:
                    prob = np.exp(delta / temperature)
                    if random.random() < prob:
                        current_fast = new_fast
                    else:
                        current_set[idx] = old_val
                else:
                    current_set[idx] = old_val
            else:
                current_set[idx] = old_val
        
        # Cool down
        temperature *= cooling
        
        # Reheat
        if temperature < min_temp:
            # Restart from best exact set or best fast set?
            # Use best exact set to bias towards the true objective
            current_set = best_exact_set[:]
            current_fast = fast_score(current_set)
            temperature = 50.0
            # Perturb
            for _ in range(5):
                p_idx = random.randint(0, 31)
                p_val = random.randint(1, 512)
                if p_val not in current_set:
                    current_set[p_idx] = p_val
            current_fast = fast_score(current_set)
            
        iteration += 1

    # Final check
    final_exact = exact_score(best_exact_set)
    # Maybe try a few local improvements on the best exact set using fast score?
    # No, time is up.
    
    return best_exact_set
