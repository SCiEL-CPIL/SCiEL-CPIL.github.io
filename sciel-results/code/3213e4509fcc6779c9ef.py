import numpy as np
import time
import random

def count_nonzero(arr):
    return int(np.count_nonzero(arr))

def square_nonzero_count(coeffs):
    # coeffs is a list of 12 integers
    # We assume the support is 0..11
    # If we allow gaps, we need to handle indices.
    # For now, stick to 0..11 support.
    sq = np.convolve(coeffs, coeffs)
    return count_nonzero(sq)

def spectral_objective(coeffs, N=8192):
    # coeffs: list of 12 ints
    # Returns a continuous surrogate: L1 norm of the DFT of coeffs^2
    # coeffs^2 in time domain corresponds to |DFT(coeffs)|^2 in frequency domain (up to scaling)
    # Actually, P(x)^2 evaluated at roots of unity is |P(w)|^2.
    # So we can compute P(w) for N-th roots of unity, square it, and take L1 norm.
    # This is O(N log N) with FFT.
    
    # Pad coeffs to length N/2 (since degree is 11, length 12 is enough, but we need N points)
    # Wait, P(x) has degree 11. P(w) where w is N-th root of unity.
    # We can just compute the DFT of the padded coeffs vector.
    
    # Create a vector of length N with coeffs in the first 12 positions
    x = np.zeros(N, dtype=np.complex128)
    x[:12] = coeffs
    
    # Compute DFT
    X = np.fft.fft(x)
    
    # P(w_k) = X[k]
    # P(w_k)^2 = X[k]^2
    # We want L1 norm of the coefficients of P(x)^2.
    # The coefficients of P(x)^2 are the IDFT of |X|^2? No.
    # Convolution theorem: IDFT(DFT(a) * DFT(b)) = a * b (circular).
    # Since our support is small (12), linear convolution = circular convolution if N is large enough.
    # N=8192 is much larger than 12+12-1 = 23. So circular conv = linear conv.
    # So coeffs of P^2 = IDFT( X * X ).
    
    # So we compute Y = X * X
    # Then Z = IDFT(Y)
    # Then return sum(|Z|)
    
    Y = X * X
    Z = np.fft.ifft(Y)
    
    # Z is complex, but should be real (up to numerical error)
    # Take absolute values and sum
    return np.sum(np.abs(Z))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin
    
    best_count = 23
    best_coeffs = [1]*12
    
    # Try alternating signs
    alt = [1 if i%2==0 else -1 for i in range(12)]
    c = square_nonzero_count(alt)
    if c < best_count:
        best_count = c
        best_coeffs = alt.copy()
        
    # We will use a hybrid approach:
    # 1. Gradient descent on the spectral objective to find promising regions.
    # 2. Round to integers and evaluate exact count.
    # 3. Local search on the exact count.
    
    # Initialize with random small integers or the alternating sign pattern
    # Let's do multiple restarts of the gradient descent
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Initialize
        if restart == 0:
            coeffs = [1.0 if i%2==0 else -1.0 for i in range(12)]
        elif restart == 1:
            coeffs = [1.0]*12
        else:
            coeffs = [random.uniform(-5, 5) for _ in range(12)]
            # Ensure not too close to 0 to avoid issues? 
            # Actually, we want to explore.
        
        # Gradient Descent
        # Objective: spectral_objective(coeffs)
        # We need the gradient.
        # Let's use numerical gradient or analytic?
        # Analytic gradient of sum(|IDFT(X*X)|) is complex.
        # Let's use a simple numerical gradient or just a random walk guided by the spectral objective?
        # Actually, let's try to minimize the spectral objective using a simple method.
        
        # Since we don't have an easy analytic gradient, let's use a "simulated annealing" style
        # but on the continuous domain, accepting moves that improve the spectral objective.
        
        current_spec = spectral_objective(coeffs)
        
        # Run for a few steps
        for step in range(1000):
            if time.time() > deadline:
                break
            
            # Pick a random index to perturb
            idx = random.randint(0, 11)
            old_val = coeffs[idx]
            coeffs[idx] += random.gauss(0, 0.5)
            
            # Clamp to [-64, 64]
            if coeffs[idx] > 64: coeffs[idx] = 64
            if coeffs[idx] < -64: coeffs[idx] = -64
            
            new_spec = spectral_objective(coeffs)
            
            # Accept if better, or with some probability (annealing)
            # For simplicity, just accept if better (hill climbing on continuous)
            if new_spec < current_spec:
                current_spec = new_spec
            else:
                coeffs[idx] = old_val
        
        # Round to nearest integer
        int_coeffs = [int(round(c)) for c in coeffs]
        
        # Ensure non-zero
        valid = True
        for i in range(12):
            if int_coeffs[i] == 0:
                # Nudge to 1 or -1
                int_coeffs[i] = 1 if coeffs[i] >= 0 else -1
            if int_coeffs[i] < -64 or int_coeffs[i] > 64:
                valid = False
                break
        
        if not valid:
            continue
            
        exact_count = square_nonzero_count(int_coeffs)
        
        if exact_count < best_count:
            best_count = exact_count
            best_coeffs = int_coeffs.copy()
            
            # Local search on the exact count from this new best
            improved = True
            while improved and time.time() < deadline:
                improved = False
                indices = list(range(12))
                random.shuffle(indices)
                for idx in indices:
                    candidates = []
                    for delta in range(-3, 4):
                        new_val = best_coeffs[idx] + delta
                        if new_val != 0 and -64 <= new_val <= 64 and new_val != best_coeffs[idx]:
                            candidates.append(new_val)
                            
                    random.shuffle(candidates)
                    for new_val in candidates:
                        old_val = best_coeffs[idx]
                        best_coeffs[idx] = new_val
                        new_count = square_nonzero_count(best_coeffs)
                        if new_count < best_count:
                            best_count = new_count
                            improved = True
                            break
                        else:
                            best_coeffs[idx] = old_val

        if best_count == 12:
            break

    return best_coeffs
