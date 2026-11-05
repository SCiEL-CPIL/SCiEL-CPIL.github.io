import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    candidates = [
        [1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1],
        [1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1],
        [1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1],
        [1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1],
        [1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1],
    ]

    M = 2**15
    k = np.arange(M)
    n_idx = np.arange(24)
    phases = np.outer(k, n_idx)
    D = np.exp(-2j * np.pi * phases / M)

    def evaluate(coeffs):
        F = D @ coeffs
        return np.max(np.abs(F)) / np.sqrt(24)

    def refine_peak(f_arr, z0, half_width=0.05, iters=50):
        """
        Golden-section search for the maximum of |P(z)| in the arc around z0.
        z0 is a complex number on the unit circle (approx).
        Returns the maximum value found and the location.
        """
        # Convert z0 to angle
        theta0 = np.angle(z0)
        
        # Define function to evaluate |P(z)| at angle theta
        def val(theta):
            z = np.exp(1j * theta)
            # P(z) = sum c_k z^k
            # We can use the FFT values? No, FFT is at discrete points.
            # We need to evaluate the polynomial directly or interpolate.
            # Direct evaluation: sum c_k * exp(i * k * theta)
            # coeffs are in current_c
            # Let's use the precomputed D columns? No, D is for discrete k.
            # We can compute P(z) using the coefficients.
            # For speed, we can use the fact that P(z) = sum c_k z^k.
            # Vectorized: np.sum(current_c * (z ** n_idx))
            # But z**n_idx is expensive.
            # Alternative: use the relationship to DFT?
            # Actually, just computing the sum is O(24), which is very fast.
            
            # To be safe and accurate, compute directly.
            # We need access to current_c. Since refine_peak is called inside loops,
            # passing coeffs is better, or using nonlocal.
            # Let's pass coeffs.
            return np.abs(np.sum(coeffs * (z ** n_idx)))
        
        # Golden section search on [theta0 - half_width, theta0 + half_width]
        a = theta0 - half_width
        b = theta0 + half_width
        gr = (math.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        fc = val(c)
        fd = val(d)
        
        for _ in range(iters):
            if fc < fd:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = val(d)
            else:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = val(c)
        
        # Final check at midpoint
        mid = (a + b) / 2
        fm = val(mid)
        
        # Also check the original z0
        f0 = val(theta0)
        
        best_val = max(fc, fd, fm, f0)
        return best_val

    def local_search_1flip(current_c, current_f, current_max):
        """Thorough 1-flip local search until no improvement."""
        improved = True
        while improved:
            improved = False
            best_flip = -1
            best_max_val = current_max
            for n in range(24):
                f_new = current_f - 2.0 * current_c[n] * D[:, n]
                max_val = np.max(np.abs(f_new))
                if max_val < best_max_val - 1e-12:
                    best_max_val = max_val
                    best_flip = n
            if best_flip != -1:
                current_c[best_flip] *= -1.0
                current_f = D @ current_c
                current_max = best_max_val
                improved = True
        return current_c, current_f, current_max

    def local_search_2flip(current_c, current_f, current_max):
        """Try 2-flips to escape local minima."""
        best_flip_2 = (-1, -1)
        best_max_2 = current_max
        for n1 in range(24):
            for n2 in range(n1 + 1, 24):
                f_new = current_f - 2.0 * (current_c[n1] * D[:, n1] + current_c[n2] * D[:, n2])
                max_val = np.max(np.abs(f_new))
                if max_val < best_max_2 - 1e-12:
                    best_max_2 = max_val
                    best_flip_2 = (n1, n2)
        if best_flip_2 != (-1, -1):
            n1, n2 = best_flip_2
            current_c[n1] *= -1.0
            current_c[n2] *= -1.0
            current_f = D @ current_c
            current_max = best_max_2
        return current_c, current_f, current_max

    best_score = float('inf')
    best_c = None

    # Evaluate all candidates and do local search on each
    for c in candidates:
        if time.time() > deadline:
            break
        c_arr = np.array(c, dtype=float)
        f_arr = D @ c_arr
        max_val = np.max(np.abs(f_arr))
        
        # Do 1-flip local search
        c_arr, f_arr, max_val = local_search_1flip(c_arr, f_arr, max_val)
        
        # Try 2-flip escapes, then re-do 1-flip
        for _ in range(5):
            if time.time() > deadline:
                break
            c_arr, f_arr, max_val = local_search_2flip(c_arr, f_arr, max_val)
            c_arr, f_arr, max_val = local_search_1flip(c_arr, f_arr, max_val)
        
        # Refine the peak found by FFT sampling
        # Find the index of the max in f_arr
        max_idx = np.argmax(np.abs(f_arr))
        # The angle corresponding to this index
        theta_max = 2 * np.pi * max_idx / M
        z_max = np.exp(1j * theta_max)
        
        # Refine around this peak
        refined_max = refine_peak(f_arr, z_max, c_arr)
        
        # The true score is max(FFT_max, refined_max) / sqrt(24)
        # But usually refined is >= FFT max if we found the right peak.
        # We should take the max of the refined value and the global FFT max?
        # Actually, the FFT max is a lower bound on the true max.
        # The refined value is an estimate of the true max at that specific peak.
        # There might be other peaks. But for local search, we assume the dominant peak is the one we found.
        # To be conservative, score = max(max_val, refined_max) / sqrt(24)
        true_max = max(max_val, refined_max)
        score = true_max / np.sqrt(24)
        
        if score < best_score:
            best_score = score
            best_c = c_arr.copy()

    # Now do iterated local search from best_c with random perturbations
    current_c = best_c.copy()
    current_f = D @ current_c
    current_max = np.max(np.abs(current_f))
    
    # Refine initial best
    max_idx = np.argmax(np.abs(current_f))
    theta_max = 2 * np.pi * max_idx / M
    z_max = np.exp(1j * theta_max)
    refined_max = refine_peak(current_f, z_max, current_c)
    current_max = max(current_max, refined_max)
    best_score = current_max / np.sqrt(24)
    best_c = current_c.copy()

    rng = np.random.RandomState(42)

    while time.time() < deadline:
        # Try 1-flip
        best_flip_1 = -1
        best_max_1 = current_max
        for n in range(24):
            f_new = current_f - 2.0 * current_c[n] * D[:, n]
            max_val = np.max(np.abs(f_new))
            # Quick check: if FFT max doesn't improve significantly, skip refinement?
            # But the peak location might shift.
            # Let's just use FFT max for screening.
            if max_val < best_max_1 - 1e-12:
                best_max_1 = max_val
                best_flip_1 = n
        
        if best_flip_1 != -1:
            current_c[best_flip_1] *= -1.0
            current_f = D @ current_c
            current_max = best_max_1
            # Refine the new peak
            max_idx = np.argmax(np.abs(current_f))
            theta_max = 2 * np.pi * max_idx / M
            z_max = np.exp(1j * theta_max)
            refined_max = refine_peak(current_f, z_max, current_c)
            current_max = max(current_max, refined_max)
            
            if current_max / np.sqrt(24) < best_score:
                best_score = current_max / np.sqrt(24)
                best_c = current_c.copy()
            continue
        
        # Try 2-flip
        best_flip_2 = (-1, -1)
        best_max_2 = current_max
        for n1 in range(24):
            for n2 in range(n1 + 1, 24):
                f_new = current_f - 2.0 * (current_c[n1] * D[:, n1] + current_c[n2] * D[:, n2])
                max_val = np.max(np.abs(f_new))
                if max_val < best_max_2 - 1e-12:
                    best_max_2 = max_val
                    best_flip_2 = (n1, n2)
        
        if best_flip_2 != (-1, -1):
            n1, n2 = best_flip_2
            current_c[n1] *= -1.0
            current_c[n2] *= -1.0
            current_f = D @ current_c
            current_max = best_max_2
            # Refine
            max_idx = np.argmax(np.abs(current_f))
            theta_max = 2 * np.pi * max_idx / M
            z_max = np.exp(1j * theta_max)
            refined_max = refine_peak(current_f, z_max, current_c)
            current_max = max(current_max, refined_max)
            
            if current_max / np.sqrt(24) < best_score:
                best_score = current_max / np.sqrt(24)
                best_c = current_c.copy()
            continue
        
        # Stuck: random restart by flipping 2-3 random bits
        n_flips = rng.choice([2, 3], p=[0.6, 0.4])
        indices = rng.choice(24, n_flips, replace=False)
        current_c[indices] *= -1.0
        current_f = D @ current_c
        current_max = np.max(np.abs(current_f))
        
        # Do a quick 1-flip local search after restart
        current_c, current_f, current_max = local_search_1flip(current_c, current_f, current_max)
        
        # Refine
        max_idx = np.argmax(np.abs(current_f))
        theta_max = 2 * np.pi * max_idx / M
        z_max = np.exp(1j * theta_max)
        refined_max = refine_peak(current_f, z_max, current_c)
        current_max = max(current_max, refined_max)
        
        if current_max / np.sqrt(24) < best_score:
            best_score = current_max / np.sqrt(24)
            best_c = current_c.copy()

    return [int(x) for x in best_c]
