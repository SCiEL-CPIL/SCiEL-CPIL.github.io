import numpy as np
import time
import random

def construct():
    start_time = time.time()
    # Safety margin: The limit is 120s (or 150s in some contexts, but 110s is safe for 120s).
    # We use 115s to be safe against the 120s hard kill, leaving 5s for overhead.
    deadline = start_time + 115.0
    
    N = 24
    sqrtN = np.sqrt(N)
    
    # Evaluation Constants
    # 2^15 = 32768 points on the unit circle. 
    # This provides angular resolution of ~1e-4 rad.
    # For N=24, the polynomial oscillates slowly enough that this grid captures all major peaks.
    M = 1 << 15 
    
    # Precompute powers for Horner's method? No, Horner is O(N) per eval, which is fast.
    # Precompute indices for FFT padding?
    # coeffs -> padded -> fft -> abs -> max
    
    def get_supnorm_fast(coeffs):
        """
        Fast sup norm estimate using FFT on a dense grid.
        Returns the maximum magnitude on the grid.
        """
        # coeffs is a numpy array of shape (N,)
        # We need to compute P(z) at z = exp(2*pi*i*k/M) for k=0..M-1
        # This is equivalent to the DFT of coeffs padded to length M.
        # np.fft.fft expects input of length M.
        # P(z_j) = sum_{k=0}^{N-1} c_k (z_j)^k
        # z_j = exp(2*pi*i*j/M)
        # So P(z_j) is the j-th component of the M-point DFT of c padded with zeros.
        
        # Optimization: Instead of creating a new padded array every time,
        # we can use the fact that np.fft.fft handles the padding implicitly if we 
        # just pass the array? No, np.fft.fft pads with zeros to length n if n > len(a).
        # But creating the padded array is cheap. 
        # Actually, np.fft.fft(coeffs, M) does exactly what we need.
        
        F = np.fft.fft(coeffs, M)
        return float(np.max(np.abs(F)))

    def refine_peak(coeffs, theta0):
        """
        Golden section refinement around theta0 to find local max of |P(e^{iθ})|.
        coeffs: array of length N
        theta0: angle in radians
        """
        # Define f(theta) = |P(e^{i theta})|
        # Using Horner's method for efficiency
        # P(z) = c_0 + c_1 z + ... + c_{N-1} z^{N-1}
        
        def f(theta):
            z = np.exp(1j * theta)
            # Horner's method
            # val = c_{N-1}
            # val = val*z + c_{N-2}
            # ...
            # val = val*z + c_0
            # We can do this with a loop or np.polynomial? 
            # A simple loop in Python is slow. 
            # Let's use the dot product with powers of z.
            # powers = z ** np.arange(N)
            # val = np.dot(coeffs, powers)
            # This involves creating the powers array every time.
            # Alternative: 
            # z is scalar. 
            # coeffs is array.
            # We can precompute nothing here easily without global state.
            # Let's try the loop but unrolled? No, N=24 is small.
            # Actually, np.dot(coeffs, np.power(z, np.arange(N))) is quite fast in NumPy.
            
            # Even faster: 
            # P(z) = sum c_k z^k
            # We can compute this as:
            # z0 = 1
            # z1 = z
            # ...
            # But creating the array of powers is O(N) memory allocation.
            
            # Let's use the iterative multiplication which avoids allocation if we are careful, 
            # but in Python/NumPy, element-wise ops on scalars are still overhead.
            
            # Best bet for N=24:
            # val = 0
            # for c in reversed(coeffs):
            #    val = val * z + c
            # This is 24 complex multiplications and 24 additions.
            # In Python, this loop is ~2-3 microseconds.
            # 50 iterations * 24 ops = ~1200 ops. ~50us per refinement.
            # With 5 peaks refined, that's 250us per evaluation.
            # If we do 1000 evals/sec, that's 0.25s. Acceptable.
            
            val = 0.0 + 0.0j
            # coeffs is a numpy array. Accessing elements is slightly slow.
            # Let's convert to list? No, keep numpy.
            # Actually, iterating over numpy array is slow.
            # Let's use the dot product approach, it's vectorized.
            # powers = z ** np.arange(N)
            # val = np.dot(coeffs, powers)
            
            # Let's try to optimize the inner loop.
            # Since N is fixed and small, maybe a pure Python loop with local vars is faster than NumPy overhead?
            # Let's assume the NumPy dot product is fine for now.
            
            # Optimization: Precompute arange(N) outside?
            # We can't easily pass it in without closure.
            
            # Let's use the standard Horner in a tight loop.
            # To speed up, we can convert coeffs to a list once outside?
            # No, coeffs changes.
            
            # Let's stick to the vectorized dot product, it's robust.
            # z is a numpy complex scalar.
            # np.arange(N) creates a new array.
            # z ** np.arange(N) creates a new array.
            # np.dot is fast.
            
            # Alternative: 
            # val = coeffs[0]
            # for k in range(1, N):
            #     val = val * z + coeffs[k]
            # This is N-1 multiplications.
            
            # Let's use the loop version, it's often faster for small N in Python if we avoid NumPy overhead per step.
            # But accessing coeffs[k] is slow.
            
            # Let's use the vectorized version:
            # It's only 24 elements.
            # z_powers = z ** np.arange(N)
            # return np.abs(np.dot(coeffs, z_powers))
            
            # Wait, we can compute P(z) using np.polyval? No, that's for real roots usually or specific order.
            
            # Let's use the most efficient NumPy way:
            # coeffs * (z ** np.arange(N))
            # sum
            # This is one line.
            
            # However, creating np.arange(N) every time is wasteful.
            # We can define a global or closure variable for indices.
            
            pass # Placeholder for logic

        # Let's implement a faster f(theta) using a closure for indices
        indices = np.arange(N)
        
        def f(theta):
            z = np.exp(1j * theta)
            # z ** indices
            # coeffs * (z ** indices)
            # sum
            # This is 3 vectorized ops on length 24 arrays.
            # Should be < 1us.
            val = np.dot(coeffs, z ** indices)
            return float(np.abs(val))
        
        # Golden section search in [theta0 - delta, theta0 + delta]
        # delta should be half the grid spacing to ensure we capture the peak if it's between grid points.
        # Grid spacing is 2*pi/M.
        # Peak is within +/- pi/M of the nearest grid point? 
        # The function is smooth. The max is likely within one grid step.
        delta = 2.0 * np.pi / M
        
        a = theta0 - delta
        b = theta0 + delta
        
        # Ensure a, b are in [0, 2*pi) ? 
        # The function is periodic. Golden section doesn't care about the domain range, 
        # just the interval. But z = exp(i theta) is periodic.
        # If a < 0 or b > 2*pi, it's fine, exp handles it.
        
        gr = (np.sqrt(5) - 1) / 2  # golden ratio conjugate
        
        x1 = b - gr * (b - a)
        x2 = a + gr * (b - a)
        f1 = f(x1)
        f2 = f(x2)
        
        # 40-50 iterations should give high precision.
        # Interval shrinks by gr^k. gr^40 ~ 1e-8.
        # Initial interval 2*delta ~ 1e-4.
        # Final interval ~ 1e-12. Sufficient.
        for _ in range(45):
            if f1 < f2:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + gr * (b - a)
                f2 = f(x2)
            else:
                b = x2
                x2 = x1
                f2 = f1
                x1 = b - gr * (b - a)
                f1 = f(x1)
        
        return max(f1, f2)

    def get_supnorm_precise(coeffs):
        """
        Sup norm with peak refinement for accuracy.
        """
        # 1. Find peaks on the grid
        F = np.fft.fft(coeffs, M)
        absF = np.abs(F)
        
        # Find top K peaks
        # Using argsort is O(M log M). M=32768. ~500k ops.
        # Alternatively, use argpartition for top K.
        K = 5
        # Get indices of top K
        top_k_indices = np.argpartition(absF, -K)[-K:]
        
        max_val = 0.0
        for i in top_k_indices:
            theta = 2.0 * np.pi * i / M
            val = refine_peak(coeffs, theta)
            if val > max_val:
                max_val = val
        return max_val

    # --- Search Logic ---
    
    # Initial Solution: Golay-Rudin-Shapiro
    def get_gr_sequence(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return np.array(p[:n], dtype=float)
    
    best_coeffs = get_gr_sequence(N)
    best_score = get_supnorm_precise(best_coeffs) / sqrtN
    
    # Simulated Annealing
    random.seed(42)
    np.random.seed(42)
    
    current_coeffs = best_coeffs.copy()
    current_score = best_score
    
    # Temperature schedule
    T_start = 0.05
    T_end = 0.0001
    # We will cool based on time or iterations.
    # Since we don't know how many iterations we'll get, time-based cooling is safer.
    # But we need to finish by deadline.
    
    # We will use a fixed number of iterations if possible, or time-based.
    # Let's use time-based cooling.
    # T(t) = T_start * (T_end/T_start)^(t/T_total)
    # T_total = deadline - start_time
    
    while time.time() < deadline - 1.0: # Keep 1s buffer for final eval
        # Calculate current temperature
        elapsed = time.time() - start_time
        total_time = deadline - start_time
        progress = elapsed / total_time
        if progress >= 1.0:
            break
        T = T_start * (T_end / T_start) ** progress
        
        # Generate candidate: Flip 1 random coefficient
        # Single bit flips are better for SA in this domain.
        idx = random.randint(0, N - 1)
        new_coeffs = current_coeffs.copy()
        new_coeffs[idx] *= -1
        
        # Evaluate candidate
        # Use fast grid first to filter out obvious bad moves?
        # The grid max is a lower bound on the true max.
        # If grid_max > current_score, it might still be better if the peak is missed by grid?
        # No, grid_max <= true_max.
        # If grid_max < current_score, true_max could be higher.
        # However, for +/-1 polynomials, the peaks are sharp.
        # The grid is dense (32768 points).
        # The difference between grid_max and true_max is usually very small.
        # To be safe and fast:
        # 1. Compute grid_max.
        # 2. If grid_max < current_score * 0.95 (significant improvement), accept.
        # 3. Else, if grid_max < current_score, likely improvement, but check precise?
        #    Checking precise is expensive.
        #    
        # Strategy:
        # Compute grid_max.
        # If grid_max < current_score:
        #    It's a candidate.
        #    We need to know the precise score to compare.
        #    But precise score is expensive.
        #    
        # Alternative:
        # Always compute precise score for candidates that look promising (grid_max < current_score + small_epsilon).
        # Since N=24 is small, and we have ~100s, how many precise evals can we do?
        # One precise eval: FFT (fast) + 5 * Golden Section (45 iters * 2 evals * ~1us) = ~50us.
        # So ~20,000 precise evals per second.
        # This is very fast!
        # So we can just always use the precise score for the candidate.
        
        new_score_precise = get_supnorm_precise(new_coeffs) / sqrtN
        
        if new_score_precise < current_score:
            # Accept
            current_coeffs = new_coeffs
            current_score = new_score_precise
            if new_score_precise < best_score:
                best_coeffs = new_coeffs.copy()
                best_score = new_score_precise
        else:
            # Accept with probability exp(-delta/T)
            delta = new_score_precise - current_score
            if random.random() < np.exp(-delta / T):
                current_coeffs = new_coeffs
                current_score = new_score_precise

    # Final refinement of the best solution found
    # Ensure we return the absolute best found
    final_score = get_supnorm_precise(best_coeffs) / sqrtN
    if final_score < best_score:
        best_score = final_score

    return [int(c) for c in best_coeffs]

# For testing purposes, if run directly
if __name__ == "__main__":
    coeffs = construct()
    print(coeffs)
