import numpy as np
import time

N = 100
M = N // 2  # Sum of h_i must be N/2

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer

    def compute_autocorr(h):
        """Compute cyclic autocorrelation A_k = sum h_i * h_{i-k}"""
        # Using FFT for efficiency, though N=100 is small
        # FFT length should be at least 2N for linear, but we want cyclic.
        # Cyclic autocorrelation is IFFT(|FFT(h)|^2)
        H = np.fft.fft(h)
        A = np.fft.ifft(np.abs(H)**2).real
        # Due to numerical errors, A might have tiny imaginary parts or negative zeros
        # Also, the order is A[0], A[1], ..., A[N-1] corresponding to lags 0, 1, ..., N-1
        return A

    def compute_overlaps_from_autocorr(A):
        """
        O_k = M - A_k for binary h.
        For general h, O_k = sum h_i (1-h_{i-k}) = sum h_i - sum h_i h_{i-k} = M - A_k.
        This holds for any h.
        """
        return M - A

    def get_max_overlap(h):
        A = compute_autocorr(h)
        overlaps = compute_overlaps_from_autocorr(A)
        return np.max(overlaps)

    def get_min_autocorr(A):
        # We want to maximize min A_k for k != 0
        # A[0] is the energy, which is M for binary, or sum(h^2) for general.
        # The overlap at k=0 is M - A[0] = M - sum(h^2).
        # For binary, sum(h^2) = M, so overlap 0 is 0.
        # We are interested in k != 0.
        return np.min(A[1:])

    # --- Simulated Annealing on Binary Space ---
    
    best_h = None
    best_score = float('inf')
    
    def evaluate_binary(h):
        # h is 0/1 array
        A = compute_autocorr(h)
        overlaps = M - A
        max_ov = np.max(overlaps)
        return max_ov

    # Initialize
    h = np.zeros(N)
    indices = np.random.choice(N, M, replace=False)
    h[indices] = 1.0
    
    current_score = evaluate_binary(h)
    best_score = current_score
    best_h = h.copy()
    
    T = 1.0
    cooling = 0.999
    min_T = 1e-6
    steps_per_temp = 200
    
    while T > min_T:
        if time.time() > deadline - 10:
            break
            
        for _ in range(steps_per_temp):
            if time.time() > deadline - 10:
                break
                
            # Pick a random 1 and a random 0 to swap
            ones = np.where(h == 1.0)[0]
            zeros = np.where(h == 0.0)[0]
            
            if len(ones) == 0 or len(zeros) == 0:
                break
                
            i = np.random.choice(ones)
            j = np.random.choice(zeros)
            
            # Swap
            h[i] = 0.0
            h[j] = 1.0
            
            new_score = evaluate_binary(h)
            delta = new_score - current_score
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_h = h.copy()
            else:
                # Revert
                h[i] = 1.0
                h[j] = 0.0
                
        T *= cooling
        
        # Check time
        if time.time() > deadline - 15:
            break

    # --- Continuous Polish ---
    # Start from the best binary solution
    h_cont = best_h.copy()
    
    # Project to ensure sum is exactly N/2 (it is for binary, but good practice)
    # And bounds [0,1] (it is for binary)
    
    def project_to_polytope(h, S):
        """Project h onto { x : sum(x) = S, 0 <= x_i <= 1 }"""
        if S < 0 or S > N:
            return np.clip(h, 0, 1)
            
        lo = np.min(h) - 1.0
        hi = np.max(h) + 1.0
        
        for _ in range(50):
            mid = (lo + hi) / 2.0
            val = np.sum(np.clip(h - mid, 0.0, 1.0))
            if val > S:
                lo = mid
            else:
                hi = mid
                
        lambda_star = (lo + hi) / 2.0
        return np.clip(h - lambda_star, 0.0, 1.0)

    def compute_overlaps(h):
        overlaps = np.zeros(N)
        for k in range(N):
            shifted = np.roll(h, -k)
            overlaps[k] = M - np.dot(h, shifted)
        return overlaps

    def smooth_max_grad(h, beta):
        overlaps = compute_overlaps(h)
        
        max_o = np.max(overlaps)
        exps = np.exp(beta * (overlaps - max_o))
        sum_exps = np.sum(exps)
        
        weights = exps / sum_exps
        
        val = max_o + np.log(sum_exps) / beta
        
        term_sum = np.zeros(N)
        for k in range(N):
            if weights[k] > 1e-10:
                term_sum += weights[k] * (np.roll(h, -k) + np.roll(h, k))
                
        grad = 1.0 - term_sum
        return val, grad

    # Gradient Descent Polish
    if time.time() < deadline - 5:
        beta = 50.0
        alpha = 0.001
        max_steps = 200
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            val, grad = smooth_max_grad(h_cont, beta)
            actual_overlaps = compute_overlaps(h_cont)
            current_max = np.max(actual_overlaps)
            
            if current_max < best_score:
                best_score = current_max
                best_h = h_cont.copy()
                
            h_cont = h_cont - alpha * grad
            h_cont = project_to_polytope(h_cont, M)
            
            beta *= 1.02
            alpha *= 0.995
            
            if np.max(np.abs(grad)) < 1e-8:
                break

    # Final projection to ensure constraints are met
    if best_h is not None:
        best_h = project_to_polytope(best_h, M)
    else:
        best_h = np.full(N, 0.5)

    return best_h
