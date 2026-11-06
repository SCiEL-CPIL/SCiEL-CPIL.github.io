import numpy as np
import time
import random

def construct():
    """
    Finds the 24 coefficients in {+1, -1} minimizing the sup norm of P(z).
    Returns the list of 24 coefficients.
    """
    N = 24
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # Initial sequence: Golay-Rudin-Shapiro
    def gr_sequence(n):
        p = [1] * n
        # Recursive definition is easier to generate directly for small n
        # p_k = 1 if popcount(k) is even, -1 if odd? No, that's Rudin-Shapiro.
        # Rudin-Shapiro: r_0=1, r_1=1, r_{2n}=r_n, r_{2n+1}=(-1)^n r_n.
        # Let's use the standard Rudin-Shapiro sequence.
        r = [1, 1]
        while len(r) < n:
            new_r = []
            for x in r:
                new_r.append(x)
                new_r.append(-x)
            r = new_r
        return r[:n]

    current = np.array(gr_sequence(N), dtype=int)
    best_coeff = current.copy()
    
    # Evaluation function
    def max_norm(c):
        # c is array of 1s and -1s
        # P(z) = sum c_k z^k
        # We want max |P(e^{i theta})|
        # Sample densely
        thetas = np.linspace(0, 2*np.pi, 1000, endpoint=False)
        z = np.exp(1j * thetas)
        # Compute P(z) for all z
        # P(z) = c_0 + c_1 z + ...
        # Using FFT or direct sum? Direct sum is O(N*M). N=24, M=1000. Fast.
        # P(z_j) = sum_k c_k z_j^k
        # We can compute this as c @ Z where Z is Vandermonde?
        # Or just loop.
        vals = np.zeros(len(thetas), dtype=complex)
        for k in range(N):
            vals += c[k] * (z ** k)
        mags = np.abs(vals)
        
        # Find peaks
        # Simple refinement of top few peaks
        # Get indices of max values
        # To be robust, check local maxima in the sampled grid
        peak_indices = []
        for i in range(len(mags)):
            # Check if mags[i] is a local max
            if i == 0:
                if mags[0] > mags[1] and mags[0] > mags[-1]:
                    peak_indices.append(0)
            elif i == len(mags)-1:
                if mags[-1] > mags[-2] and mags[-1] > mags[0]:
                    peak_indices.append(i)
            else:
                if mags[i] > mags[i-1] and mags[i] > mags[i+1]:
                    peak_indices.append(i)
        
        if not peak_indices:
            peak_indices = [np.argmax(mags)]
            
        max_val = 0.0
        # Refine top 10 peaks
        # Sort by magnitude
        peak_indices.sort(key=lambda i: mags[i], reverse=True)
        for idx in peak_indices[:10]:
            theta0 = thetas[idx]
            # Golden section search for max of |P(e^{i theta})|^2
            # Interval around theta0
            delta = 2*np.pi / len(thetas)
            a = theta0 - delta
            b = theta0 + delta
            
            # Objective: max |P(e^{i theta})|^2
            def obj(theta):
                z = np.exp(1j * theta)
                p_val = 0.0
                for k in range(N):
                    p_val += c[k] * (z ** k)
                return np.abs(p_val)**2
            
            # Golden section
            gr = (np.sqrt(5) - 1) / 2
            c1 = b - gr * (b - a)
            c2 = a + gr * (b - a)
            f1 = obj(c1)
            f2 = obj(c2)
            for _ in range(50):
                if f1 < f2:
                    a = c1
                    c1 = c2
                    f1 = f2
                    c2 = a + gr * (b - a)
                    f2 = obj(c2)
                else:
                    b = c2
                    c2 = c1
                    f2 = f1
                    c1 = b - gr * (b - a)
                    f1 = obj(c1)
            max_val = max(max_val, np.sqrt(max(f1, f2)))
            
        return max_val

    current_score = max_norm(current)
    best_score = current_score
    
    # Local Search
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        best_move = None
        best_move_score = current_score
        
        # 1-flip moves
        for i in range(N):
            c_new = current.copy()
            c_new[i] *= -1
            score = max_norm(c_new)
            if score < best_move_score:
                best_move_score = score
                best_move = c_new
        
        # 2-flip moves
        for i in range(N):
            for j in range(i+1, N):
                c_new = current.copy()
                c_new[i] *= -1
                c_new[j] *= -1
                score = max_norm(c_new)
                if score < best_move_score:
                    best_move_score = score
                    best_move = c_new
                    if (time.time() - start_time) > time_limit:
                        break
            if (time.time() - start_time) > time_limit:
                break
        
        if best_move is not None:
            current = best_move
            current_score = best_move_score
            if current_score < best_score:
                best_score = current_score
                best_coeff = current.copy()
            improved = True
            
    # Final check
    return best_coeff.tolist()
