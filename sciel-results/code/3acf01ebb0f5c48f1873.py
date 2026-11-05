import numpy as np
import time
import sys

N = 400

def construct():
    start_time = time.time()
    # Safety margin: 10 seconds before the 120s limit
    deadline = start_time + 110.0
    
    # The objective is to minimize value = abs(2 * N * max(conv(h,h)) / sum(h)^2).
    # Due to scale invariance, fix sum(h) = 1.
    # Then value = 2 * N * max_k (sum_i h_i h_{i+k}).
    # Let C = conv(h, h). We want to minimize t such that C_k <= t for all k.
    # Note: max(C) could be negative if h is such that autocorrelation is always negative?
    # But C_0 = sum(h^2) >= 0. So max(C) >= 0.
    # Thus we minimize t >= 0 such that there exists h with sum(h)=1 and h^T T_k h <= t for all k.
    
    # This is equivalent to finding the minimum t such that the SDP is feasible:
    # Minimize t
    # Subject to:
    #   sum(h) = 1
    #   h^T T_k h <= t for all k in [0, 2N-2]
    
    # Since we cannot easily call an external SDP solver with constraints on h directly in a simple loop,
    # we can use the fact that the optimal h for a fixed t is related to the eigenvector of a matrix.
    # Alternatively, we can use a bisection search on t.
    # For a given t, is there an h with sum(h)=1 and max(h^T T_k h) <= t?
    # This is still non-convex in h because of the max.
    
    # A better approach: The problem is equivalent to minimizing the spectral radius or max eigenvalue of a certain operator?
    # Actually, consider the Toeplitz matrix A_t such that A_t = sum_{k} y_k T_k.
    # If we assume the worst-case k is determined by the structure, we can use a dual approach.
    
    # Let's try a different structural approach:
    # The function f(h) = max_k (h * h)[k] is convex.
    # We want to minimize f(h) subject to sum(h)=1.
    # This is a convex optimization problem!
    # Wait, is h^T T_k h convex? Yes, it's a quadratic form.
    # The max of convex functions is convex.
    # So we are minimizing a convex function over a linear constraint.
    # This is a Convex Optimization problem.
    # We can solve it using standard convex optimization techniques, e.g., gradient descent with a convex objective,
    # but the current program failed likely due to non-smoothness (max) and poor step size.
    # However, the prompt asks for a MATERIALLY DIFFERENT strategy.
    # The current program uses gradient descent on a smoothed max.
    # A different strategy is to use the **Dual Problem**.
    
    # The primal problem is:
    # min_h  max_k (h^T T_k h)
    # s.t. sum(h) = 1
    
    # By minimax theorem (Sion's minimax theorem applies since h^T T_k h is convex in h and linear in the distribution over k),
    # min_h max_k (h^T T_k h) = max_{y in Delta} min_h (h^T (sum_k y_k T_k) h) s.t. sum(h)=1.
    
    # Let M(y) = sum_k y_k T_k.
    # min_h (h^T M(y) h) s.t. sum(h)=1.
    # If M(y) is PSD, the minimum is 0 (achieved by h=0, but sum(h)=1 prevents this? No, if M is PSD, h^T M h >= 0. 
    # If M has a nullspace containing a vector with sum 0, we can adjust. 
    # Actually, if M is PSD, the min over sum(h)=1 is not necessarily 0.
    # The Lagrangian is h^T M h - lambda (sum(h) - 1).
    # 2 M h - lambda 1 = 0 => h = lambda/2 M^{-1} 1 (if invertible).
    # This is getting complex.
    
    # Let's use a simpler structural change: **Coordinate Descent on the Convex Objective**.
    # The objective F(h) = max_k (conv(h,h)[k]) is convex and piecewise quadratic.
    # Coordinate descent is often very effective for such problems, especially when the Hessian is sparse or structured.
    # For a fixed coordinate j, F(h) is a quadratic function of h_j (since max of quadratics is convex, and restricting to one variable keeps it convex).
    # Specifically, F(h) = max_k (Q_k(h_j)) where Q_k is quadratic in h_j.
    # The minimum of a convex piecewise quadratic function in 1D can be found efficiently.
    
    # Strategy:
    # 1. Initialize h with sum 1 (e.g., uniform).
    # 2. Iterate over coordinates j = 0..N-1.
    # 3. For each j, find the optimal h_j that minimizes F(h) while keeping other h_i fixed.
    # 4. Repeat until convergence or time limit.
    
    # This is materially different from gradient descent (which uses global gradients) and from SDP (which uses matrix variables).
    # It exploits the 1D convexity structure of the objective with respect to each variable.
    
    def compute_conv(h):
        return np.convolve(h, h)
    
    def compute_obj(h):
        c = compute_conv(h)
        return np.max(c)
    
    h = np.ones(N) / N
    current_obj = compute_obj(h)
    
    # Precompute indices for efficiency? 
    # conv(h,h)[k] = sum_i h[i] h[i+k].
    # If we change h[j], which terms in conv change?
    # h[j] appears in terms where i=j or i+k=j.
    # So conv[k] depends on h[j] if j is in [0, N-1] and k-j is in [0, N-1] OR j-k is in [0, N-1].
    # Specifically, conv[k] = sum_{i} h[i] h[i+k].
    # Term involving h[j]:
    # 1. i=j => h[j] h[j+k] (valid if 0 <= j+k < N)
    # 2. i+k=j => i=j-k => h[j-k] h[j] (valid if 0 <= j-k < N)
    # Note: if k=0, both terms are h[j]^2.
    
    # So conv[k] = (terms not involving h[j]) + h[j] * (h[j+k] + h[j-k]) (with boundary checks).
    # Let A_k = sum_{i != j, i+k != j} h[i] h[i+k].
    # Let B_k = (h[j+k] if valid else 0) + (h[j-k] if valid else 0).
    # If k=0, B_0 = 2 h[j].
    # Then conv[k] = A_k + h[j] B_k.
    
    # We want to minimize max_k (A_k + h[j] B_k).
    # This is a 1D convex minimization problem: min_x max_k (A_k + x B_k).
    # Let f(x) = max_k (A_k + x B_k).
    # f(x) is the upper envelope of lines (if B_k constant) or parabolas? 
    # Wait, A_k and B_k depend on h[j]? No, A_k and B_k are computed from the *other* elements of h.
    # So for a fixed j, f(x) is max of linear functions in x? 
    # No, conv[k] is quadratic in h overall, but for fixed other h_i, conv[k] is linear in h[j]!
    # Let's verify.
    # conv[k] = sum_i h[i] h[i+k].
    # If i=j, term is h[j] h[j+k]. Linear in h[j].
    # If i+k=j, term is h[j-k] h[j]. Linear in h[j].
    # If i!=j and i+k!=j, term does not involve h[j].
    # So yes, for fixed other coordinates, conv[k] is linear in h[j].
    # Therefore, f(x) = max_k (A_k + B_k x) is the maximum of a set of linear functions.
    # This is a convex piecewise-linear function.
    # The minimum of such a function can be found by checking the intersection points of the lines.
    # Or, more simply, since it's convex, we can use ternary search or binary search on the derivative.
    # However, the number of lines is 2N-1 (approx 800).
    # We can find the minimum by sorting the slopes B_k and intercepts A_k.
    # Actually, a simpler robust method: The minimum occurs where the derivative changes sign.
    # The derivative of max_k (A_k + B_k x) is the slope of the active line(s).
    # We can find the x where the lower envelope of the slopes crosses 0? No, it's the upper envelope.
    # The subgradient is the convex hull of {B_k | A_k + B_k x is maximal}.
    # We want 0 in the subgradient.
    
    # Efficient 1D minimization of max_k (A_k + B_k x):
    # 1. Compute A_k and B_k for all k.
    # 2. Note that f(x) is convex.
    # 3. We can use a simple bisection on the slope.
    #    slope(x) = max B_k among those k that are active? No, it's the set of active B_k.
    #    Actually, we can just sample the function? No, 800 lines.
    #    We can find the minimum by checking the vertices of the upper envelope.
    #    The vertices are intersections of pairs of lines. O(M^2) is too slow (800^2 = 640k).
    #    But we can do better.
    #    Since f(x) is convex, we can use ternary search on x? 
    #    The domain of x is not bounded, but the function grows linearly to infinity.
    #    We can find bounds.
    #    Alternatively, use the fact that the optimal x is one of the intersection points.
    #    
    # Let's use a simpler heuristic: 
    # The optimal x is likely close to the current value or 0.
    # We can perform a few steps of gradient descent on f(x) using the subgradient (which is just the B_k of the argmax).
    # Since it's 1D and convex, gradient descent with decreasing step size will converge.
    # Or even better: The minimum of max(A_k + B_k x) is the smallest t such that A_k + B_k x <= t for all k.
    # This is a linear feasibility problem in 1D.
    # B_k x <= t - A_k.
    # If B_k > 0, x <= (t - A_k) / B_k.
    # If B_k < 0, x >= (t - A_k) / B_k.
    # If B_k == 0, we need t >= A_k.
    # So for a fixed t, the feasible interval for x is [max_{B_k<0} (t-A_k)/B_k, min_{B_k>0} (t-A_k)/B_k].
    # We want to find the smallest t such that this interval is non-empty.
    # This can be solved by bisection on t.
    # Low t: interval empty. High t: interval non-empty.
    # This is O(M log M) per bisection step if we pre-sort, or O(M) per step.
    # M ~ 800. 20 bisection steps -> 16000 ops. Very fast.
    
    def minimize_1d(A, B):
        # Minimize max_k (A_k + B_k x)
        # A, B are arrays of length M.
        
        # Bisection on t.
        # Lower bound for t: max(A) (since x can be 0? No, x is variable. But min_x max_k(...) <= max_k A_k by choosing x=0? No.
        # Actually, min_x max_k (A_k + B_k x) <= max_k A_k is not guaranteed if B_k x is negative.
        # But we can set low = min(A) - 10 * max(abs(B)) * 10? 
        # Let's just find a valid t.
        # t_low = -np.inf, t_high = np.inf.
        
        # Better: 
        # The function f(x) is convex.
        # We can find the optimal x by checking the intersection of the lines with the highest slopes and lowest slopes?
        
        # Let's use the interval method.
        # We want min t such that exists x with B_k x <= t - A_k for all k.
        
        # Initialize t_high to a large value.
        # max_A = np.max(A)
        # max_B = np.max(np.abs(B))
        # t_high = max_A + 10 * max_B
        
        # Let's just use a robust bisection.
        
        # To find a valid t_high:
        # If we pick x=0, t must be >= max(A).
        # If we pick x=1, t must be >= max(A+B).
        # If we pick x=-1, t must be >= max(A-B).
        # So t_high = max(max(A), max(A+B), max(A-B)) is a valid upper bound.
        
        t_high = max(np.max(A), np.max(A+B), np.max(A-B))
        t_low = -1.0 # f(x) can be negative? Yes, if h has negative values.
        
        # Actually, min_x max_k (A_k + B_k x) can be negative.
        # Let's find a lower bound.
        # t_low = min(np.min(A), np.min(A+B), np.min(A-B)) - 1.0
        
        # Bisection for 20 iterations (enough precision for double).
        for _ in range(30):
            t_mid = (t_low + t_high) / 2.0
            
            # Check if there exists x such that B_k x <= t_mid - A_k for all k.
            # Let C_k = t_mid - A_k.
            C = t_mid - A
            
            # Constraints:
            # B_k > 0 => x <= C_k / B_k
            # B_k < 0 => x >= C_k / B_k
            # B_k == 0 => 0 <= C_k (must hold)
            
            valid = True
            
            # Handle B_k == 0
            mask_zero = np.isclose(B, 0.0)
            if np.any(mask_zero):
                if np.any(C[mask_zero] < 0):
                    valid = False
            
            if valid:
                mask_pos = B > 0
                mask_neg = B < 0
                
                x_max = np.inf
                x_min = -np.inf
                
                if np.any(mask_pos):
                    bounds_pos = C[mask_pos] / B[mask_pos]
                    x_max = np.min(bounds_pos)
                    
                if np.any(mask_neg):
                    bounds_neg = C[mask_neg] / B[mask_neg]
                    x_min = np.max(bounds_neg)
                
                if x_min <= x_max:
                    # Feasible. We can shrink t_high.
                    t_high = t_mid
                else:
                    # Infeasible. Increase t_low.
                    t_low = t_mid
            else:
                t_low = t_mid
        
        return t_high, (t_low + t_high) / 2.0 # Return optimal value and an x (midpoint of last interval? No, x is not uniquely determined, but t is. We need x to update h.
        # Actually, we need to return the optimal x.
        # The optimal x is any x in the feasible interval for t_opt.
        # Let's re-compute the interval for t_opt.
        
    def get_optimal_x(A, B, t_opt):
        C = t_opt - A
        mask_zero = np.isclose(B, 0.0)
        if np.any(mask_zero) and np.any(C[mask_zero] < 0):
            return 0.0 # Should not happen if t_opt is correct
            
        x_max = np.inf
        x_min = -np.inf
        
        mask_pos = B > 0
        mask_neg = B < 0
        
        if np.any(mask_pos):
            x_max = np.min(C[mask_pos] / B[mask_pos])
            
        if np.any(mask_neg):
            x_min = np.max(C[mask_neg] / B[mask_neg])
            
        if x_min <= x_max:
            return (x_min + x_max) / 2.0
        else:
            return 0.0 # Fallback

    # Main Loop: Coordinate Descent
    # We iterate through all coordinates multiple times.
    
    # Pre-compute structure?
    # For each j, we need A_k and B_k.
    # A_k = conv[k] - h[j]*(h[j+k] if valid + h[j-k] if valid)
    # B_k = h[j+k] if valid + h[j-k] if valid
    
    # Note: B_k depends on j.
    
    # To speed up, we can vectorize the computation of A and B for a given j.
    
    # Let's run a few passes.
    max_passes = 50
    for p in range(max_passes):
        if time.time() > deadline:
            break
            
        for j in range(N):
            if time.time() > deadline:
                break
                
            # Compute A and B for coordinate j
            # conv_full = np.convolve(h, h)
            # We need to subtract the contribution of h[j]
            
            # B_k for k=0..2N-2
            # B_k = h[j+k] (if 0<=j+k<N) + h[j-k] (if 0<=j-k<N)
            
            # Vectorized computation of B
            # Indices for h[j+k]: k such that 0 <= j+k < N => -j <= k < N-j.
            # Indices for h[j-k]: k such that 0 <= j-k < N => j-N < k <= j.
            
            # Let's just loop over k? N=400, K=800. 320k ops per j.
            # N=400. Total 128M ops per pass. Too slow in Python.
            
            # We need a faster way to compute the 1D minimization.
            # Or, we can approximate.
            
            # Alternative: Use the gradient of the max term.
            # The subgradient of max_k conv[k] is convolve(p, h) where p is the distribution on k.
            # If we take p = delta_k* (where k* is argmax), the gradient is the derivative of conv[k*].
            # This is exactly the "Subgradient on Exact Max" strategy from the current program.
            
            # Is Coordinate Descent materially different enough?
            # Yes, but the implementation is slow if not optimized.
            
            # Let's optimize the 1D minimization using the fact that B_k is sparse?
            # B_k is non-zero only for k in [-j, N-1-j] U [j-N+1, j].
            # This is a range of length ~N.
            
            # Let's try to vectorize the computation of A and B for a fixed j.
            # We can compute the full conv once per pass? No, h changes.
            
            # Let's stick to a robust, fast method:
            # **Projected Gradient Descent with Barzilai-Borwein Step Size**.
            # This is different from the current "Fixed/Decaying Alpha" approach.
            # BB step size adapts to the local curvature.
            
            # However, the prompt asks for a STRUCTURAL change.
            # Coordinate Descent is structural.
            # Let's implement a faster Coordinate Descent.
            
            # We can compute the effect of h[j] on conv[k] more efficiently.
            # conv[k] = sum_i h[i] h[i+k].
            # d(conv[k])/d(h[j]) = h[j+k] + h[j-k].
            # This is B_k.
            
            # We need to minimize max_k (conv[k] + (x - h[j]) B_k).
            # Let conv_curr = np.convolve(h, h).
            # Let delta = x - h[j].
            # New conv[k] = conv_curr[k] + delta * B_k.
            
            # We can compute B_k for all k quickly?
            # B_k = h[j+k] + h[j-k].
            # This is a sum of two shifted versions of h.
            # We can compute B for all k using padding and slicing.
            
            # But we need to do this for each j.
            
            # Let's try a different structural approach:
            # **Alternating Minimization on the Frequency Domain**.
            # The max of the autoconvolution is related to the L1 norm of the Fourier transform?
            # No, max conv is the inverse FFT of |H|^2?
            # conv(h,h) = IFFT(FFT(h) * FFT(h)).
            # We want to minimize max(IFFT(|H|^2)).
            # This is minimizing the peak of the inverse spectrum.
            # This is a "Peak Minimization" problem in the frequency domain.
            # We can optimize H directly.
            # H must be real (so H[-k] = conj(H[k])).
            # sum(h) = 1 => H[0] = 1.
            
            # Objective: min_H max_k (IFFT(|H|^2)[k]).
            
            # This is still non-convex in H?
            # |H|^2 is convex in H? No, it's quadratic.
            # IFFT is linear.
            # So we are minimizing the max of linear functions of |H|^2?
            # No, IFFT(|H|^2) is linear in |H|^2.
            # Let Y = |H|^2. Y is a PSD matrix? No, Y is a vector of power.
            # Y_k = |H_k|^2.
            # The mapping H -> Y is not linear.
            
            # But if we fix the phases of H, it's convex in amplitudes.
            
            # Let's use **Gradient Descent in Frequency Domain**.
            # h = IFFT(H).
            # conv = IFFT(H * H). (Note: H*H is not |H|^2, it's the product of complex numbers. 
            # Wait, FFT(conv(h,h)) = FFT(h) * FFT(h) = H^2.
            # So conv = IFFT(H^2).
            # We want to minimize max(IFFT(H^2)).
            
            # This is a smooth function of H (if we treat H as complex).
            # Constraint: sum(h)=1 => H[0]=1.
            # Constraint: h real => H[-k] = conj(H[k]).
            
            # We can optimize the real and imaginary parts of H[1..N/2].
            
            # This is a different representation (Frequency vs Time).
            
            # Let's implement this.
            
            break # Break out of coordinate descent loop to implement frequency domain strategy
        
    # --- Frequency Domain Strategy ---
    
    # H has length N.
    # H[0] is real and fixed to 1 (since sum(h)=1).
    # H[k] = conj(H[N-k]) for k=1..N/2-1.
    # H[N/2] is real (if N even).
    
    # Variables:
    # Re(H[1..N/2-1]), Im(H[1..N/2-1]), Re(H[N/2]).
    
    # Let's define a function to get h from H.
    
    def H_to_h(H):
        # H is complex array of length N.
        # Ensure symmetry.
        # h = IFFT(H)
        h = np.fft.ifft(H).real
        # Normalize sum(h) to 1? 
        # H[0] is fixed to 1, so sum(h) = h_mean * N?
        # sum(h) = H[0] / N * N = H[0].
        # So if H[0]=1, sum(h)=1.
        return h
    
    def compute_obj_H(H):
        # conv = IFFT(H * H)
        # Note: H*H is element-wise product.
        # Is this correct?
        # FFT(conv(h,h)) = FFT(h) * FFT(h) = H * H.
        # Yes.
        C = np.fft.ifft(H * H).real
        return np.max(C)
    
    # Initial H: Uniform h -> H[0]=1, H[k!=0]=0.
    H = np.zeros(N, dtype=np.complex128)
    H[0] = 1.0
    
    # We will optimize H using gradient descent.
    # Objective f(H) = max(IFFT(H*H)).
    # This is non-smooth due to max.
    # Use Smoothed Max (Softmax) again? 
    # The prompt says "Do not merely tune the current approach".
    # The current approach uses Softmax in Time Domain.
    # Using Softmax in Frequency Domain is a different representation.
    
    # Alternatively, use the fact that max(C) = min{t | C_k <= t for all k}.
    # This is a convex optimization in H?
    # C_k = IFFT(H*H)_k.
    # C_k is a quadratic form in H.
    # Minimizing max of quadratic forms is convex?
    # Yes, the set {H | C_k <= t} is convex?
    # C_k <= t => Re(H^T M_k H) <= t?
    # H*H is not Hermitian.
    # C_k is real.
    # This is still non-convex in H generally.
    
    # Let's stick to **Subgradient Descent in Frequency Domain** with **Adaptive Step Size**.
    
    # Gradient of max(C) w.r.t H.
    # Let k* = argmax(C).
    # C_k* = IFFT(H*H)_k* = sum_m H_m H_{k*-m} (with wraparound? No, linear conv? 
    # Wait, np.convolve is linear. FFT is circular.
    # np.convolve(h,h) is linear convolution.
    # FFT(h) * FFT(h) corresponds to circular convolution.
    # To get linear convolution via FFT, we need to zero-pad.
    # Let's use zero-padded FFT.
    
    N_fft = 2 * N
    H_padded = np.zeros(N_fft, dtype=np.complex128)
    H_padded[:N] = H
    
    def compute_obj_H_padded(H_padded):
        C = np.fft.ifft(H_padded * H_padded).real
        # The linear convolution is in the first 2N-1 elements?
        # Actually, circular conv of length 2N gives linear conv of length 2N-1 in the first 2N-1 slots.
        return np.max(C)
    
    # This is getting complicated.
    
    # Let's go back to **Coordinate Descent in Time Domain** but optimize the 1D minimization.
    # The 1D minimization of max_k (A_k + B_k x) can be done in O(M) time if we just evaluate the function?
    # No, we need the minimum.
    # But we can use the fact that the optimal x is bounded.
    # And we can use a simple ternary search on x.
    # f(x) is convex.
    # Ternary search takes O(log(range/eps)) evaluations.
    # Each evaluation is O(M).
    # M ~ 800.
    # 50 evaluations * 800 = 40k ops.
    # N=400.
    # 400 * 40k = 16M ops per pass.
    # 10 passes = 160M ops.
    # In Python, this is ~10-20s. Acceptable.
    
    h = np.ones(N) / N
    
    def eval_f_x(h, j, x):
        # Compute max conv for h with h[j] set to x.
        # We don't want to recompute full conv.
        # We have conv_curr.
        # New conv[k] = conv_curr[k] + (x - h[j]) * B_k.
        # B_k = h[j+k] + h[j-k].
        
        # We need B_k.
        # We can compute B_k on the fly?
        # Or pre-compute B for the current h?
        # B depends on h, which changes in other coordinates.
        
        # So for each j, we must compute B_k.
        # B_k = h[j+k] + h[j-k].
        # We can compute this vectorized.
        
        k = np.arange(2*N - 1)
        
        # h[j+k]
        idx1 = j + k
        mask1 = (idx1 >= 0) & (idx1 < N)
        v1 = np.zeros(2*N - 1)
        v1[mask1] = h[idx1[mask1]]
        
        # h[j-k]
        idx2 = j - k
        mask2 = (idx2 >= 0) & (idx2 < N)
        v2 = np.zeros(2*N - 1)
        v2[mask2] = h[idx2[mask2]]
        
        B = v1 + v2
        
        # A_k = conv_curr[k] - h[j] * B_k
        # conv_curr is computed from h.
        
        return A, B

    # Pre-compute conv_curr
    conv_curr = np.convolve(h, h)
    
    for p in range(20):
        if time.time() > deadline:
            break
            
        conv_curr = np.convolve(h, h)
        
        for j in range(N):
            if time.time() > deadline:
                break
                
            # Compute B
            k = np.arange(2*N - 1)
            idx1 = j + k
            mask1 = (idx1 >= 0) & (idx1 < N)
            v1 = np.zeros(2*N - 1)
            v1[mask1] = h[idx1[mask1]]
            
            idx2 = j - k
            mask2 = (idx2 >= 0) & (idx2 < N)
            v2 = np.zeros(2*N - 1)
            v2[mask2] = h[idx2[mask2]]
            
            B = v1 + v2
            
            # A = conv_curr - h[j] * B
            A = conv_curr - h[j] * B
            
            # Minimize max(A + (x - h[j]) * B)
            # Let y = x - h[j].
            # Minimize max(A + y * B).
            
            # Ternary search on y.
            # Bounds for y:
            # If B is all 0, obj is constant.
            # Else, obj grows linearly.
            # Let's find a reasonable range.
            # max_A = np.max(A)
            # max_B = np.max(np.abs(B))
            # y_low = -max_A / max_B - 1
            # y_high = max_A / max_B + 1
            
            max_B = np.max(np.abs(B))
            if max_B < 1e-12:
                continue
                
            max_A = np.max(A)
            min_A = np.min(A)
            
            # The minimum of max(A + y B) is at most max_A (at y=0).
            # It can be lower.
            # Let's search in [-1, 1] * (max_A - min_A) / max_B?
            
            scale = (max_A - min_A) / max_B + 1.0
            
            y_low = -scale
            y_high = scale
            
            # Ternary search
            for _ in range(40):
                m1 = y_low + (y_high - y_low) / 3.0
                m2 = y_high - (y_high - y_low) / 3.0
                
                f_m1 = np.max(A + m1 * B)
                f_m2 = np.max(A + m2 * B)
                
                if f_m1 < f_m2:
                    y_high = m2
                else:
                    y_low = m1
            
            y_opt = (y_low + y_high) / 2.0
            h[j] += y_opt
            
            # Update conv_curr?
            # conv_curr changes.
            # We can update it incrementally?
            # Or just recompute at the end of the pass.
            # For now, recompute at end of pass.
            
    # Final normalization
    s = np.sum(h)
    if abs(s) > 1e-12:
        h = h / s
        
    return h
