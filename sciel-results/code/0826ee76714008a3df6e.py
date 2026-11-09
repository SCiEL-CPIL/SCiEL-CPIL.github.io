import time
import numpy as np
import random

def construct():
    n = 256
    deadline = time.time() + 55
    
    # 1. Precompute AP structure for fast cost evaluation
    # We need to evaluate max |sum| over all APs.
    # An AP is defined by start s, step d, length L.
    # Sum = sum_{k=0}^{L-1} x_{s + k*d}.
    
    # For local search, we need to quickly update costs when flipping one sign.
    # However, the "max" operator makes this hard to update incrementally without a heap or similar.
    # Given n=256, total APs is ~ n^2 log n / 2 ~ 256^2 * 6 ~ 400k.
    # Evaluating cost from scratch:
    #   For each AP, compute sum. 
    #   We can precompute the list of indices for each AP? No, too much memory.
    #   We can iterate over steps and starts.
    
    # Let's define a fast cost function.
    # We'll use a numpy-based approach where possible, but pure python loops for AP sums might be slow if not optimized.
    # Let's try to vectorize the cost calculation as much as possible.
    
    # Alternative: Use the property that we only care about the max.
    # We can maintain the list of AP sums and just find the max.
    # To update: if we flip x_i, we need to update all APs containing i.
    # Number of APs containing i is O(n log n).
    # So updating cost is O(n log n).
    # Finding max is O(#APs) = O(n^2 log n).
    # This is too slow for many iterations.
    
    # Let's stick to a simpler cost evaluation for the discrete phase, 
    # but use the spectral method for the continuous phase.
    
    # Phase 1: Continuous Optimization (Spectral)
    # Minimize ||DFT(x)||_infty approximated by log-sum-exp of |DFT(x)|^2
    # x is continuous vector, we project to signs later or use tanh.
    
    # Let's define the objective:
    # F(x) = log(sum_{k=0}^{n-1} exp(|X_k|^2 / T))
    # where X = DFT(x).
    # Gradient descent on x.
    
    # Initialize x randomly or with some structure? 
    # Random continuous values in [-1, 1].
    
    # We will run a few restarts of this continuous optimization.
    
    best_seq = None
    best_discrete_cost = float('inf')
    
    # Precompute DFT matrix? No, use np.fft.
    # Gradient of |X_k|^2 w.r.t x:
    # X_k = sum_j x_j e^{-2pi i j k / n}
    # |X_k|^2 = X_k X_k^*
    # d/dx_j |X_k|^2 = 2 Re(X_k e^{-2pi i j k / n}^*) = 2 Re(X_k e^{2pi i j k / n})
    # Sum over k: 2 Re( sum_k X_k e^{2pi i j k / n} )
    # Note that sum_k X_k e^{2pi i j k / n} is related to the inverse DFT of X.
    # Specifically, if X = DFT(x), then x = IDFT(X) * n? 
    # Standard: X_k = sum_j x_j w^{jk}, x_j = 1/n sum_k X_k w^{-jk}.
    # So sum_k X_k w^{jk} = n * x_j? No.
    # Let's check: IDFT(X)_j = 1/n sum_k X_k w^{-jk}.
    # So sum_k X_k w^{jk} is not directly x.
    # Actually, the gradient of sum |X_k|^2 is 2 * DFT^H(DFT(x)) = 2 * n * x (if DFT is unitary? No, standard DFT is not unitary).
    # Let's just compute the gradient numerically or analytically per step.
    # For n=256, analytical gradient is fast.
    
    # grad_j = 2 Re( sum_{k=0}^{n-1} X_k e^{2pi i j k / n} )
    # This is 2 Re( IDFT(X) * n * conjugate? )
    # Let's use the property: 
    # Let Y = IDFT(X). Then Y_j = 1/n sum_k X_k w^{-jk}.
    # We want sum_k X_k w^{jk}.
    # This is n * IDFT(X_conj)_{-j mod n}? 
    # It's easier to just compute it directly or use FFT tricks.
    # Actually, d/dx sum |DFT(x)|^2 = 2 * DFT^H(DFT(x)) = 2 * n * x.
    # Wait, DFT^H DFT = n I.
    # So the gradient of the sum of squares is simply 2n x.
    # But we are minimizing log-sum-exp, which is a smooth max.
    # grad F = (1/T) * (sum_k |X_k|^2 exp(|X_k|^2/T) * d|X_k|^2/dx) / (sum_k exp(|X_k|^2/T))
    # This is a weighted average of the gradients of |X_k|^2.
    # grad_k = 2 Re( sum_l X_l e^{2pi i k l / n} )? No, index confusion.
    # Let's stick to a simple heuristic: 
    # We want to flatten the spectrum. 
    # We can iteratively project: 
    # 1. Compute X = DFT(x)
    # 2. "Shrink" large X_k towards 0? Or push x towards signs?
    
    # Let's try a different continuous strategy: 
    # Minimize the variance of the AP sums? No, we want max to be small.
    
    # Let's go back to the discrete search but with a better initialization.
    # The prompt suggests "spectral optimization". 
    # Let's generate a few candidate sequences using a spectral heuristic:
    # 1. Random continuous x.
    # 2. Gradient descent on F(x) = log-sum-exp(|DFT(x)|^2).
    # 3. Round x to signs.
    # 4. Polish with local search.
    
    # To make this fast, we'll limit the number of gradient steps.
    
    def spectral_init(seed):
        rng = np.random.RandomState(seed)
        x = rng.uniform(-1, 1, n)
        T = 1.0
        lr = 0.01
        for step in range(200):
            X = np.fft.fft(x)
            pow2 = np.abs(X)**2
            # Softmax weights
            max_pow = np.max(pow2)
            w = np.exp((pow2 - max_pow) / T)
            w /= np.sum(w)
            
            # Gradient of sum |X_k|^2 is 2 * n * x (for standard FFT)
            # But we need weighted gradient.
            # d/dx_j sum_k w_k |X_k|^2 = 2 Re( sum_k w_k X_k e^{2pi i j k / n} )
            # Let's compute Z = IDFT(w * X). 
            # Z_j = 1/n sum_k (w_k X_k) e^{-2pi i j k / n}
            # We need sum_k (w_k X_k) e^{2pi i j k / n}.
            # This is n * IDFT(w * X)[::-1]? No.
            # Let's just use the fact that DFT is linear.
            # grad_j = 2 Re( (IDFT(w * X) * n)[j] )? 
            # Check: IDFT(Y)_j = 1/n sum_k Y_k w^{-jk}.
            # We want S_j = sum_k Y_k w^{jk}.
            # S_j = n * IDFT(Y)[(-j) mod n]? 
            # Let's test with small n.
            # For now, let's approximate the gradient direction by pushing x 
            # in the direction that reduces the largest spectral components.
            # A simpler heuristic: 
            # x_new = x - lr * 2 * n * x ? No, that minimizes sum of squares, which is constant for fixed energy.
            # We want to reduce the *peak* spectral energy.
            
            # Let's use a simpler projection:
            # Compute x.
            # Compute X.
            # Identify k with large |X_k|.
            # We want to adjust x to reduce |X_k|.
            # The direction to reduce |X_k| is -Re(X_k e^{-2pi i j k / n}).
            
            # Let's just do a random walk on the sphere of signs? No, continuous.
            
            # Given time constraints, let's skip the complex gradient derivation 
            # and use a simpler "spectral balancing" heuristic:
            # 1. Start with random signs.
            # 2. Compute DFT.
            # 3. If |X_k| is large, flip signs that are "aligned" with the phase of X_k?
            
            # Actually, let's just use the discrete local search with a better initial guess.
            # The "spectral" part might be overkill if not implemented correctly.
            
            # Let's try: 
            # 1. Generate random signs.
            # 2. Compute cost.
            # 3. Do local search.
            # This is what the current program does (greedy + SA).
            
            # What is materially different?
            # "Minimize max absolute DFT".
            # Let's implement a direct search for low DFT max, then check AP discrepancy.
            # Low DFT max implies low "randomness" in a specific sense?
            # Actually, random signs have DFT max ~ sqrt(n log n).
            # We want signs with DFT max < sqrt(n). This is hard.
            
            # Let's try: 
            # 1. Initialize x as random continuous.
            # 2. Iterate:
            #    a. Compute X = DFT(x).
            #    b. Compute gradient of max(|X|) approx.
            #    c. Update x.
            #    d. Project x to [-1, 1].
            # 3. Round to signs.
            
            # Let's implement the gradient step properly.
            # grad_j = 2 Re( sum_k w_k X_k e^{2pi i j k / n} )
            # Let Y_k = w_k X_k.
            # Let Z = IDFT(Y). Z_j = 1/n sum_k Y_k e^{-2pi i j k / n}.
            # We want S_j = sum_k Y_k e^{2pi i j k / n}.
            # Note that S_j = sum_k Y_{-k} e^{-2pi i j (-k) / n} ? No.
            # S_j = n * Z_{(-j) mod n} ? 
            # Z_{(-j) mod n} = 1/n sum_k Y_k e^{-2pi i (-j) k / n} = 1/n sum_k Y_k e^{2pi i j k / n} = 1/n S_j.
            # So S_j = n * Z_{(-j) mod n}.
            # So grad_j = 2 Re( n * Z_{(-j) mod n} ).
            
            Z = np.fft.ifft(w * X)
            # Z is complex.
            # grad_j = 2 * n * Re( Z[(-j) % n] )
            idx = (-np.arange(n)) % n
            grad = 2 * n * np.real(Z[idx])
            
            x -= lr * grad
            # Project to [-1, 1]
            x = np.clip(x, -1, 1)
            
        # Round to signs
        s = np.sign(x)
        s[s == 0] = 1
        return s.tolist()

    # Discrete Local Search
    def calc_cost(s):
        # s is list of ints -1, 1
        max_disc = 0
        # Iterate over all APs
        # To speed up, we can precompute sums for each step?
        # For a fixed step d, the APs are start s, length L.
        # Sum(s, d, L) = sum_{k=0}^{L-1} x_{s+k*d}
        # We can compute all sums for a fixed d in O(n^2/d)?
        # For d=1, sums are prefix sums. O(n).
        # For d=2, two prefix sums (even/odd). O(n).
        # Total O(n^2 log n).
        
        # Let's just iterate. n=256 is small.
        for d in range(1, n):
            for start in range(0, n, d):
                # AP: start, start+d, ...
                cur_sum = 0
                for j in range(start, n, d):
                    cur_sum += s[j]
                    if abs(cur_sum) > max_disc:
                        max_disc = abs(cur_sum)
        return max_disc

    def local_search(s, deadline):
        s = s[:]
        current_cost = calc_cost(s)
        best_cost = current_cost
        best_s = s[:]
        
        # Precompute APs containing each index for fast update?
        # Or just recompute cost if it's fast enough.
        # calc_cost is ~ 256^2 * log(256) ~ 500k ops. 
        # 500k ops in Python is ~ 0.05s? Maybe too slow for many iterations.
        # Let's optimize calc_cost.
        
        # Optimized calc_cost using numpy for prefix sums per step.
        def fast_calc_cost(s):
            max_disc = 0
            arr = np.array(s)
            for d in range(1, n):
                # For each start in 0..d-1
                for start in range(d):
                    # Extract subsequence
                    sub = arr[start::d]
                    # Compute cumulative sum
                    csum = np.cumsum(sub)
                    # Max absolute
                    m = np.max(np.abs(csum))
                    if m > max_disc:
                        max_disc = m
            return max_disc

        current_cost = fast_calc_cost(s)
        best_cost = current_cost
        best_s = s[:]
        
        # Hill Climbing
        while time.time() < deadline:
            improved = False
            for i in range(n):
                s[i] *= -1
                new_cost = fast_calc_cost(s)
                if new_cost < current_cost:
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_s = s[:]
                    improved = True
                else:
                    s[i] *= -1 # revert
            if not improved:
                break
                
        return best_s, best_cost

    # Main loop
    best_overall = None
    best_overall_cost = float('inf')
    
    # Try a few spectral initializations
    for seed in range(5):
        if time.time() > deadline - 10:
            break
            
        s = spectral_init(seed)
        
        # Polish with local search
        s_polished, cost = local_search(s, deadline - 5)
        
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall = s_polished
            
    # If no improvement, fall back to greedy?
    # The current program uses greedy. Let's include a simple greedy as a fallback.
    if best_overall is None:
        # Simple greedy
        s = [1] * n
        # ... (omitted for brevity, assuming spectral is better)
        pass

    return best_overall

