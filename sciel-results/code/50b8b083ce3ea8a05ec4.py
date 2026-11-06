import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill

    # Precompute APs: all arithmetic progressions of length >= 2 within [0, N)
    ap_terms = []
    for d in range(1, N + 1):
        for s in range(0, N - d + 1):
            terms = list(range(s, N, d))
            if len(terms) > 1:
                ap_terms.append(terms)

    num_aps = len(ap_terms)

    # Build incidence matrix M (num_aps x N) as float64 for precision
    M = np.zeros((num_aps, N), dtype=np.float64)
    pos_to_aps = [[] for _ in range(N)]
    for j, terms in enumerate(ap_terms):
        for i in terms:
            M[j, i] = 1.0
            pos_to_aps[i].append(j)

    pos_to_aps_arr = [np.array(aps, dtype=np.int32) for aps in pos_to_aps]

    # C = M.T @ M
    C = (M.T @ M).astype(np.float64)
    K = np.diag(C).copy()

    def hard_max_of_signs(signs):
        sums = M @ signs
        return int(np.max(np.abs(sums)))

    best_signs = None
    best_hard_max = N + 1

    # Phase 1: DFT-based proxy optimization
    # The key insight: AP sums are related to Fourier coefficients.
    # For a sign sequence s, the sum over an AP with step d starting at a is:
    #   sum_{k=0}^{L-1} s[a + k*d]
    # This is related to the DFT of s at frequencies that are multiples of 1/d.
    # We minimize a smooth proxy: sum over low-frequency DFT coefficients of |X[f]|^2
    # weighted to emphasize periodic structures that create large AP sums.
    
    def dft_proxy_loss(x):
        """Compute a differentiable proxy for AP discrepancy using DFT."""
        X = np.fft.fft(x)
        # Focus on low frequencies where periodic structures live
        # For N=256, frequencies 1..32 capture most relevant periodicity
        freqs = np.arange(1, 33)  # frequencies 1 to 32
        X_low = X[freqs]
        # Also include negative frequencies (conjugate symmetric for real input)
        X_low_neg = X[N - freqs]
        # Weight: higher weight for lower frequencies (more "periodic" structure)
        weights = 1.0 / np.sqrt(freqs.astype(np.float64))
        loss = np.sum(weights * (np.abs(X_low)**2 + np.abs(X_low_neg)**2))
        return loss

    def dft_proxy_grad(x):
        """Gradient of the DFT proxy loss."""
        X = np.fft.fft(x)
        freqs = np.arange(1, 33)
        weights = 1.0 / np.sqrt(freqs.astype(np.float64))
        
        # d/dx of sum(w * |X[f]|^2) for real x
        # |X[f]|^2 = X[f] * conj(X[f])
        # For real x, X[N-f] = conj(X[f])
        # Gradient involves inverse FFT of weighted frequency-domain values
        
        grad_freq = np.zeros(N, dtype=np.complex128)
        grad_freq[freqs] = 2.0 * weights * X[freqs]
        grad_freq[N - freqs] = 2.0 * weights * X[N - freqs]
        
        grad = np.fft.ifft(grad_freq).real
        return grad

    # Phase 1a: Gradient descent on DFT proxy
    if time.time() < deadline - 20:
        gd_seeds = [11, 22, 33, 44, 55, 66, 77, 88]
        
        for gseed in gd_seeds:
            if time.time() > deadline - 20:
                break
                
            rng = np.random.RandomState(gseed)
            x = rng.uniform(-0.5, 0.5, size=N).astype(np.float64)
            
            # Adam optimizer
            m = np.zeros(N)
            v = np.zeros(N)
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            lr = 0.05
            
            for step in range(3000):
                if time.time() > deadline - 20:
                    break
                
                grad = dft_proxy_grad(x)
                
                # Adam update
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                m_hat = m / (1 - beta1**(step + 1))
                v_hat = v / (1 - beta2**(step + 1))
                
                x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
                
                # Soft projection: encourage values to be near +/-1
                # Use tanh to keep in [-1, 1] while maintaining differentiability
                x = np.tanh(x * 0.5)
                
                # Every 200 steps, evaluate and update best
                if step % 200 == 199:
                    signs_rounded = np.sign(x)
                    signs_rounded[signs_rounded == 0] = 1.0
                    hm = hard_max_of_signs(signs_rounded)
                    if hm < best_hard_max:
                        best_hard_max = hm
                        best_signs = signs_rounded.copy()
                    
                    # Adaptive learning rate
                    if step % 500 == 499:
                        lr *= 0.8

    # Phase 1b: Also try direct sum-of-squares minimization as backup
    if time.time() < deadline - 15:
        cont_seeds = [101, 202, 303]
        
        for cseed in cont_seeds:
            if time.time() > deadline - 15:
                break
                
            x = np.random.RandomState(cseed).uniform(-0.5, 0.5, size=N).astype(np.float64)
            
            m = np.zeros(N)
            v = np.zeros(N)
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            lr = 0.01
            
            for step in range(2000):
                if time.time() > deadline - 15:
                    break
                
                grad = 2.0 * (C @ x)
                
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                m_hat = m / (1 - beta1**(step + 1))
                v_hat = v / (1 - beta2**(step + 1))
                
                x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
                x = np.tanh(x * 0.5)
                
                if step % 200 == 199:
                    signs_rounded = np.sign(x)
                    signs_rounded[signs_rounded == 0] = 1.0
                    hm = hard_max_of_signs(signs_rounded)
                    if hm < best_hard_max:
                        best_hard_max = hm
                        best_signs = signs_rounded.copy()

    # Phase 2: Fast local search with incremental updates
    if best_signs is not None and time.time() < deadline - 5:
        signs = best_signs.copy().astype(np.float64)
        sums = M @ signs
        current_max = np.max(np.abs(sums))
        
        # Multi-pass local search: try all positions, flip if it reduces max
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for k in range(N):
                delta = -2.0 * signs[k]
                idxs = pos_to_aps_arr[k]
                new_sums = sums[idxs] + delta
                new_max = np.max(np.abs(new_sums))
                
                if new_max < current_max - 1e-9:
                    signs[k] = -signs[k]
                    sums[idxs] += delta
                    current_max = new_max
                    improved = True
            
            if not improved:
                break
            
            current_hm = int(current_max)
            if current_hm < best_hard_max:
                best_hard_max = current_hm
                best_signs = signs.copy()

    # Phase 3: SA with incremental updates for escape from local optima
    if time.time() < deadline - 3:
        seeds = [42, 123, 456, 789, 1000, 2000]
        
        for seed in seeds:
            if time.time() > deadline - 3:
                break
                
            rng = random.Random(seed)
            
            if best_signs is not None:
                signs = best_signs.copy().astype(np.float64)
                for _ in range(rng.randint(3, 8)):
                    kk = rng.randint(0, N - 1)
                    signs[kk] = -signs[kk]
            else:
                signs = np.zeros(N, dtype=np.float64)
                for i in range(N):
                    signs[i] = 1.0 if rng.random() < 0.5 else -1.0
            
            sums = M @ signs
            G = C @ signs
            current_max = np.max(np.abs(sums))
            
            current_hard_max = int(current_max)
            if current_hard_max < best_hard_max:
                best_hard_max = current_hard_max
                best_signs = signs.copy()
            
            T = 3.0
            T_min = 0.1
            T_decay = 0.999
            steps = 0
            no_improve = 0
            
            while time.time() < deadline - 3:
                k = rng.randint(0, N - 1)
                delta = -2.0 * signs[k]
                
                delta_J = 2.0 * delta * G[k] + (delta * delta) * K[k]
                
                accept = False
                if delta_J < 0:
                    accept = True
                else:
                    if delta_J / T > 80.0:
                        accept = False
                    else:
                        accept = rng.random() < math.exp(-delta_J / T)
                
                if accept:
                    signs[k] = -signs[k]
                    idxs = pos_to_aps_arr[k]
                    sums[idxs] += delta
                    G += delta * C[:, k]
                    steps += 1
                    
                    if steps % 20 == 0:
                        current_max = np.max(np.abs(sums))
                        current_hard_max = int(current_max)
                        if current_hard_max < best_hard_max:
                            best_hard_max = current_hard_max
                            best_signs = signs.copy()
                            no_improve = 0
                        else:
                            no_improve += 1
                        
                        if no_improve > 100:
                            for _ in range(rng.randint(2, 5)):
                                kk = rng.randint(0, N - 1)
                                dd = -2.0 * signs[kk]
                                signs[kk] = -signs[kk]
                                sums[pos_to_aps_arr[kk]] += dd
                                G += dd * C[:, kk]
                            no_improve = 0
                
                if steps % 300 == 0:
                    T *= T_decay
                    if T < T_min:
                        T = T_min
                
                if best_hard_max <= 1:
                    break

    # Final polish
    if best_signs is not None and time.time() < deadline - 1:
        signs = best_signs.copy().astype(np.float64)
        sums = M @ signs
        current_max = np.max(np.abs(sums))
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for k in range(N):
                delta = -2.0 * signs[k]
                idxs = pos_to_aps_arr[k]
                new_sums = sums[idxs] + delta
                new_max = np.max(np.abs(new_sums))
                if new_max < current_max - 1e-9:
                    signs[k] = -signs[k]
                    sums[idxs] += delta
                    current_max = new_max
                    improved = True
        
        current_hm = int(current_max)
        if current_hm < best_hard_max:
            best_hard_max = current_hm
            best_signs = signs.copy()

    if best_signs is None:
        return [1 if random.random() < 0.5 else -1 for _ in range(N)]

    return [int(s) for s in best_signs]
