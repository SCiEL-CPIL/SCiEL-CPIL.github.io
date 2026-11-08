import time
import numpy as np

N = 100
DEADLINE = 140.0  # safety margin under 150s

def _overlap_from_power(power):
    """Given power spectrum (length N, power[0] is DC), return overlap at each circular shift.
    overlap[k] = (1/N) * sum_j h_j * h_{j+k} (circular)."""
    # autocorrelation via FFT: autocorr = ifft(|fft(h)|^2)
    # We have power = |fft(h)|^2 / N  (so that autocorr[k] = (1/N) sum power[k] e^{i 2pi k m / N})
    # Actually: autocorr[m] = (1/N) * IDFT(power)[m] where IDFT uses 1/sqrt(N) convention... let's be careful.
    # numpy fft: H = fft(h). |H|^2. ifft(|H|^2)[m] = (1/N) sum_k |H_k|^2 e^{i 2pi k m/N} = sum_j h_j h_{j+m} (circular, unnormalized)
    # So autocorr[m] = ifft(power_full)[m] where power_full = |H|^2.
    # We want overlap[m] = (1/N) * autocorr[m] = (1/N) * ifft(power_full)[m].
    # Let p = power_full / N^2 ... let's just compute directly.
    # Define: we store P[k] = |H_k|^2 / N. Then autocorr[m] = (1/N) sum_k |H_k|^2 e^{i2pi k m/N} = sum_k P[k] e^{i2pi k m/N} = N * ifft(P)[m] ... 
    # ifft(P)[m] = (1/N) sum_k P[k] e^{i2pi k m/N} = (1/N) autocorr[m].
    # So autocorr[m] = N * ifft(P)[m], where P = |H|^2/N.
    # overlap[m] = (1/N) autocorr[m] = ifft(P)[m].
    # So overlap = ifft(P).real where P = |H|^2 / N.
    return np.real(np.fft.ifft(power))

def _h_from_power(power):
    """Reconstruct h from power spectrum. h has mean 0.5, so DC component fixed.
    power[0] = N * 0.5^2 = N/4 (since |H_0|^2 = (sum h)^2 = (N/2)^2 = N^2/4, so |H_0|^2/N = N/4).
    For k>=1, h_k = ifft contribution. We set H_k = sqrt(N * power[k]) * e^{i*0} (zero phase) for reconstruction.
    Actually: H_k = |H_k| e^{i phi_k} = sqrt(N * power[k]) * 1 (zero phase).
    Then h = ifft(H). We need h in [0,1] and sum = N/2."""
    H = np.zeros(N, dtype=complex)
    H[0] = N / 2.0  # sum h = N/2
    for k in range(1, N):
        H[k] = np.sqrt(N * power[k])
        # zero phase; symmetry handled by conjugate pairs automatically since power is real & symmetric
    h = np.real(np.fft.ifft(H))
    return h

def _max_overlap(h):
    """Maximum overlap of h with shifted (1-h)."""
    # overlap at shift m: (1/N) sum_j h_j (1 - h_{j+m}) = 0.5 - (1/N) sum_j h_j h_{j+m}
    # We want to minimize max_m overlap_m = 0.5 - min_m autocorr_m
    # So minimize max overlap = 0.5 - min autocorr.
    # autocorr[m] = (1/N) sum_j h_j h_{j+m}
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(np.abs(H)**2)) / N
    return 0.5 - np.min(autocorr)

def construct():
    start = time.time()
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)  # should be 0.5 - 0.25 = 0.25? Let's check: autocorr[0]=mean(h^2)=0.25, min autocorr for constant is 0.25, so overlap=0.25.
    # Wait, for constant h=0.5: autocorr[m] = 0.25 for all m. overlap[m] = 0.5 - 0.25 = 0.25. Max overlap = 0.25.
    
    # Optimize power spectrum
    # power[0] is fixed at N/4. Optimize power[1:].
    # Constraints: power[k] >= 0, and sum_k power[k] = sum h^2 (total power).
    # Total power = sum_j h_j^2. For h in [0,1] with mean 0.5, sum h^2 is between N/4 (constant) and N/2 (binary).
    # We'll optimize power[1:] >= 0 with a soft constraint on total power, and reconstruct.
    
    # Initialize power: start with a low-frequency bias
    np.random.seed(42)
    
    def _init_power(freq_bias):
        # power[0] = N/4
        p = np.zeros(N)
        p[0] = N / 4.0
        # For k>=1, initialize with decay
        for k in range(1, N):
            # symmetric: power[k] = power[N-k]
            p[k] = freq_bias * (1.0 / (1.0 + 0.1 * k))
        # ensure symmetry
        for k in range(1, N//2 + 1):
            p[N-k] = p[k]
        return p
    
    def _project_power(p, total_power_target):
        """Project p onto p[0]=N/4, p[k]>=0 for k>=1, and sum p = total_power_target."""
        p = p.copy()
        p[0] = N / 4.0
        p[1:] = np.maximum(p[1:], 0.0)
        # enforce symmetry
        for k in range(1, N//2 + 1):
            avg = 0.5 * (p[k] + p[N-k])
            p[k] = avg
            p[N-k] = avg
        # adjust to match total power
        current = np.sum(p)
        if current > 0:
            p[1:] *= (total_power_target - N/4.0) / (current - N/4.0)
            p[1:] = np.maximum(p[1:], 0.0)
        return p
    
    # Multi-start over different total power targets and frequency biases
    total_power_targets = [N/4 + 0.01*N, N/4 + 0.05*N, N/4 + 0.1*N, N/4 + 0.2*N]
    freq_biases = [0.01, 0.05, 0.1, 0.2]
    
    best_power = None
    best_p_score = best_score
    
    for tp in total_power_targets:
        for fb in freq_biases:
            if time.time() - start > DEADLINE - 5:
                break
            p = _init_power(fb)
            p = _project_power(p, tp)
            
            # Projected gradient descent on power spectrum
            lr = 0.01
            for it in range(2000):
                if time.time() - start > DEADLINE - 5:
                    break
                h = _h_from_power(p)
                # clip to [0,1] and renormalize
                h_clipped = np.clip(h, 0, 1)
                # renormalize to sum = N/2
                s = np.sum(h_clipped)
                if s > 0:
                    h_clipped *= (N/2.0) / s
                    h_clipped = np.clip(h_clipped, 0, 1)
                
                score = _max_overlap(h_clipped)
                if score < best_p_score:
                    best_p_score = score
                    best_power = p.copy()
                    best_h = h_clipped.copy()
                
                # Gradient: we want to minimize max overlap. Use subgradient.
                H = np.fft.fft(h_clipped)
                autocorr = np.real(np.fft.ifft(np.abs(H)**2)) / N
                overlap = 0.5 - autocorr
                # max overlap is at argmax(overlap) = argmin(autocorr)
                m_star = np.argmin(autocorr)
                # d(overlap_m)/dpower_k = d(autocorr_m)/dpower_k * (-1) ... 
                # autocorr[m] = ifft(power)[m] = (1/N) sum_k power[k] e^{i2pi k m/N}
                # d autocorr[m] / d power[k] = (1/N) e^{i2pi k m/N}
                # d overlap[m] / d power[k] = -(1/N) e^{i2pi k m/N}
                # For real power, use real part: d overlap[m] / d power[k] = -(1/N) cos(2pi k m/N)
                # We want to decrease overlap[m_star], so increase autocorr[m_star].
                # Gradient of -autocorr[m_star] w.r.t. power[k] = -(1/N) cos(2pi k m/N)
                # So gradient of overlap[m_star] = (1/N) cos(2pi k m/N) ... wait sign.
                # overlap = 0.5 - autocorr, d overlap/d power = -d autocorr/d power = -(1/N) e^{i...}
                # To minimize overlap, we want to move power in direction -grad(overlap) = (1/N) e^{i...}
                # For real: direction = (1/N) cos(2pi k m/N)
                
                k_vals = np.arange(N)
                grad = -(1.0/N) * np.cos(2 * np.pi * k_vals * m_star / N)
                # We want to decrease overlap, so update p -= lr * grad = p += lr * (1/N) cos(...)
                # But we need to be careful: this is only the gradient at the active constraint.
                # Use a smoothed version: weight all overlaps
                weights = np.exp(overlap / 0.01)  # soft-max weighting
                weights /= np.sum(weights)
                grad_smooth = np.zeros(N)
                for m in range(N):
                    grad_smooth += weights[m] * (-(1.0/N) * np.cos(2 * np.pi * k_vals * m / N))
                
                p_new = p - lr * grad_smooth
                p_new = _project_power(p_new, tp)
                p = p_new
            
            # After GD, try coordinate descent / local refinement
            for _ in range(50):
                if time.time() - start > DEADLINE - 5:
                    break
                h = _h_from_power(p)
                h_clipped = np.clip(h, 0, 1)
                s = np.sum(h_clipped)
                if s > 0:
                    h_clipped *= (N/2.0) / s
                    h_clipped = np.clip(h_clipped, 0, 1)
                score = _max_overlap(h_clipped)
                if score < best_p_score:
                    best_p_score = score
                    best_power = p.copy()
                    best_h = h_clipped.copy()
                
                # Try perturbing one power coefficient
                improved = False
                for k_try in range(1, N//2 + 1):
                    for delta in [0.001, -0.001, 0.01, -0.01]:
                        p_test = p.copy()
                        p_test[k_try] += delta
                        p_test[N-k_try] += delta
                        p_test = _project_power(p_test, tp)
                        h_test = _h_from_power(p_test)
                        h_test = np.clip(h_test, 0, 1)
                        s_test = np.sum(h_test)
                        if s_test > 0:
                            h_test *= (N/2.0) / s_test
                            h_test = np.clip(h_test, 0, 1)
                        score_test = _max_overlap(h_test)
                        if score_test < best_p_score - 1e-6:
                            p = p_test
                            best_p_score = score_test
                            best_power = p.copy()
                            best_h = h_test.copy()
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
    
    # Final: ensure best_h is valid
    if best_h is not None:
        h_final = np.clip(best_h, 0, 1)
        s = np.sum(h_final)
        if s > 0:
            h_final *= (N/2.0) / s
            h_final = np.clip(h_final, 0, 1)
        # Ensure sum is exactly N/2
        diff = N/2.0 - np.sum(h_final)
        # distribute diff to the entry with most room
        if abs(diff) > 1e-10:
            # add to the smallest entry (most room to grow) or subtract from largest
            if diff > 0:
                idx = np.argmin(h_final)
                h_final[idx] = min(1.0, h_final[idx] + diff)
                # if still off, distribute
                remaining = N/2.0 - np.sum(h_final)
                if abs(remaining) > 1e-10:
                    mask = h_final < 1.0
                    if np.any(mask):
                        room = 1.0 - h_final[mask]
                        total_room = np.sum(room)
                        if total_room > 0:
                            h_final[mask] += remaining * room / total_room
            else:
                idx = np.argmax(h_final)
                h_final[idx] = max(0.0, h_final[idx] + diff)
                remaining = N/2.0 - np.sum(h_final)
                if abs(remaining) > 1e-10:
                    mask = h_final > 0.0
                    if np.any(mask):
                        total = np.sum(h_final[mask])
                        if total > 0:
                            h_final[mask] += remaining * h_final[mask] / total
        # Final check
        if not (np.all(h_final >= -1e-9) and np.all(h_final <= 1.0 + 1e-9) and abs(np.sum(h_final) - N/2.0) < 1e-6):
            return np.full(N, 0.5)
        return h_final
    return np.full(N, 0.5)
