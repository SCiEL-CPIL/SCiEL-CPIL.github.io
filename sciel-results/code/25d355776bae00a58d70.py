"""Minimize peak autoconvolution of L2-normalized nonnegative step function."""
import numpy as np
import time

N = 600
DELTA = 0.5 / N
DELTA2 = DELTA * DELTA
TARGET_SQ = 1.0 / DELTA

def construct():
    t0 = time.time()
    deadline = t0 + 138.0

    def autoconv(h):
        nfft = 2 * N
        hp = np.zeros(nfft)
        hp[:N] = h
        F = np.fft.rfft(hp)
        ac = np.fft.irfft(F * F.conj(), n=nfft)
        return ac[:N] * DELTA2

    def peak(h):
        return float(np.max(autoconv(h)))

    # Initialize
    h = np.full(N, np.sqrt(TARGET_SQ / N))
    best_h = h.copy()
    best_s = peak(h)

    tau = 0.2
    lr = 0.1
    it = 0
    no_improve = 0

    while time.time() < deadline:
        it += 1
        ac = autoconv(h)
        m = np.max(ac)
        w = np.exp((ac - m) / tau)
        w /= w.sum()

        # Gradient via convolution: grad = DELTA2 * (w * h + w_rev * h) where * is linear conv, take first N
        # w * h means sum_k w[k] * h[j+k] -> this is cross-correlation
        # Use FFT for efficiency
        nfft = 2 * N
        hp = np.zeros(nfft); hp[:N] = h
        wp = np.zeros(nfft); wp[:N] = w
        Fh = np.fft.rfft(hp)
        Fw = np.fft.rfft(wp)
        # sum_k w[k] * h[j+k] = (w reversed convolved with h)[j]
        # Let's compute: corr1[j] = sum_k w[k] * h[j+k] for valid j
        # This equals (w_flip * h)[j] where w_flip[k] = w[N-1-k]... 
        # Simpler: use np.convolve
        # corr1 = sum_k w[k]*h[j+k]: for j=0, sum_k w[k]*h[k] = dot(w, h)
        # This is like cross-correlation. Let's just use a loop over top-k weights.
        
        # Efficient: only top weights matter
        top_k = min(50, N)
        idx = np.argsort(w)[::-1][:top_k]
        grad = np.zeros(N)
        for k in idx:
            wk = w[k]
            if wk < 1e-8:
                break
            c = wk * DELTA2
            # h[j+k] term
            if k < N:
                grad[:N-k] += c * h[k:]
            # h[j-k] term  
            if k > 0:
                grad[k:] += c * h[:N-k]
            else:
                grad[:] += c * h[:]

        h_new = h - lr * grad
        h_new = np.maximum(h_new, 0.0)
        nsq = np.sum(h_new * h_new)
        if nsq < 1e-30:
            h_new = h.copy()
        else:
            h_new *= np.sqrt(TARGET_SQ / nsq)

        s = peak(h_new)
        if s < best_s - 1e-13:
            best_s = s
            best_h = h_new.copy()
            no_improve = 0
        else:
            no_improve += 1

        h = h_new

        if it % 10 == 0:
            lr *= 0.95
            tau *= 0.97
            if lr < 1e-6:
                lr = 1e-6

        if no_improve > 100:
            break

    if not np.all(np.isfinite(best_h)):
        return [1.0] * N
    return [float(max(0.0, x)) for x in best_h]
