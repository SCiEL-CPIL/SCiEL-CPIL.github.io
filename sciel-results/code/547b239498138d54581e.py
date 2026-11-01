import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin

    def get_max_sidelobe(seq):
        """
        Compute the upper bound C3: max absolute sidelobe of autocorrelation.
        For a signed step function with values in {-1, 1}, the autocorrelation
        R[k] = sum_{i} seq[i] * seq[i+k].
        The score is the maximum |R[k]| for k != 0.
        """
        # Use FFT for efficient computation
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(seq, n=n_fft)
        power = F * np.conj(F)
        R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
        
        # R[0] is zero-lag (peak), R[1:N] and R[N:2N-1] are sidelobes
        # Due to floating point, take real part
        R = R.real
        
        side1 = R[1:N]
        side2 = R[N:2*N-1]
        max_sidelobe = np.max(np.abs(np.concatenate([side1, side2])))
        return max_sidelobe

    def _thue_morse(n):
        """Generate Thue-Morse sequence (balanced, low autocorrelation)."""
        seq = np.zeros(n)
        for i in range(n):
            c = 0
            m = i
            while m:
                c += m & 1
                m >>= 1
            seq[i] = 1.0 if c % 2 == 0 else -1.0
        return seq

    def _logistic_sequence(n, r=3.99, x0=0.5):
        """Generate a chaotic logistic map sequence and binarize it."""
        x = np.zeros(n)
        val = x0
        for i in range(n):
            val = r * val * (1.0 - val)
            x[i] = 1.0 if val > 0.5 else -1.0
        return x

    def _van_der_corput(n, base=2):
        """Generate a van der Corput low-discrepancy sequence and binarize it."""
        seq = np.zeros(n)
        for i in range(1, n + 1):
            v = 0.0
            f = 1.0 / base
            m = i
            while m > 0:
                m, r = divmod(m, base)
                v += r * f
                f /= base
            seq[i-1] = 1.0 if v > 0.5 else -1.0
        return seq

    def _de_bruijn_like(n, order=4):
        """
        Generate a sequence based on de Bruijn torus principles.
        For binary alphabet, a de Bruijn sequence of order k contains every
        possible k-bit subsequence exactly once in a cycle.
        We generate a long de Bruijn sequence and take the first n bits.
        """
        if order == 2:
            # Simple alternating
            seq = np.zeros(n)
            for i in range(n):
                seq[i] = 1.0 if i % 2 == 0 else -1.0
            return seq
        
        # Generate de Bruijn sequence of given order
        # Using the standard algorithm
        a = [0] * (1 << order)
        seq_list = []
        def db(t, p):
            if t > 1 << order:
                if t <= p:
                    for i in range(p):
                        seq_list.append(a[i])
            else:
                a[t] = a[t - p]
                db(t + 1, p)
                for j in range(a[t - p] + 1, 2):
                    a[t] = j
                    db(t + 1, t)
        
        db(1, 1)
        # seq_list contains the de Bruijn sequence (0s and 1s)
        # Map to {-1, 1}
        seq = np.zeros(n)
        for i in range(n):
            # Use modular indexing to cycle through the de Bruijn sequence
            val = seq_list[i % len(seq_list)]
            seq[i] = 1.0 if val == 0 else -1.0
        return seq

    best_h = None
    best_score = np.inf

    # Generate diverse initializations
    initializations = []
    
    # 1. Thue-Morse
    initializations.append(_thue_morse(N))
    
    # 2. Logistic maps with different parameters
    for r in [3.9, 3.99, 3.999]:
        for x0 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            initializations.append(_logistic_sequence(N, r=r, x0=x0))
    
    # 3. Van der Corput
    for base in [2, 3, 5]:
        initializations.append(_van_der_corput(N, base=base))
    
    # 4. De Bruijn-like sequences
    for order in [3, 4, 5, 6]:
        try:
            initializations.append(_de_bruijn_like(N, order=order))
        except:
            pass
    
    # 5. Random initializations
    rng = np.random.RandomState(42)
    for _ in range(10):
        initializations.append(rng.choice([-1, 1], size=N))

    # Local search function
    def local_search(x, max_iter=5000, time_limit=10.0):
        start = time.time()
        current_score = get_max_sidelobe(x)
        best_x = x.copy()
        best_s = current_score
        no_improve = 0
        max_no_improve = 50
        
        for it in range(max_iter):
            if time.time() - start > time_limit:
                break
            if time.time() > deadline:
                break
            
            # Try flipping a random position
            idx = np.random.randint(N)
            x_new = x.copy()
            x_new[idx] = -x_new[idx]
            new_s = get_max_sidelobe(x_new)
            
            if new_s < current_s - 1e-12:
                x = x_new
                current_s = new_s
                no_improve = 0
                if current_s < best_s:
                    best_x = x.copy()
                    best_s = current_s
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
        
        return best_x, best_s

    # Evaluate each initialization with local search
    for i, x_init in enumerate(initializations):
        if time.time() > deadline - 5.0:
            break
        
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        
        time_limit = min(10.0, remaining / (len(initializations) - i + 1))
        x_trial, s_trial = local_search(x_init.copy(), max_iter=3000, time_limit=time_limit)
        
        if s_trial < best_score:
            best_score = s_trial
            best_h = x_trial.copy()

    if best_h is None:
        best_h = np.ones(N)

    return best_h
