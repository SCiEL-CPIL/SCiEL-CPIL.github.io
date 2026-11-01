"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

DEADLINE = 150.0
TIME_MARGIN = 15.0  # safety margin in seconds

def _compute_c3_slow(x):
    """
    Compute the upper bound C3 for a signed step function x of length N using FFT.
    Used for initialization and verification.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    R0 = R[0]
    if R0 == 0:
        return 1.0
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    max_sidelobe = np.max(np.abs(sidelobes))
    return max_sidelobe / R0

class _FastAutocorr:
    def __init__(self, x):
        self.x = x.copy()
        self.R = np.zeros(N)
        self._update_R_full()
        self.current_c3 = self._calc_c3()
        self.m_range = np.arange(1, N)

    def _update_R_full(self):
        # Use FFT for initial full computation
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(self.x, n=n_fft)
        power = F * np.conj(F)
        R_full = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
        self.R[0] = R_full[0]
        self.R[1:N] = R_full[1:N]
        self.current_c3 = self._calc_c3()

    def _calc_c3(self):
        if self.R[0] == 0:
            return 1.0
        # Max absolute sidelobe
        max_sidelobe = np.max(np.abs(self.R[1:N]))
        return max_sidelobe / self.R[0]

    def _compute_delta_R(self, idx):
        """Compute the change in R[1..N-1] if x[idx] is flipped."""
        old_val = self.x[idx]
        delta_R = np.zeros(N)
        
        m_range = self.m_range
        
        # Part 1: m such that idx + m < N
        mask1 = m_range < (N - idx)
        if np.any(mask1):
            delta_R[mask1] -= 2.0 * old_val * self.x[idx + m_range[mask1]]
            
        # Part 2: m such that idx - m >= 0
        mask2 = m_range <= idx
        if np.any(mask2):
            delta_R[mask2] -= 2.0 * old_val * self.x[idx - m_range[mask2]]
            
        return delta_R

    def flip(self, idx):
        """
        Flip x[idx] and update R incrementally.
        Returns the change in C3 (new_c3 - old_c3).
        """
        old_c3 = self.current_c3
        old_x_val = self.x[idx]
        
        delta_R = self._compute_delta_R(idx)
        self.R[1:N] += delta_R[1:N]
        
        self.x[idx] = -old_x_val
        
        new_c3 = self._calc_c3()
        return new_c3 - old_c3, new_c3

    def get_best_x(self):
        return self.x.copy()

def _fast_local_search(x_init, max_iter=50000, time_limit=None):
    start_time = time.time()
    fa = _FastAutocorr(x_init)
    best_x = fa.get_best_x()
    best_c3 = fa.current_c3
    no_improve_count = 0
    max_no_improve = 500 # Increased due to speedup
    
    for iteration in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        idx = random.randint(0, N - 1)
        delta, new_c3 = fa.flip(idx)
        
        if delta < -1e-12:
            no_improve_count = 0
            if new_c3 < best_c3:
                best_x = fa.get_best_x()
                best_c3 = new_c3
        else:
            # Revert
            fa.flip(idx) # Flip back to restore state
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
    
    return best_x, best_c3

def _fast_2flip_local_search(x_init, max_iter=20000, time_limit=None):
    """
    Local search that flips two bits at a time.
    """
    start_time = time.time()
    fa = _FastAutocorr(x_init)
    best_x = fa.get_best_x()
    best_c3 = fa.current_c3
    no_improve_count = 0
    max_no_improve = 200
    
    for iteration in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        if i == j:
            continue
            
        # Flip i
        fa.flip(i)
        # Flip j
        fa.flip(j)
        
        new_c3 = fa.current_c3
        
        if new_c3 < best_c3 - 1e-12:
            no_improve_count = 0
            best_x = fa.get_best_x()
            best_c3 = new_c3
        else:
            # Revert both
            fa.flip(j)
            fa.flip(i)
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
    
    return best_x, best_c3

def _fast_simulated_annealing(x_init, max_iter=100000, T_start=0.5, T_end=0.01, time_limit=None):
    start_time = time.time()
    fa = _FastAutocorr(x_init)
    best_x = fa.get_best_x()
    best_c3 = fa.current_c3
    current_c3 = fa.current_c3
    T = T_start
    
    for iteration in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        T = T_start * (T_end / T_start) ** (iteration / max_iter)
        
        idx = random.randint(0, N - 1)
        delta, new_c3 = fa.flip(idx)
        
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            current_c3 = new_c3
            if current_c3 < best_c3:
                best_x = fa.get_best_x()
                best_c3 = current_c3
        else:
            # Revert
            fa.flip(idx)
    
    return best_x, best_c3

def _construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_x = np.ones(N)
    best_c3 = _compute_c3_slow(best_x)
    
    # Strategy 1: Random initializations with fast local search
    for trial in range(20):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        x_init = np.random.choice([-1, 1], size=N)
        time_limit = max(2.0, 10.0 - trial * 0.4)
        x_trial, c3_trial = _fast_local_search(x_init, max_iter=50000, time_limit=time_limit)
        if c3_trial < best_c3:
            best_x = x_trial
            best_c3 = c3_trial
    
    # Strategy 2: 2-Flip Local Search from best
    if time.time() < DEADLINE - TIME_MARGIN:
        x2, c3_2 = _fast_2flip_local_search(best_x, max_iter=50000, time_limit=DEADLINE - TIME_MARGIN - time.time())
        if c3_2 < best_c3:
            best_x = x2
            best_c3 = c3_2

    # Strategy 3: Simulated Annealing from best
    if time.time() < DEADLINE - TIME_MARGIN:
        x3, c3_3 = _fast_simulated_annealing(best_x, max_iter=100000, T_start=0.5, T_end=0.01,
                                             time_limit=DEADLINE - TIME_MARGIN - time.time())
        if c3_3 < best_c3:
            best_x = x3
            best_c3 = c3_3

    # Strategy 4: Final Polish (1-flip)
    if time.time() < DEADLINE - TIME_MARGIN:
        x4, c3_4 = _fast_local_search(best_x, max_iter=100000, time_limit=DEADLINE - TIME_MARGIN - time.time())
        if c3_4 < best_c3:
            best_x = x4
            best_c3 = c3_4

    # Strategy 5: Final 2-Flip Polish
    if time.time() < DEADLINE - TIME_MARGIN:
        x5, c3_5 = _fast_2flip_local_search(best_x, max_iter=50000, time_limit=DEADLINE - TIME_MARGIN - time.time())
        if c5_5 < best_c3:
            best_x = x5
            best_c3 = c3_5

    return best_x

def construct():
    x = _construct()
    return np.sign(x).astype(int)
# EVOLVE-BLOCK-END
