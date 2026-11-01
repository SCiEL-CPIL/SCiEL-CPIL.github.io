import numpy as np
import time

N = 400

def construct():
       start_time = time.time()
       deadline = start_time + 110.0 # 10s margin

       best_h = None
       best_score = np.inf

       # Initialize with random binary sequence
       h = np.random.choice([-1, 1], N)

       # Helper to compute max sidelobe
       def get_max_sidelobe(seq):
           # Autocorrelation
           # np.correlate is O(N^2) in pure python/numpy if not optimized?
           # Actually np.correlate is efficient.
           # Or use FFT:
           # F = np.fft.fft(seq, 2*N)
           # R = np.fft.ifft(F * np.conj(F)).real
           # R is circular autocorrelation. We need linear.
           # Zero padding to 2*N ensures linear autocorrelation in the first N points?
           # Let's use np.correlate for simplicity and correctness, N=400 is small.
           R = np.correlate(seq, seq, mode='full')
           # R has length 2N-1. Center is N-1.
           # Sidelobes are R[0:N-1] and R[N:2N-1]
           # Exclude R[N-1] (peak)
           side1 = R[:N-1]
           side2 = R[N:]
           return np.max(np.abs(np.concatenate([side1, side2])))

       current_max = get_max_sidelobe(h)
       if current_max < best_score:
           best_score = current_max
           best_h = h.copy()

       while time.time() < deadline:
           # Try a random flip
           idx = np.random.randint(N)
           old_val = h[idx]
           h[idx] = -old_val

           new_max = get_max_sidelobe(h)

           if new_max < current_max:
               current_max = new_max
               if new_max < best_score:
                   best_score = new_max
                   best_h = h.copy()
           else:
               # Revert if not improved? Or keep if simulated annealing?
               # For now, greedy: revert if not improved to stay in basin.
               # But greedy can get stuck.
               # Let's do: accept if better, else revert.
               h[idx] = old_val

       return best_h
