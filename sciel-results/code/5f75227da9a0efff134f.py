"""Strategy: Frequency-domain convex optimization to minimize peak sidelobe of autocorrelation."""
import numpy as np
import time
from scipy.fft import fft, ifft
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # Deadline guard: 150s limit, keep 10s margin for safety.
    deadline = start_time + 140.0
    
    # Helper to compute the objective in time domain for evaluation
    def _time_obj(x):
        n = len(x)
        size = 1
        while size < 2 * n - 1:
            size <<= 1
        X = fft(x, size)
        ac = ifft(X * np.conj(X)).real[:2 * n - 1]
        s = np.sum(x)
        if s == 0:
            return float('inf')
        # Exclude lag 0 (index N-1)
        ac_no_lag0 = np.delete(ac, N - 1)
        max_val = np.max(np.abs(ac_no_lag0))
        return max_val / (s * s)

    # Initial best: All ones
    best_x = np.ones(N, dtype=np.float64)
    best_score = _time_obj(best_x)
    
    # Convex optimization in frequency domain
    # We want to minimize max |X[k]|^2 subject to |X[0]|^2 = N^2 (DC gain fixed for sum=N)
    # Actually, for binary sequence, sum is not fixed to N, it can be anything.
    # But the objective is normalized by (sum x)^2.
    # Let's try to find a spectrum with low sidelobes relative to DC.
    # The DC term X[0] = sum(x).
    # The sidelobes are X[k] for k != 0.
    # We want to minimize max_{k!=0} |X[k]|^2 / |X[0]|^2.
    
    # Variables: log-magnitude or magnitude?
    # Let's use magnitude squared a[k] = |X[k]|^2.
    # Constraint: a[k] >= 0.
    # Objective: min max_{k!=0} a[k] / a[0].
    # This is a convex problem if we fix a[0] or normalize.
    # Let's fix a[0] = 1 (scale invariant). Minimize max_{k!=0} a[k].
    # But we need to recover a valid time-domain sequence.
    # The time-domain sequence is x[n] = (1/N) sum X[k] e^{i 2 pi k n / N}.
    # If we optimize a[k], we lose phase.
    # This is the "flat spectrum" problem.
    # A flat spectrum corresponds to a Dirac delta in time (ideal).
    # For a finite sequence, we can't have a perfect delta.
    # However, minimizing the max sidelobe in frequency is equivalent to minimizing the max sidelobe in time?
    # No, Parseval relates energy, not peak.
    # But for "almost" flat spectra, the time-domain sequence has low autocorrelation sidelobes.
    
    # Let's try to solve:
    # min t
    # s.t. a[k] <= t for k != 0
    #      a[0] = 1
    #      a[k] >= 0
    #      x = ifft(sqrt(a) * phase) is "close" to binary.
    
    # This is non-convex due to the binary constraint.
    # But we can ignore the binary constraint initially, find the optimal continuous spectrum,
    # then project to binary.
    
    # The optimal continuous spectrum for minimizing max sidelobe with fixed DC is a flat spectrum (all a[k] equal).
    # If all a[k] = 1, then x[n] is a delta (if N points) or sinc-like.
    # For a length N sequence, the "ideal" is a Barker-like sequence.
    
    # Let's try a different approach:
    # Use the fact that for a binary sequence, the autocorrelation sidelobes are bounded.
    # We will use the convex optimization to find a "target" spectrum,
    # then use gradient descent in time domain to match it, projecting to binary at each step.
    
    # Actually, let's just use the convex problem:
    # Minimize max_{k != 0} |X[k]|^2
    # Subject to |X[0]|^2 = 1 (normalized)
    # And X is the FFT of a length-N sequence.
    
    # This is equivalent to minimizing the peak sidelobe of the power spectrum.
    
    # Let's define the variables as the real and imaginary parts of X[k] for k=0..N-1.
    # Constraints:
    # 1. X[0] is real and positive (we can fix phase of DC). X[0] = 1.
    # 2. X is Hermitian symmetric? No, X is the DFT of a real sequence, so X[N-k] = conj(X[k]).
    #    So we only need to optimize k=1..N-1, and X[0] is fixed.
    #    Wait, the sequence is length N. The DFT is length N.
    #    The autocorrelation is computed via zero-padding to 2N-1.
    #    The power spectrum of the zero-padded sequence is |X_full[k]|^2.
    #    X_full[k] is related to X[k] (DFT of x) by interpolation.
    
    # This is getting complicated. Let's stick to the time-domain search but use a better initialization
    # derived from the frequency domain insight.
    
    # Insight: Sequences with flat power spectra have low autocorrelation sidelobes.
    # We can generate random phases for X[k] (k!=0), set X[0]=N, compute x = ifft(X),
    # and project to binary. Then refine with local search.
    
    def _freq_init():
        # Generate random phases
        phases = np.random.uniform(0, 2 * np.pi, N)
        # Set DC to N (sum = N)
        X = np.zeros(N, dtype=np.complex128)
        X[0] = N
        # For k != 0, set magnitude to sqrt(N) (flat spectrum approximation)
        # Actually, for a binary sequence, the energy is N.
        # So sum |X[k]|^2 = N^2.
        # If flat, |X[k]|^2 = N for all k.
        # So |X[k]| = sqrt(N).
        X[1:] = np.sqrt(N) * np.exp(1j * phases[1:])
        # Ensure Hermitian symmetry for real x?
        # The DFT of a real sequence has Hermitian symmetry: X[N-k] = conj(X[k]).
        # If we don't enforce this, x will be complex.
        # We can enforce it by constructing X carefully.
        # Or just take the real part of ifft.
        x = np.real(ifft(X))
        # Project to binary
        x_bin = np.sign(x)
        # Handle zeros
        x_bin[x_bin == 0] = 1
        return x_bin

    # Try multiple frequency-domain initializations
    num_inits = 20
    for i in range(num_inits):
        if time.time() > deadline - 10.0:
            break
            
        x = _freq_init()
        current_score = _time_obj(x)
        
        if current_score < best_score:
            best_score = current_score
            best_x = x.copy()
            
        # Local search refinement
        # Use simulated annealing or simple hill climbing
        T = 0.1
        cooling = 0.999
        steps = 0
        max_steps = 5000
        
        while time.time() < deadline - 5.0 and steps < max_steps:
            # Flip 1 bit
            idx = np.random.randint(N)
            x_new = x.copy()
            x_new[idx] *= -1
            
            new_score = _time_obj(x_new)
            
            if new_score < current_score:
                x = x_new
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
            else:
                delta_e = new_score - current_score
                if T > 1e-6:
                    prob = np.exp(-delta_e / T)
                    if np.random.random() < prob:
                        x = x_new
                        current_score = new_score
            
            T *= cooling
            steps += 1
            
            if T < 1e-5:
                break
                
    return best_x
# EVOLVE-BLOCK-END
