import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # 1. Precompute Chebyshev basis for smooth parameterization
    # We use a low-dimensional subspace to make the problem tractable.
    x = np.linspace(-1, 1, N)
    K_basis = 25  # Slightly higher dimension for better approximation
    basis_cheb = np.zeros((N, K_basis + 1))
    for k in range(K_basis + 1):
        basis_cheb[:, k] = np.cos(k * np.arccos(x))
    
    basis_sums = np.sum(basis_cheb, axis=0)

    def compute_value(h):
        """Exact evaluation of the objective function."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return float('inf')
        if not np.any(np.isfinite(h)):
            return float('inf')
        z = np.convolve(h, h)
        return abs(2 * N * np.max(z) / (s * s))

    def objective_coeffs(c):
        """Exact objective for Chebyshev coefficients."""
        if not np.all(np.isfinite(c)):
            return float('inf')
        h = c @ basis_cheb.T
        s = c @ basis_sums
        if abs(s) < 1e-9:
            return float('inf')
        if not np.all(np.isfinite(h)):
            return float('inf')
        z = np.convolve(h, h)
        return abs(2 * N * np.max(z) / (s * s))

    best_h = np.ones(N)
    best_val = compute_value(best_h) # 2.0

    # 2. SDP-inspired Structural Strategy
    # The goal is to minimize C such that max_k (h*h)[k] <= C.
    # This is equivalent to finding a vector h with sum(h)=1 such that its
    # autocorrelation is as flat as possible.
    #
    # In the Fourier domain, the autocorrelation is the inverse FFT of |H|^2.
    # Minimizing the peak of the inverse FFT of a PSD is related to flattening the PSD.
    # A "flat-top" PSD corresponds to a sinc-like h.
    #
    # We will construct a "Sinc-like" structure and optimize it.
    # Sinc function: h[i] = sin(pi * t) / (pi * t)
    # This has a very flat main lobe and low sidelobes in the frequency domain,
    # which translates to low peak autocorrelation in the time domain.
    
    t = np.arange(N) - N/2
    # Avoid division by zero
    t_safe = np.where(t == 0, 1e-9, t)
    h_sinc = np.sin(np.pi * t_safe) / (np.pi * t_safe)
    h_sinc /= np.sum(h_sinc)
    
    # Project to Chebyshev subspace
    B = basis_cheb
    c_sinc = np.linalg.lstsq(B, h_sinc, rcond=None)[0]
    
    # Also try a "Gaussian" start (good low sidelobes)
    sigma = N / 8.0
    h_gauss = np.exp(-t**2 / (2 * sigma**2))
    h_gauss /= np.sum(h_gauss)
    c_gauss = np.linalg.lstsq(B, h_gauss, rcond=None)[0]
    
    # Also try a "Blackman-Harris" window start
    t_norm = (np.arange(N) - N/2) / (N/2)
    h_bh = 0.35875 - 0.48829 * np.cos(np.pi * t_norm) + 0.14128 * np.cos(2*np.pi*t_norm) - 0.01168 * np.cos(3*np.pi*t_norm)
    h_bh /= np.sum(h_bh)
    c_bh = np.linalg.lstsq(B, h_bh, rcond=None)[0]
    
    starts = [c_sinc, c_gauss, c_bh]
    
    # Add a random start for diversity
    rng = np.random.default_rng(42)
    c_rand = rng.normal(0, 1, size=K_basis+1)
    c_rand[0] = 1.0 # Ensure non-zero sum roughly
    starts.append(c_rand)
    
    # 3. NEW STRATEGY: Levenshtein Sequence (Binary +/- 1)
    # Generate a binary sequence with flat autocorrelation sidelobes.
    # We use a quadratic residue modulo a prime p close to N.
    # For p = 401 (prime), the sequence length is 401. We truncate to 400.
    
    def generate_levenshtein(p):
        # Quadratic residues modulo p
        residues = set((i*i) % p for i in range(p))
        # Sequence: 1 if i is a residue, -1 otherwise (for i=1..p-1)
        # Usually defined on Z_p. Let's define for indices 0..p-1.
        # 0 is a square.
        seq = np.zeros(p)
        for i in range(p):
            if i in residues:
                seq[i] = 1.0
            else:
                seq[i] = -1.0
        return seq

    p = 401
    lev_seq = generate_levenshtein(p)
    
    # We need to map this to N=400.
    # We can try shifting the sequence and taking the first 400 elements.
    # Also, we can try flipping signs of the sequence (multiply by -1) to see if it helps
    # (though value is abs, so global sign doesn't matter, but relative sign patterns might).
    # Actually, the objective is scale invariant and uses abs(max/sum^2).
    # Sum of lev_seq is roughly 0 (since p is odd prime, #residues = (p-1)/2 + 1 = 201, #non-residues = 200).
    # Sum = 201*1 + 200*(-1) = 1.
    # So sum is small. This might be bad if sum is close to 0.
    # We can scale the sequence or add a constant? No, h must be the sequence.
    # If sum is small, value = 2N * max / sum^2 will be huge.
    # We need sum(h) to be large relative to max(h).
    # For binary +/- 1, max(h) is 1. sum(h) is at most N.
    # If sum(h) is small, value is large.
    # The Levenshtein sequence has sum 1. So value ~ 2*400*1/1 = 800. Very bad.
    # We need a sequence with large sum.
    # Maybe use a sequence that is mostly 1s with some -1s?
    # Or scale the sequence? No, "step heights" can be any real numbers.
    # The strategy says "binary +/- 1 sequence".
    # If we use h = lev_seq, sum is 1.
    # What if we use h = lev_seq + c?
    # The prompt says "Choose real, possibly negative step heights".
    # It does NOT restrict to binary.
    # But the strategy specifically suggests "binary +/- 1 sequence".
    # Perhaps we should project the Levenshtein sequence onto the Chebyshev basis
    # and optimize from there? Or just use it as a start point for the optimizer?
    # Let's try projecting it.
    
    # Truncate to N=400
    lev_N = lev_seq[:N]
    
    # Project to Chebyshev subspace
    c_lev = np.linalg.lstsq(B, lev_N, rcond=None)[0]
    starts.append(c_lev)
    
    # Also try a modified Levenshtein: 
    # If we add a large constant to make the sum large, the "flatness" relative to the sum might improve?
    # value = 2N * max(conv) / sum^2
    # If we add C to all h, sum increases by N*C.
    # max(conv) increases significantly.
    # Let's just use the projection as a start.
    
    # 4. Optimization Loop
    for c_start in starts:
        if time.time() > deadline - 5:
            break
        try:
            # Use L-BFGS-B with exact objective (robust for small dim)
            res_local = minimize(
                objective_coeffs, 
                c_start, 
                method='L-BFGS-B',
                bounds=[(0.1, 10.0)] + [(-10.0, 10.0)] * K_basis,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res_local.fun < best_val:
                h_check = res_local.x @ B.T
                if np.all(np.isfinite(h_check)) and abs(np.sum(h_check)) > 1e-9:
                    val_check = compute_value(h_check)
                    if val_check < best_val:
                        best_val = val_check
                        best_h = h_check
        except Exception:
            pass

    # Final Verification
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
        
    return best_h

if __name__ == "__main__":
    h = construct()
    print(len(h))
    print(np.sum(h))
    z = np.convolve(h, h)
    print(abs(2 * N * np.max(z) / np.sum(h)**2))
