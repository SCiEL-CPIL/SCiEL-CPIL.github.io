"""Rewritten program for third autocorrelation inequality optimization."""
import numpy as np
import time
import scipy.optimize

N = 400
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 5.0  # seconds before deadline to stop

def _compute_autocorr(h):
    """Compute full autocorrelation array R[0..N-1] for h of length N."""
    n_fft = 1 << (2 * N - 1).bit_length()
    f = np.zeros(n_fft, dtype=np.float64)
    f[:N] = h
    F = np.fft.rfft(f)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    return acf

def _max_sidelobe(R):
    """Return max absolute value of R[1:]."""
    return np.max(np.abs(R[1:]))

def _compute_complex_autocorr(z):
    """
    Compute autocorrelation for complex unit-modulus vector z.
    Returns the full autocorrelation array R[0..N-1] where R[k] = sum z[n]*conj(z[n+k]).
    We want to minimize |R[k]| for k >= 1.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    f = np.zeros(n_fft, dtype=np.complex128)
    f[:N] = z
    F = np.fft.rfft(f)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft)[:N]
    return acf

def _sidelobe_energy(z):
    """
    Compute sidelobe energy (sum of |R[k]|^2 for k >= 1) for complex vector z.
    This is a smooth objective on the unit-modulus manifold.
    """
    R = _compute_complex_autocorr(z)
    sidelobes = R[1:]
    return np.sum(np.abs(sidelobes)**2)

def _phase_gradient_descent(z_init, time_limit, lr=0.01, momentum=0.9):
    """
    Perform phase gradient descent on the unit-modulus manifold.
    z is a complex vector with |z[n]| = 1 for all n.
    We parameterize z[n] = exp(i*theta[n]) and optimize theta.
    
    The gradient of the sidelobe energy with respect to theta can be computed
    via the chain rule. For z = exp(i*theta), dz/dtheta = i*z.
    
    We use a momentum-based gradient descent with periodic re-normalization.
    """
    start_time = time.time()
    theta = np.angle(z_init)
    z = np.exp(1j * theta)
    
    # Velocity for momentum
    v = np.zeros(N, dtype=np.float64)
    
    best_energy = np.inf
    best_theta = theta.copy()
    
    # Precompute for gradient computation
    # Energy E = sum_{k=1}^{N-1} |R[k]|^2 where R[k] = sum_n z[n]*conj(z[n+k])
    # dE/dtheta[m] = 2 * Re[ (dE/dR) * (dR/dtheta[m]) ]
    # This is complex; let's use a simpler approach: finite differences or 
    # directly compute gradient via FFT
    
    # Alternative: use the fact that for z = exp(i*theta),
    # the gradient of sum |R[k]|^2 w.r.t. theta can be computed efficiently.
    
    # Let's use a simpler iterative approach: 
    # 1. Compute autocorrelation
    # 2. Compute gradient via conjugate gradient method
    # 3. Update theta
    
    iteration = 0
    no_improve_count = 0
    best_energy = _sidelobe_energy(z)
    best_theta = theta.copy()
    
    while time.time() < time_limit:
        if time.time() - start_time > time_limit:
            break
        
        # Compute current energy
        R = _compute_complex_autocorr(z)
        sidelobes = R[1:]
        energy = np.sum(np.abs(sidelobes)**2)
        
        if energy < best_energy - 1e-10:
            best_energy = energy
            best_theta = theta.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Compute gradient of energy w.r.t. theta
        # E = sum_{k=1}^{N-1} |R[k]|^2
        # R[k] = sum_n z[n] * conj(z[n+k])
        # dR[k]/dtheta[m] = i*z[m]*conj(z[m+k]) - z[m-k]*i*conj(z[m]) (with appropriate bounds)
        
        # More efficiently: 
        # grad_theta[m] = 2 * Re[ sum_k conj(R[k]) * dR[k]/dtheta[m] ]
        
        # dR[k]/dtheta[m]:
        # R[k] = sum_n z[n]*conj(z[n+k])
        # dR[k]/dtheta[m] = i*z[m]*conj(z[m+k]) - i*z[m-k]*conj(z[m])
        # (second term only if m-k >= 0)
        
        # Let's compute this via convolution-like operations
        # For each m, grad[m] = 2 * Re[ sum_{k=1}^{N-1} conj(R[k]) * (i*z[m]*conj(z[m+k]) - i*z[m-k]*conj(z[m])) ]
        
        # This is O(N^2) which might be slow. Let's use FFT-based approach.
        
        # Alternative simpler approach: use the relation between autocorrelation 
        # and power spectrum. 
        # |R[k]|^2 summed over k is related to the L4 norm of the spectrum.
        # E = sum_{k=1}^{N-1} |R[k]|^2 = (1/N) * sum_{f} |P(f)|^4 - |R[0]|^2
        # where P(f) is the DFT of z.
        
        # Actually, let's just use a simpler gradient: 
        # The gradient of sum |R[k]|^2 w.r.t. z[m] can be computed, then project 
        # to the tangent space of the unit-modulus manifold.
        
        # For now, let's use a heuristic update:
        # Update theta based on the autocorrelation sidelobes
        # This is inspired by the Gerchberg-Saxton / error reduction approach
        
        # Compute gradient approximation:
        # For each m, the contribution to sidelobes from z[m] is related to 
        # how z[m] correlates with other elements.
        
        # Simple approach: compute gradient via finite differences on a subset
        # Or use the following: 
        # grad_theta[m] ≈ 2 * sum_{k=1}^{N-1} Re[conj(R[k]) * i * z[m] * conj(z[m+k])]
        #                    - 2 * sum_{k=1}^{N-1} Re[conj(R[k]) * i * z[m-k] * conj(z[m])]
        
        # Let's compute this efficiently using convolution
        # Term 1: sum_k conj(R[k]) * z[m] * conj(z[m+k]) = z[m] * sum_k conj(R[k]) * conj(z[m+k])
        # Let A = conj(R) (padded to length N), B = conj(z) (padded)
        # Then sum_k A[k] * B[m+k] is a cross-correlation
        
        # This is getting complex. Let's use a simpler momentum-based update 
        # with a heuristic gradient.
        
        # Simple heuristic: adjust theta to reduce the maximum sidelobe
        # For each m, compute the "pressure" from sidelobes
        pressure = np.zeros(N, dtype=np.complex128)
        for k in range(1, N):
            # Contribution of R[k] to each position
            # R[k] = sum_n z[n]*conj(z[n+k])
            # d(R[k])/dtheta[m] = i*z[m]*conj(z[m+k]) - i*z[m-k]*conj(z[m])
            # Gradient contribution: conj(R[k]) * dR[k]/dtheta[m]
            # Sum over k: 
            # For term i*z[m]*conj(z[m+k]): sum_k conj(R[k])*conj(z[m+k]) * i*z[m]
            # For term -i*z[m-k]*conj(z[m]): -sum_k conj(R[k])*z[m-k] * i*conj(z[m])
            
            # This is O(N^2) per iteration which is 400^2 = 160000 ops, times iterations...
            # Might be too slow. Let's use FFT-based approach.
            pass
        
        # FFT-based gradient computation:
        # Let's define:
        # G1[m] = sum_{k=1}^{N-1} conj(R[k]) * conj(z[m+k])
        # G2[m] = sum_{k=1}^{N-1} conj(R[k]) * z[m-k]
        
        # G1 is a cross-correlation of conj(R) and conj(z)
        # G2 is a cross-correlation of conj(R) and z
        
        n_fft = 1 << (2 * N - 1).bit_length()
        
        # G1[m] = sum_k conj(R[k]) * conj(z[m+k])
        # This is (conj(R) * conj(z))[m] in the sense of cross-correlation
        # Using FFT: G1 = ifft(conj(fft(conj(R))) * fft(conj(z)))
        
        R_pad = np.zeros(n_fft, dtype=np.complex128)
        R_pad[:N] = R
        z_pad = np.zeros(n_fft, dtype=np.complex128)
        z_pad[:N] = z
        
        FR = np.fft.rfft(R_pad)
        Fz = np.fft.rfft(z_pad)
        
        # G1[m] = sum_k conj(R[k]) * conj(z[m+k])
        # = sum_k R_pad[k]^* * z_pad[m+k]^*
        # Cross-correlation: ifft(conj(fft(a)) * fft(b)) gives sum a[k]*b[k+m]
        # We want sum conj(R[k]) * conj(z[m+k]) = sum R[k]^* * z[m+k]^*
        # Let a = R, b = z. Then sum a[k]^* * b[m+k]^* = conj(sum a[k] * b[m+k])
        # sum a[k] * b[m+k] = ifft(fft(a) * conj(fft(b)))
        
        G1_complex = np.fft.irfft(FR * np.conj(Fz), n=n_fft)[:N]
        G1 = np.conj(G1_complex)
        
        # G2[m] = sum_k conj(R[k]) * z[m-k]
        # = sum_k R[k]^* * z[m-k]
        # Let a = R, b = z. sum a[k]^* * b[m-k]
        # Cross-correlation: ifft(conj(fft(a)) * fft(b)) gives sum a[k]^* * b[k+m]
        # We want sum a[k]^* * b[m-k] = sum a[k]^* * b[-(k-m)] 
        # This is a different correlation. Let's use:
        # sum_k a[k]^* * b[m-k] = sum_j a[m-j]^* * b[j] (substitution j = m-k)
        # = sum_j b[j] * a[m-j]^*
        # This is (b * conj(a))[m] in convolution sense
        # Using FFT: ifft(fft(b) * conj(fft(a)))
        
        G2_complex = np.fft.irfft(Fz * np.conj(FR), n=n_fft)[:N]
        G2 = G2_complex
        
        # Gradient:
        # grad_theta[m] = 2 * Re[ i*z[m]*G1[m] - i*z[m]^* * G2[m] ]
        # Wait, let me re-derive:
        # dE/dtheta[m] = 2 * Re[ sum_k conj(R[k]) * dR[k]/dtheta[m] ]
        # dR[k]/dtheta[m] = i*z[m]*conj(z[m+k]) - i*z[m-k]*conj(z[m])
        # So sum_k conj(R[k]) * dR[k]/dtheta[m] = i*z[m]*sum_k conj(R[k])*conj(z[m+k]) - i*conj(z[m])*sum_k conj(R[k])*z[m-k]
        # = i*z[m]*G1[m] - i*conj(z[m])*G2[m]
        
        grad = 2.0 * np.real(1j * z * G1 - 1j * np.conj(z) * G2)
        
        # Update with momentum
        v = momentum * v - lr * grad
        theta = theta + v
        
        # Renormalize (keep unit modulus by construction since z = exp(i*theta))
        z = np.exp(1j * theta)
        
        iteration += 1
        
        if no_improve_count > 50:
            # Reduce learning rate
            lr *= 0.5
            v = v * 0.5
            if lr < 1e-6:
                break
    
    return np.exp(1j * best_theta), best_energy

def _project_to_signed(z):
    """
    Project complex unit-modulus vector to signed real values.
    Use the real part: h[n] = sign(Re(z[n]))
    Or use a more sophisticated projection.
    """
    # Simple projection: use real part
    h = np.sign(np.real(z))
    h[h == 0] = 1.0
    return h

def _refine_with_local_search(h, R, current_max, time_limit):
    """
    Perform greedy local search starting from h, R, current_max.
    """
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        for i in range(N):
            if time.time() > time_limit:
                return h, R, current_max
            
            delta = np.zeros(N)
            
            if i > 0:
                delta[1:i+1] -= 2.0 * h[i] * h[:i][::-1]
            
            if i < N - 1:
                delta[1:N-i] -= 2.0 * h[i] * h[i+1:]
            
            new_R = R - delta
            new_max = _max_sidelobe(new_R)
            
            if new_max < current_max - 1e-12:
                h[i] *= -1.0
                R = new_R
                current_max = new_max
                improved = True
                
    return h, R, current_max

def construct():
    """
    Construct a signed step function with N intervals to minimize the
    upper bound on C3 (maximum autocorrelation sidelobe).
    
    Uses continuous phase gradient descent on the unit-modulus manifold,
    then projects to signed values and refines with local search.
    """
    start_time = time.time()
    best_h = None
    best_val = np.inf
    
    def try_phase_descent(seed, time_limit):
        """Try phase gradient descent with a given seed and time limit."""
        np.random.seed(seed)
        
        # Initialize with random phases
        theta_init = np.random.uniform(0, 2*np.pi, size=N)
        z_init = np.exp(1j * theta_init)
        
        # Run phase gradient descent
        z_opt, energy = _phase_gradient_descent(z_init, time_limit)
        
        # Project to signed values
        h = _project_to_signed(z_opt)
        
        # Compute actual metric
        R = _compute_autocorr(h)
        max_sidelobe = _max_sidelobe(R)
        
        return h, R, max_sidelobe
    
    # Try multiple random initializations with phase gradient descent
    num_restarts = 10
    time_per_restart = (DEADLINE - SAFETY_MARGIN - 5.0) / num_restarts
    
    for seed in range(num_restarts):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        remaining = DEADLINE - SAFETY_MARGIN - time.time()
        if remaining < 2.0:
            break
        
        # Allocate time for phase descent and local search
        descent_time = min(time_per_restart * 0.6, remaining * 0.5)
        local_time = remaining - descent_time - 1.0
        
        if descent_time < 0.5:
            descent_time = 0.5
            local_time = remaining - 0.5 - 1.0
        
        h_candidate, R, current_max = try_phase_descent(seed, descent_time)
        
        if current_max < best_val:
            best_val = current_max
            best_h = h_candidate.copy()
        
        # Local search polish
        if local_time > 0.5:
            h_candidate, R, current_max = _refine_with_local_search(
                h_candidate, R, current_max, time.time() + local_time
            )
            
            if current_max < best_val:
                best_val = current_max
                best_h = h_candidate.copy()
    
    return best_h if best_h is not None else np.ones(N)
