import numpy as np
import time

N = 400
DEADLINE = 120.0

def _compute_autocorr(signed_step):
    """
    Compute the autocorrelation of the signed step function.
    Returns the autocorrelation array for lags -N+1 to N-1 (centered).
    """
    # signed_step is length N, values -1 or 1
    # Domain is [-1/4, 1/4], length 0.5.
    # Delta x = 0.5 / N
    dx = 0.5 / N
    
    # Use FFT for efficient convolution
    # f * f (cross-correlation is same as autocorrelation for real signals, 
    # but we need R(tau) = int f(s) f(s+tau) ds.
    # This is correlation, not convolution.
    # Correlation R(τ) = (f ⋆ f)(τ).
    # In FFT, correlation is IFFT(FFT(f) * conj(FFT(f))).
    
    size = 2 * N
    f_padded = np.zeros(size)
    f_padded[:N] = signed_step
    
    F = np.fft.fft(f_padded, n=size)
    R = np.fft.ifft(F * np.conj(F)).real
    
    # R is circular correlation.
    # We want linear correlation.
    # The linear correlation lags are 0, 1, ..., N-1 and -(N-1), ..., -1.
    # In the FFT result of size 2N, the lags are 0, 1, ..., N-1, -N, ..., -1.
    # Specifically, R[k] corresponds to lag k for k < N, and lag k-N for k >= N.
    
    # Scale by dx
    R = R * dx
    
    return R

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound.
    Assumption: C3 is the Peak Sidelobe Level (PSL), i.e., max |R(tau)| for tau != 0.
    """
    R = _compute_autocorr(signed_step)
    
    # R[0] is the peak (lag 0).
    # R[1:N] are positive lags 1 to N-1.
    # R[N:2N-1] are negative lags -N to -1.
    # We exclude lag 0 (index 0).
    
    # Extract sidelobes
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    
    if len(sidelobes) == 0:
        return 0.0
        
    psl = np.max(np.abs(sidelobes))
    return psl

def _compute_sidelobe_energy(signed_step):
    """
    Compute the sum of squares of the sidelobes.
    This is a smooth surrogate for PSL.
    """
    R = _compute_autocorr(signed_step)
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    return np.sum(sidelobes ** 2)

def construct():
    start_time = time.time()
    
    # 1. Random Initialization
    # Try multiple random restarts and keep the best
    best_step = None
    best_psl = float('inf')
    best_energy = float('inf')
    
    # Number of restarts
    num_restarts = 20
    iters_per_restart = 5000
    
    # Simulated Annealing parameters
    T_start = 0.1
    T_end = 0.001
    alpha = (T_end / T_start) ** (1.0 / max(iters_per_restart, 1))
    
    for r in range(num_restarts):
        # Check time
        if time.time() - start_time > DEADLINE - 10.0:
            break
            
        # Initialize with random signs
        step = np.random.choice([-1, 1], size=N).astype(np.int8)
        
        # Calculate initial metrics
        current_psl = _compute_c3(step)
        current_energy = _compute_sidelobe_energy(step)
        
        # Simulated Annealing
        T = T_start
        for i in range(iters_per_restart):
            if time.time() - start_time > DEADLINE - 5.0:
                break
                
            # Flip a random bit
            idx = np.random.randint(N)
            old_val = step[idx]
            step[idx] = -old_val
            
            # Calculate new metrics
            # For speed, we can approximate the change in energy, 
            # but for PSL we might need full recalc or approximation.
            # Given N=400, FFT is fast (~1ms). 5000 iters * 20 restarts = 100k iters.
            # 100k * 1ms = 100s. This is tight.
            # Let's optimize: 
            # 1. Use fewer restarts but more iters?
            # 2. Use a faster metric update?
            
            # Let's try 10 restarts, 10000 iters.
            # Or just run until time limit.
            
            new_psl = _compute_c3(step)
            new_energy = _compute_sidelobe_energy(step)
            
            # Acceptance criterion
            # We want to minimize PSL.
            # Use energy as a guide for acceptance to smooth the landscape.
            delta_energy = new_energy - current_energy
            delta_psl = new_psl - current_psl
            
            # If PSL improves, accept.
            # If PSL worsens but energy improves significantly, accept with probability exp(-delta/T).
            # Actually, just use PSL for acceptance to target the metric directly, 
            # but use a softer metric for the "temperature" part?
            
            # Simple SA on PSL
            if new_psl < current_psl:
                # Accept
                current_psl = new_psl
                current_energy = new_energy
            else:
                # Reject with probability exp(-delta_psl / T)
                # delta_psl is positive
                prob = np.exp(-delta_psl / T)
                if np.random.rand() < prob:
                    current_psl = new_psl
                    current_energy = new_energy
                else:
                    # Revert
                    step[idx] = old_val
                    # current_psl and current_energy remain unchanged
            
            T *= alpha
        
        # Update global best
        if current_psl < best_psl:
            best_psl = current_psl
            best_step = step.copy()
            best_energy = current_energy

    # Final Polish: Local Search on PSL
    if best_step is not None:
        step = best_step.copy()
        current_psl = best_psl
        improved = True
        while improved and (time.time() - start_time < DEADLINE - 2.0):
            improved = False
            for i in range(N):
                if time.time() - start_time > DEADLINE - 2.0:
                    break
                old_val = step[i]
                step[i] = -old_val
                new_psl = _compute_c3(step)
                if new_psl < current_psl:
                    current_psl = new_psl
                    improved = True
                else:
                    step[i] = old_val
    
    return step

# Helper to ensure valid output
def _validate(step):
    if len(step) != N:
        return np.ones(N, dtype=np.int8)
    if not np.all((step == 1) | (step == -1)):
        return np.ones(N, dtype=np.int8)
    return step

# Wrap construct to ensure safety
_original_construct = construct
def construct():
    try:
        result = _original_construct()
        return _validate(result)
    except Exception as e:
        # Fallback to seed (all ones)
        return np.ones(N, dtype=np.int8)
