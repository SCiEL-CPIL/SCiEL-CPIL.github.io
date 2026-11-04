"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))

def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _generate_zadoff_chu_candidates(rng, deadline):
    """
    Generate candidates from complex Zadoff-Chu sequences.
    x[n] = Re( exp(i * pi * q * n^2 / N) ) * sign(Re( ... )) 
    Actually, let's just use the real part of the complex sequence, 
    and also try thresholding it to +/- 1 if it helps the integral normalization?
    The task asks for "signed step heights". It doesn't explicitly restrict to +/- 1, 
    but the seed used +/- 1. 
    However, "signed step functions" usually implies real values.
    Let's try both: 
    1. x[n] = Re( exp(i * pi * q * n^2 / N) )
    2. x[n] = sign(Re( exp(i * pi * q * n^2 / N) )) (binary +/- 1)
    
    The metric divides by (integral)^2. 
    For random +/- 1, integral ~ sqrt(N). Max autoconv ~ N. Ratio ~ N/N = 1.
    For continuous values, integral might be larger or smaller.
    Zadoff-Chu has constant amplitude. Re( ... ) has amplitude <= 1.
    If we use continuous values, the "step heights" are real numbers.
    Let's generate a variety of q values (root indices).
    """
    candidates = []
    
    # q must be coprime to N for good properties, but let's just sweep a range.
    # N = 400. 
    # Standard Zadoff-Chu length is usually prime or power of 2. 400 is not prime.
    # We can still use the phase formula.
    
    # Sweep q from 1 to N//2
    # We don't need to check every q, maybe step by 1 or 2.
    # Let's try all q in [1, N//2] that are coprime to N, plus some others.
    # Actually, just sweep q in 1..N//2. It's fast to compute.
    
    n_vals = np.arange(N)
    
    for q in range(1, N // 2 + 1):
        if time.time() > deadline:
            break
            
        # Complex sequence: z[n] = exp(1j * pi * q * n^2 / N)
        # Phase = pi * q * n^2 / N
        phase = np.pi * q * n_vals * n_vals / N
        z = np.exp(1j * phase)
        
        # Candidate 1: Real part
        x1 = z.real.copy()
        # Candidate 2: Imaginary part
        x2 = z.imag.copy()
        # Candidate 3: Sign of Real part (Binary +/- 1)
        x3 = np.sign(x1)
        # Ensure no zeros (sign(0) is 0)
        x3[x3 == 0] = 1.0
        # Candidate 4: Sign of Imaginary part
        x4 = np.sign(x2)
        x4[x4 == 0] = 1.0
        
        for x in [x1, x2, x3, x4]:
            # Quick check: if integral is too small, metric is bad.
            s = np.sum(x)
            if abs(s) < 1e-6:
                continue
            val = _metric(x)
            candidates.append((val, x))
            
    return candidates

def _local_search_sign_flip(x, deadline, rng, max_iters=100000):
    """
    Local search by flipping signs of individual elements.
    This preserves the 'chaotic' structure while allowing fine-tuning.
    """
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    
    # Precompute FFT of x for faster updates? 
    # Flipping sign of x[i] changes X[k] by -2*x[i]*exp(-2*pi*i*k/N).
    # This allows O(N) update of spectrum, then O(N log N) for conv?
    # Actually, full recalculation is O(N log N). 
    # N=400. N log N ~ 3200 ops. 
    # 100000 iters * 3200 = 320M ops. Feasible in ~10-20s with NumPy if optimized.
    
    # Optimization: Instead of copying the whole array every time, 
    # we can update the FFT directly.
    # Let X = FFT(x).
    # If we flip x[i] -> -x[i], delta_x = -2*x[i] at index i.
    # Delta_X[k] = -2*x[i] * exp(-2*pi*i*k/N).
    # New_X[k] = X[k] + Delta_X[k].
    # Then compute conv from New_X.
    # This saves the forward FFT. We still need inverse FFT.
    # Inverse FFT is O(N log N).
    
    X = np.fft.rfft(best_x, _FFT_SIZE)
    
    # Precompute twiddle factors for the update?
    # twiddle[k] = exp(-2*pi*i*k/N) for k in 0..N/2
    # But we need to multiply by x[i].
    # Let's stick to the simple copy method first, it might be fast enough for N=400.
    # 400 points FFT is very fast in numpy.
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        if no_improve > 1000:
            break
            
        # Pick random index
        idx = rng.integers(0, N)
        
        # Flip sign
        x_test = best_x.copy()
        x_test[idx] *= -1
        
        val_test = _metric(x_test)
        
        if val_test < best_val:
            best_val = val_test
            best_x = x_test
            X = np.fft.rfft(best_x, _FFT_SIZE) # Update spectrum cache if we used it, but we didn't really use it for speed yet.
            no_improve = 0
        else:
            no_improve += 1
            
    return best_x, best_val

def _construct():
    deadline = time.time() + 145.0
    
    # Start with a random baseline
    rng = np.random.default_rng(42)
    best_x = rng.integers(0, 2, size=N).astype(float) * 2 - 1
    best_val = _metric(best_x)
    
    # Phase 1: Generate structured candidates from Zadoff-Chu
    gen_deadline = time.time() + 30.0 # Give 30s to generation
    candidates = _generate_zadoff_chu_candidates(rng, gen_deadline)
    
    if not candidates:
        return best_x
        
    # Sort by metric
    candidates.sort(key=lambda item: item[0])
    
    # Phase 2: Refine top candidates
    top_k = min(20, len(candidates))
    
    refine_deadline = deadline - 5.0
    # Allocate remaining time among top candidates
    # We have ~110s left. Top 20 candidates. 5.5s each.
    
    for i in range(top_k):
        if time.time() > refine_deadline:
            break
            
        val, x = candidates[i]
        
        # If this candidate is already worse than our current best, 
        # we might still improve it, but let's prioritize better ones.
        # However, local search can change the basin.
        
        # Allocate time based on remaining budget
        time_left = refine_deadline - time.time()
        iters_remaining = top_k - i
        time_per_iter_block = time_left / iters_remaining
        max_iters_for_this = int(time_per_iter_block * 5000) # Rough estimate of iters per second
        # Cap max iters to avoid running too long on one if others are slow
        max_iters_for_this = min(max_iters_for_this, 20000)
        
        x_opt, val_opt = _local_search_sign_flip(x, time.time() + time_per_iter_block, rng, max_iters=max_iters_for_this)
        
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt
            
    # Phase 3: Final polish on the absolute best found
    if time.time() < deadline - 10:
        x_final, val_final = _local_search_sign_flip(best_x, deadline - 2.0, rng, max_iters=50000)
        if val_final < best_val:
            best_val = val_final
            best_x = x_final

    return best_x

def construct():
    return _construct()
# EVOLVE-BLOCK-END
