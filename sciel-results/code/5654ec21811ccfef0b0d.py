"""Optimized program for the third autocorrelation inequality task."""
import numpy as np
import time
import sys

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    def compute_value(h):
        """Compute the objective value (lower is better)."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = np.convolve(h, h)
        return abs(2.0 * N * np.max(conv) / (s * s))
    
    def smooth_max(x, tau=0.01):
        """Soft approximation of max using log-sum-exp."""
        x_max = np.max(x)
        return x_max + tau * np.log(np.sum(np.exp((x - x_max) / tau)))
    
    def objective_signed(params):
        """Objective allowing signed h, with penalty if sum is small."""
        h = params
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=0.05)
        return 2.0 * N * sm / (s * s)
    
    def zc_sequence(n, u, v):
        """Generate Zadoff-Chu sequence: z[i] = exp(-j*pi*u*v*i*(i+1)/n) for i=0..n-1.
        Returns real-valued sequence by taking real part."""
        i = np.arange(n)
        phase = -np.pi * u * v * i * (i + 1) / n
        return np.cos(phase)
    
    def zc_objective(params):
        """Objective for Zadoff-Chu parameters: [u, v, scale, offset, phase_shift].
        h[i] = scale * zc[i] + offset, where zc is Zadoff-Chu with params u, v.
        phase_shift allows cyclic rotation of the sequence."""
        u, v, scale, offset, phase_shift = params
        # Ensure u and v are coprime integers in [1, N-1]
        u_int = int(np.round(u))
        v_int = int(np.round(v))
        if u_int < 1: u_int = 1
        if u_int >= N: u_int = N - 1
        if v_int < 1: v_int = 1
        if v_int >= N: v_int = N - 1
        
        zc = zc_sequence(N, u_int, v_int)
        # Apply cyclic shift
        shift = int(np.round(phase_shift * N)) % N
        if shift != 0:
            zc = np.roll(zc, shift)
        
        h = scale * zc + offset
        s = np.sum(h)
        if abs(s) < 1e-4:
            return 1e10 + (1e-4 - abs(s)) * 1e6
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=0.05)
        return 2.0 * N * sm / (s * s)
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Phase 1: Search over Zadoff-Chu parameters using differential evolution
    if time.time() < deadline - 40.0:
        try:
            from scipy.optimize import differential_evolution
            # u and v should be coprime integers in [1, N-1]
            # For ZC sequences, gcd(u, N) = 1 is required for good autocorrelation properties
            # N = 400 = 2^4 * 5^2, so u must be odd and not divisible by 5
            valid_u = [u for u in range(1, N) if np.gcd(u, N) == 1]
            valid_v = [v for v in range(1, N) if np.gcd(v, N) == 1]
            
            # Use a subset for DE to keep it tractable
            # Sample from valid values
            rng_de = np.random.RandomState(42)
            u_samples = rng_de.choice(valid_u, size=20, replace=False)
            v_samples = rng_de.choice(valid_v, size=20, replace=False)
            
            bounds = [
                (min(u_samples), max(u_samples)),  # u
                (min(v_samples), max(v_samples)),  # v
                (0.5, 2.0),  # scale
                (-0.5, 0.5),  # offset
                (0.0, 1.0)   # phase_shift (cyclic rotation fraction)
            ]
            
            de_result = differential_evolution(
                zc_objective,
                bounds,
                maxiter=30,
                popsize=10,
                seed=42,
                tol=1e-6,
                polish=True
            )
            if de_result.fun < best_val:
                u, v, scale, offset, phase_shift = de_result.x
                u_int = int(np.round(u))
                v_int = int(np.round(v))
                if u_int < 1: u_int = 1
                if u_int >= N: u_int = N - 1
                if v_int < 1: v_int = 1
                if v_int >= N: v_int = N - 1
                
                zc = zc_sequence(N, u_int, v_int)
                shift = int(np.round(phase_shift * N)) % N
                if shift != 0:
                    zc = np.roll(zc, shift)
                h_zc = scale * zc + offset
                val_zc = compute_value(h_zc)
                if val_zc < best_val and np.all(np.isfinite(h_zc)) and abs(np.sum(h_zc)) > 1e-6:
                    best_h = h_zc.copy()
                    best_val = val_zc
                    print(f"ZC DE: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"ZC DE error: {e}", file=sys.stderr)
    
    # Phase 2: Grid search over key ZC parameters for better coverage
    if time.time() < deadline - 30.0:
        try:
            valid_u = [u for u in range(1, N) if np.gcd(u, N) == 1]
            valid_v = [v for v in range(1, N) if np.gcd(v, N) == 1]
            
            # Test a diverse set of (u, v) pairs
            test_pairs = []
            for u in valid_u[:15]:
                for v in valid_v[:15]:
                    test_pairs.append((u, v))
            
            # Also test some specific pairs known to have good properties
            test_pairs.extend([(3, 1), (7, 1), (9, 1), (13, 1), (15, 1),
                              (3, 3), (7, 7), (9, 9), (13, 13), (15, 15),
                              (3, 7), (7, 3), (9, 13), (13, 9)])
            
            # Remove duplicates
            test_pairs = list(set(test_pairs))
            
            for u, v in test_pairs[:40]:  # Limit to 40 pairs
                if time.time() > deadline - 15.0:
                    break
                
                zc = zc_sequence(N, u, v)
                
                # Try different scales and offsets
                for scale in [0.5, 0.8, 1.0, 1.2, 1.5]:
                    for offset in [-0.2, 0.0, 0.2]:
                        h_test = scale * zc + offset
                        s_test = np.sum(h_test)
                        if abs(s_test) < 1e-6:
                            continue
                        val_test = compute_value(h_test)
                        if val_test < best_val and np.all(np.isfinite(h_test)):
                            best_h = h_test.copy()
                            best_val = val_test
                            print(f"ZC grid ({u},{v},s={scale},o={offset}): {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"ZC grid error: {e}", file=sys.stderr)
    
    # Phase 3: Local refinement using L-BFGS-B from best
    if time.time() < deadline - 10.0:
        try:
            from scipy.optimize import minimize
            result = minimize(
                objective_signed,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_opt = result.x
            val_opt = compute_value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"L-BFGS-B refine: improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"L-BFGS-B error: {e}", file=sys.stderr)
    
    # Phase 4: Try multiple ZC-based initializations with L-BFGS-B
    rng = np.random.RandomState(123)
    for trial in range(8):
        if time.time() > deadline - 5.0:
            break
        
        # Pick random valid u, v
        valid_u = [u for u in range(1, N) if np.gcd(u, N) == 1]
        valid_v = [v for v in range(1, N) if np.gcd(v, N) == 1]
        u = int(rng.choice(valid_u))
        v = int(rng.choice(valid_v))
        
        zc = zc_sequence(N, u, v)
        shift = int(rng.randint(0, N))
        zc = np.roll(zc, shift)
        
        scale = rng.uniform(0.5, 1.5)
        offset = rng.uniform(-0.3, 0.3)
        h0 = scale * zc + offset
        
        s0 = np.sum(h0)
        if abs(s0) < 1.0:
            h0 = h0 + (1.0 - s0) / N
        
        try:
            from scipy.optimize import minimize
            result = minimize(
                objective_signed,
                h0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_opt = result.x
            val_opt = compute_value(h_opt)
            if val_opt < best_val and np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-6:
                best_h = h_opt.copy()
                best_val = val_opt
                print(f"ZC trial {trial} ({u},{v}): improved to {best_val:.6f}", file=sys.stderr)
        except Exception as e:
            print(f"ZC trial {trial}: error {e}", file=sys.stderr)
    
    # Final safety check
    if time.time() > deadline - 2.0:
        return np.ones(N)
    
    if best_val < 2.0 and np.all(np.isfinite(best_h)) and abs(np.sum(best_h)) > 1e-6:
        return best_h
    else:
        return np.ones(N)
    
# EVOLVE-BLOCK-END
