import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    time_limit = 140.0 # Safety margin
    
    best_score = np.inf
    best_x = None
    
    # Helper to evaluate
    def evaluate(x):
        # x is 1D array
        # Normalize to unit energy to make gradient scale invariant?
        # Or just compute ratio directly.
        energy = np.dot(x, x)
        if energy < 1e-9:
            return np.inf
        # Zero pad to 2N for linear autoconvolution
        X = np.fft.fft(x, 2*N)
        P = np.abs(X)**2
        c = np.fft.ifft(P).real
        # c[0] is lag 0.
        # Lags 1..N-1 are c[1:N]
        # Lags -(N-1)..-1 are c[N:2N]
        # We want max abs of all lags except 0
        # c[N] is lag -1? 
        # For real x, c[k] = c[-k].
        # So max(c[1:N], c[N:2N-1]) is sufficient?
        # c[2N-1] corresponds to lag -(N-1)?
        # Let's just take max of all except c[0].
        max_side = np.max(np.abs(np.concatenate([c[1:N], c[N:]])))
        return max_side / energy

    # Gradient Descent on continuous relaxation
    def optimize(x_init, max_steps=5000, lr=0.01):
        x = x_init.astype(float)
        # Normalize to unit energy initially
        norm = np.linalg.norm(x)
        if norm > 0:
            x = x / norm
        
        for i in range(max_steps):
            if time.time() - start_time > time_limit:
                break
            
            # Compute Autoconvolution
            X = np.fft.fft(x, 2*N)
            P = np.abs(X)**2
            c = np.fft.ifft(P).real
            
            # Find max side lobe
            # Mask out c[0]
            c_masked = c.copy()
            c_masked[0] = -np.inf
            # Note: c is real.
            k_star = np.argmax(np.abs(c_masked))
            
            if k_star == 0:
                break # Should not happen
            
            # Get the value
            c_val = c[k_star]
            sign_c = np.sign(c_val)
            if sign_c == 0:
                sign_c = 1.0
            
            # Gradient of |c_k| w.r.t x
            # c_k = sum x_j x_{j+k}
            # d c_k / d x_j = x_{j+k} + x_{j-k}
            # Indices must be valid (0 to N-1)
            
            grad = np.zeros(N)
            
            # Term 1: x_{j+k}
            # j ranges 0 to N-1
            # j+k ranges k to N-1+k
            # Valid if j+k < N => j < N-k
            if k_star < N:
                # j from 0 to N-k_star-1
                idx_j = np.arange(N - k_star)
                idx_x = np.arange(k_star, N)
                grad[idx_j] += x[idx_x]
                
                # Term 2: x_{j-k}
                # j-k >= 0 => j >= k
                # j ranges k_star to N-1
                idx_j2 = np.arange(k_star, N)
                idx_x2 = np.arange(0, N - k_star)
                grad[idx_j2] += x[idx_x2]
            
            # Wait, the formula for linear correlation gradient is slightly different if we consider boundaries.
            # c_k = sum_{j=0}^{N-1-k} x_j x_{j+k}
            # d c_k / d x_j:
            # If j < N-k, x_j is in the first term (as x_j) and second term (as x_{j+k} when index is j-k? No).
            # Let's use the property: c_k = <x, shift(x, k)>
            # d/dx <x, y> = y. d/dy <x, y> = x.
            # Here y = shift(x, k).
            # So grad = shift(x, -k) + shift(x, k) ?
            # Let's verify with small example.
            # x = [a, b, c], k=1.
            # c_1 = a*b + b*c.
            # d/da = b. d/db = a+c. d/dc = b.
            # shift(x, 1) = [b, c, 0] (assuming zero padding outside).
            # shift(x, -1) = [0, a, b].
            # Sum = [b, a+c, b]. Correct.
            
            # So we can compute gradient using np.roll or slicing.
            grad = np.zeros(N)
            
            # shift x by k_star (circular? No, linear correlation uses zero padding effectively)
            # For linear correlation, the "shifted" vector has zeros outside.
            
            # Contribution from x_{j+k}:
            # This is x shifted by -k_star?
            # If we define shift_forward(x, k) such that (shift(x, k))_j = x_{j+k},
            # then grad += shift_forward(x, k_star)
            # And grad += shift_forward(x, -k_star)
            
            # Implement shift_forward(x, k)
            # If k > 0:
            #   out[0:N-k] = x[k:N]
            #   out[N-k:N] = 0
            # If k < 0:
            #   out[-k:N] = x[0:N+k]
            #   out[0:-k] = 0
            
            def shift_forward(x, k):
                out = np.zeros(N)
                if k > 0:
                    out[:N-k] = x[k:]
                elif k < 0:
                    out[-k:] = x[:N+k]
                return out
            
            grad += shift_forward(x, k_star)
            grad += shift_forward(x, -k_star)
            
            # Gradient of |c_k| is sign(c_k) * grad_c_k
            grad = sign_c * grad
            
            # Update x
            x = x - lr * grad
            
            # Project to unit sphere (keep energy constant)
            norm = np.linalg.norm(x)
            if norm > 0:
                x = x / norm
            else:
                x = np.random.choice([-1, 1], N)
                x = x / np.linalg.norm(x)
                
            # Optional: Soft threshold or just keep continuous
            # We will threshold at the end
            
        # Threshold to +/- 1
        x_bin = np.sign(x)
        x_bin[x_bin == 0] = 1.0 # Handle zeros
        return x_bin

    # Run multiple restarts
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() - start_time > time_limit - 5:
            break
            
        x_init = np.random.choice([-1, 1], N)
        x_opt = optimize(x_init, max_steps=2000, lr=0.05)
        
        # Evaluate the binary result
        score = evaluate(x_opt)
        
        if score < best_score:
            best_score = score
            best_x = x_opt
            
        # Also evaluate the continuous result? 
        # The task asks for "signed step heights". 
        # If I return continuous, score might be different.
        # But "step functions" usually implies constant values in bins.
        # Binary is the standard interpretation for "signed" in this context (Barker etc).
        
    # If best_x is None (shouldn't happen), fallback to random
    if best_x is None:
        best_x = np.random.choice([-1, 1], N)
        
    return best_x
