import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Initialize x
    # Start with a random balanced sequence
    np.random.seed(42)
    x = np.random.uniform(-0.5, 0.5, N)
    x -= np.mean(x)
    x = np.clip(x, -0.5, 0.5)
    
    best_x = x.copy()
    best_score = np.inf
    
    beta = 10.0  # Soft-max temperature
    eta = 0.1    # Learning rate
    
    for i in range(5000):
        if time.time() > deadline:
            break
            
        # Compute autocorrelation A
        X = np.fft.fft(x)
        P = np.abs(X)**2
        A = np.real(np.fft.ifft(P))
        
        # Mask out k=0
        A_masked = A.copy()
        A_masked[0] = -np.inf
        
        # Find current max
        current_max = np.max(A_masked)
        
        if current_max < best_score:
            best_score = current_max
            best_x = x.copy()
            
        # Compute weights for soft-max
        # Shift A to avoid overflow
        A_shift = A_masked - current_max
        weights = np.exp(beta * A_shift)
        weights[0] = 0
        sum_w = np.sum(weights)
        if sum_w > 0:
            weights /= sum_w
        else:
            weights = np.zeros(N)
            weights[np.argmax(A_masked)] = 1.0 # Fallback
            
        # Compute gradient
        # G_j = sum_k weights_k * (x_{j-k} + x_{j+k})
        G = np.zeros(N)
        for k in range(1, N):
            if weights[k] > 0:
                # x_{j-k} is x shifted right by k (or left?)
                # A_k = sum x_i x_{i+k}
                # d/dx_j = x_{j-k} + x_{j+k}
                # x_{j-k} corresponds to x shifted by -k
                # x_{j+k} corresponds to x shifted by k
                G += weights[k] * (np.roll(x, k) + np.roll(x, -k))
                
        # Update
        x = x - eta * G
        
        # Project to sum 0
        x -= np.mean(x)
        
        # Project to box
        x = np.clip(x, -0.5, 0.5)
        
        # Re-project to sum 0 (clipping might break it)
        x -= np.mean(x)
        
        # Final check on bounds (if mean subtraction pushed out)
        x = np.clip(x, -0.5, 0.5)

    # Construct h
    h = 0.5 + best_x
    
    # Ensure exact sum
    # h sum is N/2 + sum(best_x). sum(best_x) is approx 0.
    # Adjust one element to make sum exact?
    # The task says "sum is exactly len/2".
    # Floating point errors might accumulate.
    # Let's normalize h to sum exactly N/2 while keeping bounds?
    # If we scale h, bounds might be violated.
    # Better: adjust the largest element by the residual.
    residual = N/2 - np.sum(h)
    idx = np.argmax(h)
    h[idx] += residual
    
    # Clip again to be safe
    h = np.clip(h, 0, 1)
    
    return h
