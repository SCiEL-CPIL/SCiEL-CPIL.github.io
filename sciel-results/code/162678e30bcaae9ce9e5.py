import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    """
    Construct the sequence using Ising spin chain optimization with soft-max loss.
    """
    # Number of iterations for the internal optimization
    max_iter = 500
    beta = 50.0  # Sharpness of soft-max
    
    # Initialize spins randomly
    np.random.seed(42)
    s = np.random.randn(N)
    
    # Lagrange multiplier for sum constraint
    lam = 0.0
    
    def get_sigma(s):
        # Use tanh to map to [-1, 1]
        return np.tanh(s)
        
    def loss_and_grad(s):
        sigma = get_sigma(s)
        # Enforce sum constraint softly or via projection?
        # Let's use a penalty term for sum != 0
        sum_sigma = np.sum(sigma)
        penalty = 100.0 * sum_sigma**2
        
        # Calculate C_k
        C = np.zeros(N)
        for k in range(1, N):
            shifted = np.roll(sigma, -k)
            # C_k = sum sigma_i * (1 - sigma_{i+k}) / 4 + N/4 ?
            # Recall: C_k = N/4 - 1/4 S_k
            # S_k = sum sigma_i sigma_{i+k}
            S_k = np.sum(sigma * shifted)
            C[k] = N/4.0 - S_k/4.0
            
        # Soft-max loss
        C_max = np.max(C[1:])
        exp_C = np.exp(beta * (C[1:] - C_max))
        sum_exp = np.sum(exp_C)
        loss = C_max + np.log(sum_exp) / beta + penalty
        
        # Gradient
        # dL/dsigma_j = sum_k (weight_k * dC_k/dsigma_j) + d(penalty)/dsigma_j
        # dC_k/dsigma_j = -1/4 * (sigma_{j-k} + sigma_{j+k})
        grad_sigma = np.zeros(N)
        for k in range(1, N):
            weight = beta * exp_C[k-1] / sum_exp
            # dS_k/dsigma_j = sigma_{j-k} + sigma_{j+k}
            # dC_k/dsigma_j = -1/4 * (sigma_{j-k} + sigma_{j+k})
            grad_sigma += -0.25 * weight * (np.roll(sigma, k) + np.roll(sigma, -k))
            
        # Penalty gradient
        grad_sigma += 2.0 * 100.0 * sum_sigma * np.ones(N)
        
        # Chain rule: dsigma/ds = sech^2(s) = 1 - tanh^2(s)
        sech2 = 1.0 - sigma**2
        grad_s = grad_sigma * sech2
        
        return loss, grad_s

    # Optimize s
    # We can use L-BFGS-B or Adam
    # Let's use a simple gradient descent with learning rate schedule
    
    lr = 0.01
    for i in range(max_iter):
        loss, grad = loss_and_grad(s)
        s -= lr * grad
        # Optional: normalize s to prevent tanh saturation
        s /= (np.std(s) + 1e-6)
        
        if i % 100 == 0:
            # Check current max C
            sigma = get_sigma(s)
            C = np.zeros(N)
            for k in range(1, N):
                shifted = np.roll(sigma, -k)
                S_k = np.sum(sigma * shifted)
                C[k] = N/4.0 - S_k/4.0
            # print(f"Iter {i}, Loss {loss}, Max C {np.max(C[1:])}")
            
    # Binarize
    sigma = np.sign(get_sigma(s))
    # Ensure exactly N/2 ones (sigma=1)
    # Count ones
    ones = np.sum(sigma == 1)
    target_ones = N // 2
    
    if ones > target_ones:
        # Flip some 1s to -1
        # Choose those with lowest contribution to reduction?
        # Just flip random ones
        indices_1 = np.where(sigma == 1)[0]
        to_flip = np.random.choice(indices_1, ones - target_ones, replace=False)
        sigma[to_flip] = -1
    elif ones < target_ones:
        indices_neg = np.where(sigma == -1)[0]
        to_flip = np.random.choice(indices_neg, target_ones - ones, replace=False)
        sigma[to_flip] = 1
        
    # Convert to h
    h = (sigma + 1) / 2.0
    
    return h
