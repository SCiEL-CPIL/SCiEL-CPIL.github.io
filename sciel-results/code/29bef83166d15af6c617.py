import numpy as np
import time

def equation(strain, temp, params):
    """
    Gaussian Process-inspired Universal Approximator.
    
    Strategy:
    Instead of a fixed physical form, we use a sum of Gaussian basis functions 
    (a simplified GP with fixed centers and parameterized variances/amplitudes).
    This allows the model to fit arbitrary non-monotonic or complex yield transitions.
    
    The 10 parameters control:
    p[0]: Global scale (amplitude)
    p[1]: Length scale for strain (controls width of basis functions)
    p[2]: Length scale for temperature
    p[3]: Center offset for strain (shift)
    p[4]: Center offset for temperature (shift)
    p[5]: Amplitude weight for first basis function
    p[6]: Amplitude weight for second basis function
    p[7]: Amplitude weight for third basis function
    p[8]: Amplitude weight for fourth basis function
    p[9]: Linear baseline slope
    
    Model:
    sigma = p[9] * strain + p[0] * sum_{i=1}^{4} w_i * exp(-0.5 * ((s - c_si)^2 / l_s^2 + (t - c_ti)^2 / l_t^2))
    
    To avoid hardcoding centers, we derive them from parameters to maintain differentiability.
    However, a sum of 4 fixed Gaussians might be too restrictive. 
    Let's use a more flexible approach: a single hidden layer neural network style.
    
    Actually, let's stick to the prompt's suggestion: "Neural Network-inspired Universal Approximator 
    with a single hidden layer of 5 sigmoidal basis functions".
    
    Model:
    h_i = sigmoid(p[0] * (strain - p[1]) + p[2] * (temp - p[3]))  # for i=1..5? 
    But we only have 10 params. We can't have 5 distinct centers and scales.
    
    Let's reinterpret the "10 params" constraint for a 1-hidden-layer NN.
    Input: strain, temp.
    Hidden layer size: 4.
    Weights W (4x2) = 8 params.
    Bias b (4) = 4 params.
    Output weight v (4) = 4 params.
    Total = 16 params. Too many.
    
    Let's reduce hidden layer to 2 neurons.
    W (2x2) = 4 params.
    b (2) = 2 params.
    v (2) = 2 params.
    Total = 8 params. We have 2 left for scaling/offset.
    
    Let's try:
    p[0], p[1]: Weights for hidden neuron 1 (strain, temp)
    p[2], p[3]: Weights for hidden neuron 2 (strain, temp)
    p[4], p[5]: Biases for hidden neurons 1 and 2
    p[6], p[7]: Output weights for hidden neurons 1 and 2
    p[8]: Linear coefficient for strain
    p[9]: Linear coefficient for temp (or bias)
    
    Model:
    z1 = p[0]*strain + p[1]*temp + p[4]
    z2 = p[2]*strain + p[3]*temp + p[5]
    a1 = sigmoid(z1)
    a2 = sigmoid(z2)
    sigma = p[6]*a1 + p[7]*a2 + p[8]*strain + p[9]*temp
    
    This is a very flexible 2-neuron neural network. It can approximate any continuous function 
    on a compact domain given enough capacity, and 2 neurons with sigmoid activation is often 
    sufficient for smooth univariate/bivariate trends if the weights are well-scaled.
    BFGS should be able to optimize these 10 parameters effectively.
    """
    s = strain
    t = temp
    p = params
    
    # Hidden layer 1
    z1 = p[0] * s + p[1] * t + p[4]
    # Hidden layer 2
    z2 = p[2] * s + p[3] * t + p[5]
    
    # Stable sigmoid to avoid overflow in exp
    # sigmoid(z) = 1 / (1 + exp(-z))
    # For large positive z, exp(-z) is small, sigmoid ~ 1
    # For large negative z, exp(-z) is large, sigmoid ~ 0
    # np.exp can overflow if z is very negative (e.g., -1000).
    # Use a stable implementation.
    def stable_sigmoid(z):
        # Clip z to prevent extreme values, or use piecewise definition
        # Since BFGS starts at 1, and data is normalized-ish, z shouldn't be huge initially.
        # But during optimization, it might grow.
        # A common trick: 
        # if z >= 0: 1 / (1 + exp(-z))
        # else: exp(z) / (1 + exp(z))
        # Vectorized:
        pos = z >= 0
        res = np.empty_like(z)
        exp_neg = np.exp(-z[pos])
        res[pos] = 1.0 / (1.0 + exp_neg)
        exp_pos = np.exp(z[~pos])
        res[~pos] = exp_pos / (1.0 + exp_pos)
        return res

    a1 = stable_sigmoid(z1)
    a2 = stable_sigmoid(z2)
    
    # Output layer
    stress = p[6] * a1 + p[7] * a2 + p[8] * s + p[9] * t
    
    return stress
