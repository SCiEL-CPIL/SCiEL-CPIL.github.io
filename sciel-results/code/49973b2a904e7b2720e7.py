"""
Neural network strategy: Single hidden layer ReLU network.
The 10 parameters are the weights and biases of the hidden layer.
This allows the function to be piecewise linear with up to 10 breakpoints,
offering flexibility for non-smooth or highly curved hidden functions
that a global polynomial might struggle with.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the hidden function using a single-layer ReLU neural network.
    
    Architecture:
    Input x -> Linear layer (10 units) -> ReLU -> Linear layer (1 unit, fixed weights) -> Output
    
    The 'params' array of length 10 defines the weights and biases of the hidden layer.
    To keep the number of parameters exactly 10 as required by the evaluator, we map
    these 10 parameters to 5 weights and 5 biases for a 5-unit hidden layer, OR
    10 weights and 10 biases for a 10-unit hidden layer?
    
    Wait, the problem states: "The evaluator fits params, an array of length 10".
    So we have exactly 10 degrees of freedom.
    
    Let's use a 5-unit hidden layer.
    params[0:5] are weights w_i for the 5 units.
    params[5:10] are biases b_i for the 5 units.
    
    The output is a linear combination of the ReLU activations.
    To ensure the output scale is reasonable and the optimizer doesn't struggle with
    scaling, we can fix the output weights to be 1.0 (or a small constant) and let the
    hidden layer weights/biases adapt. However, if the target function has a large amplitude,
    the hidden layer weights must be large.
    
    A more robust approach: 
    Use 10 parameters as:
    - 5 weights for the hidden layer
    - 5 biases for the hidden layer
    
    The output is sum(relu(w_i * x + b_i)).
    This is equivalent to a network with fixed output weights of 1.
    This might be too restrictive if the function has negative slopes or specific amplitudes
    that require negative output weights.
    
    Better: 
    Let's use 10 parameters to define a more flexible 2-layer structure or a different mapping.
    
    Actually, let's look at the degrees of freedom.
    A linear function y = mx + c needs 2 params.
    A quadratic needs 3.
    ...
    A 9th degree polynomial needs 10 params.
    
    A ReLU network with 5 units has 5*2 = 10 params (weights and biases).
    If we fix the output weights to 1, we lose the ability to scale the output negatively or
    independently per unit.
    
    Let's try a different parameterization:
    Use 10 params to define:
    - 1 bias for the output layer? No, let's stick to the hidden layer.
    
    Alternative: 
    Use 10 params as:
    - 5 weights w_i
    - 5 biases b_i
    And fix the output weights to be 1.
    This represents functions of the form sum(relu(w_i x + b_i)).
    This can represent any continuous piecewise linear function with up to 5 breakpoints
    (if w_i are distinct and non-zero).
    
    However, if the target function is, say, x^2, it can be approximated well.
    If the target function is sin(x), it can be approximated well.
    
    Is there a risk that the optimizer gets stuck?
    ReLU networks have non-smooth gradients. BFGS assumes smoothness.
    However, the subgradient is well-defined almost everywhere.
    The "kink" at the zero of the ReLU can cause issues for BFGS.
    
    To mitigate this, we can use a smooth approximation of ReLU, e.g., Softplus or GELU.
    But the directive says "ReLU". Let's stick to ReLU but be aware of potential issues.
    
    Actually, let's consider the scaling.
    If x is in [0.1, 6], and w_i are initialized to 1, then w_i x + b_i is roughly [1.1, 7].
    ReLU(positive) = positive.
    So the output is roughly sum(x) = 5x.
    If the target is, say, 100 * sin(x), the optimizer needs to increase w_i by a factor of 20.
    This is a large change.
    
    To help the optimizer, we can normalize x.
    u = (x - 3.05) / 2.95  # in [-1, 1]
    
    Then w_i * u + b_i.
    If w_i ~ 1, b_i ~ 1, the activations are in [0, 2].
    Output is sum of 5 terms, so ~ 5-10.
    
    Let's implement this.
    
    params[0:5] -> weights w
    params[5:10] -> biases b
    """
    x = np.asarray(x, dtype=float)
    
    # Normalize x to [-1, 1]
    u = (x - 3.05) / 2.95
    
    # Extract weights and biases
    w = params[0:5]
    b = params[5:10]
    
    # Compute hidden layer activations
    # h_i = relu(w_i * u + b_i)
    # Shape: (5, N)
    h = np.maximum(0, w[:, None] * u[None, :] + b[:, None])
    
    # Output is sum of hidden activations
    # This assumes output weights are all 1.
    # To allow for negative slopes or different scales, this might be limiting.
    # But with 5 positive weights, we can only create functions with non-negative derivative?
    # No, if w_i is negative, the slope is negative in the active region.
    # But ReLU kills the negative part.
    # So each unit contributes a "kink" that goes up.
    # The sum of such units is a convex function?
    # No, sum of convex functions is convex.
    # ReLU(w*x+b) is convex.
    # So sum(ReLU(w_i*x+b_i)) is a convex function.
    # This is a HUGE limitation. It cannot represent concave functions like -x^2 or sin(x) (parts of it).
    
    # We need a more flexible architecture.
    # How about alternating signs?
    # We don't have enough parameters to define separate output weights.
    
    # Alternative: Use tanh instead of ReLU?
    # tanh is smooth, odd, and can represent both positive and negative slopes.
    # Sum of tanh(w_i*x + b_i) is not convex. It can represent a wide variety of functions.
    # And it is smooth, which is better for BFGS.
    
    # Let's switch to Tanh.
    # h_i = tanh(w_i * u + b_i)
    # y = sum(h_i)
    
    # This uses 10 parameters (5 weights, 5 biases).
    # The output weights are fixed to 1.
    # This is still a limitation: the output scale is tied to the number of units.
    # But the weights w_i can scale the contribution.
    
    # Let's try this.
    h = np.tanh(w[:, None] * u[None, :] + b[:, None])
    y = np.sum(h, axis=0)
    
    return y
# EVOLVE-BLOCK-END
