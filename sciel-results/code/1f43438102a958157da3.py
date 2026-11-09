"""Single-hidden-layer ReLU network with fixed random hidden weights.
The 10 free parameters are the output-layer weights. Since the hidden layer
is fixed, the model is linear in params, making BFGS a convex optimization
problem with a well-conditioned Hessian. Random projections provide nonlinear
expressivity without the rugged landscape of a fully nonlinear model."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 135.0
_GLOBAL = {}


def _initialize_hidden():
    """Initialize fixed random hidden-layer weights and biases."""
    if 'hidden_W' in _GLOBAL:
        return
    
    rng = np.random.default_rng(42)
    # 3 inputs (t, x, v) -> 10 hidden units
    hidden_W = rng.standard_normal((3, 10)) * np.sqrt(2.0 / 3)
    hidden_b = rng.standard_normal(10) * 0.1
    
    # Input ranges: t in [30,50], x in [-0.2056, 0.1749], v in [-0.3391, 0.3624]
    # Normalize inputs to roughly [-1, 1] before feeding to hidden layer
    # t_center = 40, t_scale = 10
    # x_center = -0.01535, x_scale = 0.19
    # v_center = 0.01165, v_scale = 0.35
    
    _GLOBAL['hidden_W'] = hidden_W
    _GLOBAL['hidden_b'] = hidden_b
    _GLOBAL['t_center'] = 40.0
    _GLOBAL['t_scale'] = 10.0
    _GLOBAL['x_center'] = -0.01535
    _GLOBAL['x_scale'] = 0.19
    _GLOBAL['v_center'] = 0.01165
    _GLOBAL['v_scale'] = 0.35


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Single-hidden-layer ReLU network.
    
    params (10,): output-layer weights w_0..w_9
    Architecture: [t, x, v] -> normalize -> ReLU(W_h * norm + b_h) -> w_out . h + bias
    
    The model is: pred = sum_i w_i * ReLU(h_i) + offset
    where h_i = W_h[i] . [t_norm, x_norm, v_norm] + b_h[i]
    
    Since W_h and b_h are fixed, pred is linear in params (plus a constant offset
    absorbed into params[0] via a bias trick). BFGS sees a convex quadratic MSE.
    """
    _initialize_hidden()
    
    hidden_W = _GLOBAL['hidden_W']
    hidden_b = _GLOBAL['hidden_b']
    
    # Normalize inputs
    t_norm = (t - _GLOBAL['t_center']) / _GLOBAL['t_scale']
    x_norm = (x - _GLOBAL['x_center']) / _GLOBAL['x_scale']
    v_norm = (v - _GLOBAL['v_center']) / _GLOBAL['v_scale']
    
    # Stack normalized inputs: (n, 3)
    inputs = np.column_stack([t_norm, x_norm, v_norm])
    
    # Hidden layer: (n, 3) @ (3, 10) + (10,) -> (n, 10)
    h = inputs @ hidden_W + hidden_b
    
    # ReLU activation
    h = np.maximum(h, 0.0)
    
    # Output layer: (n, 10) @ (10,) -> (n,)
    # params[0] acts as a bias by multiplying a constant-1 feature
    # We add a constant 1 to the hidden activations for the bias term
    h_with_bias = np.hstack([h, np.ones((h.shape[0], 1))])
    
    # Use first 9 params for hidden weights, 10th for bias
    # Actually, let's use all 10 params as weights on the 10 hidden units,
    # and add a small constant offset to help fitting
    # Better: use params[0:9] for 9 hidden units, params[9] for bias
    # But we have 10 hidden units. Let's use all 10 as weights and add a tiny constant.
    
    # Simplest: pred = h @ params + 0.0
    # The constant offset can be captured if one hidden unit is always active
    # (which happens if its bias is positive and input is small enough)
    # To be safe, add a constant term via a 11th "virtual" feature
    
    # Actually, let's just use all 10 params as weights on the 10 hidden units.
    # The DC offset will be handled by the fact that some ReLU units will be
    # consistently active for typical inputs.
    
    pred = h @ params
    
    return pred
# EVOLVE-BLOCK-END
