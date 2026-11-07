"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Neural Network approach for symbolic regression.
    
    Strategy:
    Instead of a fixed linear basis of polynomials and trig functions, this program
    uses a parametric non-linear model: a single hidden neuron (Tanh) combined with
    a linear bypass. This allows the model to learn arbitrary non-linear interactions
    between t, x, and v, which a fixed polynomial basis might miss or require too
    many terms to approximate.
    
    Architecture:
    y = w_out * tanh(w_in1 * t + w_in2 * x + w_in3 * v + b_in) + b_out 
        + c_lin1 * t + c_lin2 * x + c_lin3 * v + c_lin0
    
    Parameters (10 total):
    p[0]: w_in1 (weight for t in hidden layer)
    p[1]: w_in2 (weight for x in hidden layer)
    p[2]: w_in3 (weight for v in hidden layer)
    p[3]: b_in  (bias for hidden layer)
    p[4]: w_out (weight from hidden layer to output)
    p[5]: b_out (bias for output)
    p[6]: c_lin0 (linear bias / offset)
    p[7]: c_lin1 (linear weight for t)
    p[8]: c_lin2 (linear weight for x)
    p[9]: c_lin3 (linear weight for v)
    
    This formulation is non-linear in the parameters (due to the product of weights
    inside the tanh and outside), so BFGS will be used to fit them.
    """
    p = params
    
    # Extract parameters
    w_in1 = p[0]
    w_in2 = p[1]
    w_in3 = p[2]
    b_in = p[3]
    w_out = p[4]
    b_out = p[5]
    c_lin0 = p[6]
    c_lin1 = p[7]
    c_lin2 = p[8]
    c_lin3 = p[9]
    
    # Calculate hidden layer activation
    # Note: We do not normalize inputs explicitly to keep the expression simple,
    # but the optimizer should find appropriate scales for the weights.
    z = w_in1 * t + w_in2 * x + w_in3 * v + b_in
    h = np.tanh(z)
    
    # Calculate output
    # Non-linear part
    y_nonlin = w_out * h + b_out
    
    # Linear bypass part
    y_lin = c_lin0 + c_lin1 * t + c_lin2 * x + c_lin3 * v
    
    y = y_nonlin + y_lin
    
    return y

# EVOLVE-BLOCK-END
