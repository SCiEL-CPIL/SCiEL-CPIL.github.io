"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a 2-layer Neural Network with Hard-Tanh activations.
    
    Strategy:
    The linear model (seed) assumes additivity. Biological growth often involves
    interactions (e.g., substrate saturation, temperature optima, pH constraints).
    A small MLP with 1 hidden layer (3 units) and Hard-Tanh activations can
    approximate non-linear interactions.
    
    Architecture:
    Inputs: b, s, temp, pH (4 features)
    Hidden: 3 units, Hard-Tanh activation
    Output: 1 unit, Linear
    
    Weights (10 params):
    p[0], p[1], p[2], p[3] -> Weights from b, s, temp, pH to hidden unit 0
    p[4]                    -> Bias for hidden unit 0
    p[5]                    -> Weight from hidden unit 0 to output
    p[6]                    -> Weights from b, s, temp, pH to hidden unit 1? 
                               Wait, I only have 10 params total.
    
    Let's design a specific 10-param MLP:
    Input: [b, s, temp, pH]
    
    Option A: 2 Hidden Units
    H1 = tanh(w1_b*b + w1_s*s + w1_t*temp + w1_p*pH + b1)
    H2 = tanh(w2_b*b + w2_s*s + w2_t*temp + w2_p*pH + b2)
    Out = w_o1*H1 + w_o2*H2 + b_o
    
    Params:
    w1_b, w1_s, w1_t, w1_p, b1 (5 params)
    w2_b, w2_s, w2_t, w2_p, b2 (5 params)
    Total so far: 10. But no output weights/bias.
    This forces the output to be a linear combo of H1, H2 with fixed weights 1,1.
    This might be too restrictive.
    
    Option B: 1 Hidden Unit, but with direct path?
    H1 = tanh(w1_b*b + w1_s*s + w1_t*temp + w1_p*pH + b1)
    Out = w_o1*H1 + w_o2*b + w_o3*s + w_o4*temp + w_o5*pH + b_o
    Params:
    w1_b, w1_s, w1_t, w1_p, b1 (5)
    w_o1, w_o2, w_o3, w_o4, w_o5, b_o (6)
    Total: 11. Too many.
    
    Option C: 1 Hidden Unit, shared linear path?
    Let's try:
    H1 = tanh(w1_b*b + w1_s*s + w1_t*temp + w1_p*pH + b1)
    Out = w_o1*H1 + w_o2*(b + s + temp + pH) + b_o
    Params:
    w1_b, w1_s, w1_t, w1_p, b1 (5)
    w_o1, w_o2, b_o (3)
    Total: 8. Can use 2 more for a second hidden unit or direct terms.
    
    Let's go with Option D: 2 Hidden Units, but simplified output.
    H1 = tanh(w1_b*b + w1_s*s + w1_t*temp + w1_p*pH + b1)
    H2 = tanh(w2_b*b + w2_s*s + w2_t*temp + w2_p*pH + b2)
    Out = w_o1*H1 + w_o2*H2 + b_o
    Params:
    w1_b, w1_s, w1_t, w1_p, b1 (5)
    w2_b, w2_s, w2_t, w2_p, b2 (5)
    Wait, that's 10 params for inputs/hidden biases. No output weights.
    This implies Out = H1 + H2.
    
    Let's try Option E: 1 Hidden Unit, 3 Direct Terms, 1 Bias.
    H1 = tanh(w1_b*b + w1_s*s + w1_t*temp + w1_p*pH + b1)
    Out = w_o1*H1 + w_o2*b + w_o3*s + w_o4*temp + w_o5*pH + b_o
    Params:
    w1_b, w1_s, w1_t, w1_p, b1 (5)
    w_o1, w_o2, w_o3, w_o4, w_o5, b_o (6)
    Total 11.
    
    Let's drop one direct term? Maybe pH is less important? Or temp?
    Or use a different structure.
    
    How about:
    H1 = tanh(w1_b*b + w1_s*s + w1_t*temp + w1_p*pH + b1)
    Out = w_o1*H1 + w_o2*b*s + w_o3*temp + w_o4*pH + b_o
    Interaction term b*s?
    Params:
    w1_b, w1_s, w1_t, w1_p, b1 (5)
    w_o1, w_o2, w_o3, w_o4, b_o (5)
    Total 10.
    
    This allows the network to learn a non-linear transformation of the inputs (H1)
    and add back specific linear/interaction terms. The Hard-Tanh provides the
    non-linearity. BFGS can optimize these 10 parameters.
    
    Implementation:
    p[0]=w1_b, p[1]=w1_s, p[2]=w1_t, p[3]=w1_p, p[4]=b1
    p[5]=w_o1, p[6]=w_o2, p[7]=w_o3, p[8]=w_o4, p[9]=b_o
    
    H1 = np.tanh(p[0]*b + p[1]*s + p[2]*temp + p[3]*pH + p[4])
    # Use np.tanh for smoothness, or Hard-Tanh?
    # np.tanh is smooth and bounded [-1, 1]. Good for BFGS.
    # Hard-Tanh is piecewise linear, gradients are 0 or 1. Can get stuck.
    # Let's use np.tanh.
    
    Out = p[5]*H1 + p[6]*(b*s) + p[7]*temp + p[8]*pH + p[9]
    
    Wait, b*s interaction might be large if b and s are large.
    b range [1, 5], s range [1, 40]. b*s max 200.
    This might cause scaling issues.
    
    Alternative:
    H1 = np.tanh(p[0]*b + p[1]*s + p[2]*temp + p[3]*pH + p[4])
    Out = p[5]*H1 + p[6]*b + p[7]*s + p[8]*temp + p[9]*pH
    This is 10 params.
    p[0..4] define the non-linear feature.
    p[5..9] define the output layer (weights for H1, b, s, temp, pH).
    This is a valid 1-hidden-layer MLP with 1 unit, where the output layer
    has direct connections to inputs (skip connections).
    This is strictly more powerful than the seed (which is just the linear part
    with p[5]=0, p[6..9] = seed weights, and p[0..4] irrelevant if p[5]=0).
    Actually, if p[5]=0, it reduces to linear model.
    This is a good design.
    """
    # Extract params
    w1_b, w1_s, w1_t, w1_p, b1 = params[0], params[1], params[2], params[3], params[4]
    w_o_h, w_o_b, w_o_s, w_o_t, w_o_p = params[5], params[6], params[7], params[8], params[9]
    
    # Hidden layer
    z = w1_b * b + w1_s * s + w1_t * temp + w1_p * pH + b1
    h = np.tanh(z)
    
    # Output layer
    return w_o_h * h + w_o_b * b + w_o_s * s + w_o_t * temp + w_o_p * pH

# EVOLVE-BLOCK-END
