import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression for a nonlinear oscillator using a Duffing-van der Pol hybrid model.
    
    This function models the acceleration (or force) of a driven nonlinear oscillator.
    The structure is:
    a = damping_term + restoring_term + driving_term
    
    Specifically:
    a = p0 * v * (1 - p1 * v^2)       # Van der Pol style self-excitation/damping
      + p2 * x + p3 * x^3             # Duffing style nonlinear spring
      + p4 * cos(p5 * t)              # Driving force (approximated via phase if t not available, 
                                       # but here we only have x, v. 
                                       # Wait, the task says inputs are x, v. 
                                       # Usually, symbolic regression for ODEs predicts the next state or the derivative.
                                       # The target range [-0.1, 0.1] suggests it might be predicting acceleration or a small change.
                                       # Let's assume the target is the acceleration a = x''.
                                       # A standard Duffing equation: x'' + delta*x' + alpha*x + beta*x^3 = gamma*cos(omega*t)
                                       # Since we don't have t, we must rely on x and v to infer the state or the equation is autonomous?
                                       # If it's autonomous, driving term is 0 or dependent on state?
                                       # Or maybe the "driving" is implicit in the initial conditions?
                                       # Let's look at the inspiration: "driven Duffing oscillator".
                                       # If t is not an input, we cannot explicitly use cos(omega*t) unless we can estimate phase from x,v.
                                       # However, often in these tasks, if t is not provided, the system might be considered autonomous or the target is the vector field.
                                       # Let's try a general nonlinear function that captures the essential physics:
                                       # Nonlinear damping: v * (p0 + p1 * v^2)
                                       # Nonlinear stiffness: x * (p2 + p3 * x^2 + p4 * v^2)  # Velocity-dependent stiffness?
                                       # Plus some cross terms.
    
    Let's refine the structure to use all 10 parameters meaningfully without t:
    a = p0 * v + p1 * v^3                    # Damping
      + p2 * x + p3 * x^3                    # Stiffness
      + p4 * x * v                           # Coupling
      + p5 * cos(p6 * x + p7 * v)            # Periodic nonlinearity in state space?
      + p8 * sin(p9 * x * v)                 # Another harmonic
    
    Actually, the prompt suggests: 
    `v * tanh(p0) - x * (p1 + p2 * tanh(p3 * x^2 + p4 * v))`
    This uses 5 params. We have 10.
    Let's expand this physics-informed structure.
    
    Model:
    a = damping + restoring + nonlinear_coupling
    
    1. Damping: p0 * v * (1 + p1 * v^2)  (p0, p1)
    2. Restoring: -x * (p2 + p3 * x^2)   (p2, p3) -> Note sign, usually -kx
    3. Velocity-dependent stiffness: -x * p4 * v^2 (p4)
    4. Cross term: p5 * x * v (p5)
    5. Higher order nonlinearity: p6 * x^2 * v (p6)
    6. Saturation/Softening: p7 * x * tanh(p8 * x) (p7, p8)
    7. Bias/Offset: p9 (p9)
    
    Let's check the scale. x in [-0.5, 0.5], v in [-0.2, 0.3].
    Target in [-0.1, 0.1].
    
    Let's try:
    a = p0 * v + p1 * v^3 
      + p2 * x + p3 * x^3 
      + p4 * x * v 
      + p5 * x^2 * v 
      + p6 * v * tanh(p7 * x) 
      + p8 * x * tanh(p9 * v)
      
    This is a rich, physically motivated basis that is smooth and bounded (via tanh) where appropriate, and polynomial elsewhere.
    """
    p = params
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = p
    
    # Compute terms
    # Damping terms
    damp_lin = p0 * v
    damp_cub = p1 * v**3
    
    # Restoring terms
    rest_lin = p2 * x
    rest_cub = p3 * x**3
    
    # Coupling terms
    coup_xv = p4 * x * v
    coup_x2v = p5 * x**2 * v
    
    # Nonlinear modulation terms (using tanh for stability/saturation)
    mod1 = p6 * v * np.tanh(p7 * x)
    mod2 = p8 * x * np.tanh(p9 * v)
    
    # Sum
    result = damp_lin + damp_cub + rest_lin + rest_cub + coup_xv + coup_x2v + mod1 + mod2
    
    return result
