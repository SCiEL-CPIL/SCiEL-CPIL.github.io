import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Parametric limit-cycle oscillator model.
    
    a = -k(x)*x - c(v)*v + F(t) + coupling + offset
    
    where:
    k(x) = p0 + p1*tanh(p2*x)      (position-dependent stiffness)
    c(v) = p3 + p4*tanh(p5*v)      (velocity-dependent damping)
    F(t) = p6*cos(t) + p7*sin(t)   (periodic driving)
    coupling = p8*x*v              (parametric coupling)
    offset = p9
    
    All 10 parameters are fitted.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Stiffness term: k(x) * x
    # k(x) is a smooth, bounded function of x
    k_x = p0 + p1 * np.tanh(p2 * x)
    force_x = -k_x * x
    
    # Damping term: c(v) * v
    # c(v) is a smooth, bounded function of v
    c_v = p3 + p4 * np.tanh(p5 * v)
    force_v = -c_v * v
    
    # Driving force: F(t)
    force_t = p6 * np.cos(t) + p7 * np.sin(t)
    
    # Coupling term
    force_couple = p8 * x * v
    
    # Total acceleration
    return force_x + force_v + force_t + force_couple + p9
