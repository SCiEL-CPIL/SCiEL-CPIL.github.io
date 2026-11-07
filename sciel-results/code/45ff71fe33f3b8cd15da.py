import numpy as np

def equation(b, s, temp, pH, params):
    """
    Flexible additive model with interaction terms.
    Uses a combination of linear terms, quadratic terms, and interaction terms
    to allow the optimizer to fit complex non-linear relationships.
    
    Inputs:
    b: population density
    s: substrate concentration
    temp: temperature
    pH: pH level
    
    Params (10 constants):
    p0: bias
    p1: linear b
    p2: linear s
    p3: linear temp
    p4: linear pH
    p5: quadratic b
    p6: quadratic s
    p7: quadratic temp
    p8: interaction b*s
    p9: interaction temp*pH
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Linear terms
    y = p0 + p1*b + p2*s + p3*temp + p4*pH
    
    # Quadratic terms (allowing for curvature in each dimension)
    y += p5*b**2
    y += p6*s**2
    y += p7*temp**2
    
    # Interaction terms (allowing for coupled effects)
    y += p8*b*s
    y += p9*temp*pH
    
    return y
