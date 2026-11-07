import numpy as np

def equation(x, y, params):
    # p0..p3: First exponential term: w * exp(a*x + b*y + c)
    w1, a1, b1, c1 = params[0:4]
    # p4..p7: Second exponential term: w * exp(a*x + b*y + c)
    w2, a2, b2, c2 = params[4:8]
    # p8, p9: Linear baseline
    m, b = params[8:10]
    
    term1 = w1 * np.exp(a1 * x + b1 * y + c1)
    term2 = w2 * np.exp(a2 * x + b2 * y + c2)
    linear = m * x + b
    
    return term1 + term2 + linear
