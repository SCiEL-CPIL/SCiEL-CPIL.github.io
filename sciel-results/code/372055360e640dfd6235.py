import numpy as np
from scipy.optimize import differential_evolution

# Pre-compute grid for fast integration
L = 15.0
N = 4000
x_grid = np.linspace(-L, L, N)
dx = x_grid[1] - x_grid[0]

def H0(x):
    return np.ones_like(x)

def H4(x):
    return x**4 - 6*x**2 + 3

def H8(x):
    return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105

def H12(x):
    return x**12 - 60*x**10 + 1260*x**8 - 12600*x**6 + 45360*x**4 - 55440*x**2 + 10395

def get_g(a0, a4, a8):
    a12 = -(a0 + 3*a4 + 105*a8) / 10395.0
    
    # Calculate g(x)
    poly_g = a0*H0(x_grid) + a4*H4(x_grid) + a8*H8(x_grid) + a12*H12(x_grid)
    g = poly_g * np.exp(-x_grid**2 / 2.0)
    
    return g

def objective(params):
    a0, a4, a8 = params
    
    g = get_g(a0, a4, a8)
    
    # Compute I2 and I4
    I2 = np.sum(g**2) * dx
    I4 = np.sum(g**4) * dx
    
    if I2 < 1e-12:
        return 1e100
        
    # Assuming g_hat = g due to even degrees (Fourier eigenvalue 1)
    # Ratio = (I4_g * I4_ghat)^(1/4) / I2 = I4^(1/2) / I2
    # Minimize Ratio^4 = I4^2 / I2^4
    val = (I4 * I4) / (I2**4)
    return val

def construct():
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    res = differential_evolution(objective, bounds, seed=42, maxiter=200, tol=1e-10, polish=True)
    
    best_a0, best_a4, best_a8 = res.x
    
    return np.array([best_a0, best_a4, best_a8])
