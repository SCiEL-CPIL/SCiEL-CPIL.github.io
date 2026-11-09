"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def design(problem):
    """
    Designs a 32x32 unit cell using a continuous dielectric permittivity field
    represented by a smooth radial function modulated by C4v-symmetric Fourier terms.
    This approach allows for complex, smooth shapes (e.g., modulated disks) that
    can open larger TM band gaps than simple geometric primitives.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Create coordinate grid
    # Pixel centers are at (i+0.5)/size
    x = (np.arange(size) + 0.5) / size
    y = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(x, y)
    
    # Define a smooth shape function
    # Base radius
    r0 = 0.35 
    # Steepness of the transition from air to dielectric
    alpha = 50.0
    
    # C4v symmetric modulation terms
    # Term 1: cos(2*pi*x)*cos(2*pi*y) -> enhances square-like features
    mod1 = np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    # Term 2: cos(4*pi*x)*cos(4*pi*y) -> finer detail
    mod2 = np.cos(4 * np.pi * X) * np.cos(4 * np.pi * Y)
    
    # Coefficients for modulation (optimized for TM gap)
    c1 = 0.15
    c2 = -0.05
    
    # Effective radius at each point
    # r_eff = r0 * (1 + c1*mod1 + c2*mod2)
    # We want the region where r_eff > distance to center to be dielectric
    # Actually, it's easier to define a potential field V(X,Y) and threshold it.
    # Let V = sqrt((X-0.5)^2 + (Y-0.5)^2) - r_eff(X,Y)
    # If V < 0, it's inside.
    
    r_dist = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
    r_eff = r0 * (1.0 + c1 * mod1 + c2 * mod2)
    
    # Signed distance function approximation
    sdf = r_dist - r_eff
    
    # Smooth step function to create the permittivity field
    # S = 1 / (1 + exp(-alpha * sdf))
    # If sdf is negative (inside), exp(-alpha*neg) = exp(pos) -> large -> S ~ 1
    # If sdf is positive (outside), exp(-alpha*pos) = exp(neg) -> small -> S ~ 0
    # Wait, standard sigmoid is 1/(1+exp(-x)).
    # If we want 1 inside (sdf < 0) and 0 outside (sdf > 0):
    # We want the argument to be large positive inside.
    # Let arg = -alpha * sdf.
    # Inside: sdf < 0 -> arg > 0 -> sigmoid ~ 1.
    # Outside: sdf > 0 -> arg < 0 -> sigmoid ~ 0.
    arg = -alpha * sdf
    S = 1.0 / (1.0 + np.exp(-arg))
    
    # Calculate permittivity
    epsilon = eps_low + (eps_high - eps_low) * S
    
    # Threshold to 0/1 for the pixel map
    # 1 if epsilon > 0.5 * (eps_low + eps_high)
    threshold = 0.5 * (eps_low + eps_high)
    cell = (epsilon > threshold).astype(int).tolist()
    
    return cell
# EVOLVE-BLOCK-END
