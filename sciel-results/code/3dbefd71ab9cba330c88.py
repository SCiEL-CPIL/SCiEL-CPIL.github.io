import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # params length 10
    # p[0]: const
    # p[1]: x coeff
    # p[2]: y coeff
    # p[3]: alpha (width)
    # p[4]: w_center
    # p[5]: w_ne
    # p[6]: w_nw
    # p[7]: w_se
    # p[8]: w_sw
    # p[9]: extra? maybe w_edge_mid? or just ignore? 
    # Better to use all 10.
    
    # Let's define 6 Gaussians to use 10 params:
    # p[0]: const
    # p[1]: x
    # p[2]: y
    # p[3]: alpha
    # p[4]: w0 (0,0)
    # p[5]: w1 (0.5, 0.5)
    # p[6]: w2 (-0.5, -0.5)
    # p[7]: w3 (0.5, -0.5)
    # p[8]: w4 (-0.5, 0.5)
    # p[9]: w5 (0, 0.8) ?
    
    # This seems arbitrary.
    
    # Alternative: Use p[9] for a second width or a quadratic term.
    # Let's add a quadratic term p[9] * (x**2 + y**2) to capture radial curvature.
    
    alpha = params[3]
    # Ensure alpha is positive to avoid explosion? 
    # BFGS might try negative alpha. 
    # exp(-alpha * dist^2). If alpha < 0, it explodes.
    # We should clamp alpha or use abs(alpha)?
    # The task says "closed-form expression". 
    # Using abs() is non-differentiable at 0, but BFGS can handle it? 
    # Better to use max(alpha, 1e-6) or similar? 
    # Or just assume BFGS finds positive alpha? 
    # Starting at 1, it's likely to stay positive if the function is localized.
    # If the function is global, alpha might go to 0.
    
    # Let's use a safe alpha: 
    a = np.maximum(params[3], 0.1)
    
    w0 = params[0]
    wx = params[1]
    wy = params[2]
    w_c = params[4]
    w_ne = params[5]
    w_nw = params[6]
    w_se = params[7]
    w_sw = params[8]
    w_rad = params[9]
    
    r2 = x**2 + y**2
    
    pred = w0 + wx * x + wy * y + w_rad * r2
    
    # Gaussians
    pred += w_c * np.exp(-a * r2)
    pred += w_ne * np.exp(-a * ((x-0.5)**2 + (y-0.5)**2))
    pred += w_nw * np.exp(-a * ((x+0.5)**2 + (y+0.5)**2))
    pred += w_se * np.exp(-a * ((x-0.5)**2 + (y-0.5)**2))
    pred += w_sw * np.exp(-a * ((x+0.5)**2 + (y-0.5)**2))
    
    return pred
