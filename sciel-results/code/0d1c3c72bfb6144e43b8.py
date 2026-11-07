import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale inputs from [0.5, 2] to [-1, 1] to improve conditioning
    # u = 2*x - 1.5, range [2*0.5-1.5, 2*2-1.5] = [-0.5, 2.5]?
    # Wait.
    # x in [0.5, 2].
    # Map to [0, 1]: x_scaled = (x - 0.5) / 1.5. Range [0, 1].
    # Map to [-1, 1]: u = 2 * x_scaled - 1 = 2*(x-0.5)/1.5 - 1 = (2x - 1)/1.5 - 1 = (2x - 2.5)/1.5.
    # Let's just use simple scaling to [0, 1] or [-1, 1].
    # u = (x - 0.5) / 1.5 -> [0, 1]
    # v = (y - 0.5) / 1.5 -> [0, 1]
    # w = (z - 0.5) / 1.5 -> [0, 1]
    
    u = (x - 0.5) / 1.5
    v = (y - 0.5) / 1.5
    w = (z - 0.5) / 1.5
    
    p = params
    
    # Additive Cubic Basis
    # 1. Constant
    # 2. u
    # 3. v
    # 4. w
    # 5. u^2
    # 6. v^2
    # 7. w^2
    # 8. u^3
    # 9. v^3
    # 10. w^3
    
    t1 = np.ones_like(u)
    t2 = u
    t3 = v
    t4 = w
    t5 = u * u
    t6 = v * v
    t7 = w * w
    t8 = u * u * u
    t9 = v * v * v
    t10 = w * w * w
    
    return (p[0] * t1 + 
            p[1] * t2 + 
            p[2] * t3 + 
            p[3] * t4 + 
            p[4] * t5 + 
            p[5] * t6 + 
            p[6] * t7 + 
            p[7] * t8 + 
            p[8] * t9 + 
            p[9] * t10)
