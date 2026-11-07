import numpy as np
def equation(b, s, temp, pH, params):
        p = params
        
        # Normalize to [-1, 1]
        b_norm = 2 * (b - 1.0) / (4.998 - 1.0) - 1
        s_norm = 2 * (s - 1.0) / (4.999 - 1.0) - 1
        temp_norm = 2 * (temp - 1.002) / (39.98 - 1.002) - 1
        ph_norm = 2 * (pH - 2.0) / (12.0 - 2.0) - 1
        
        # Clip to [-1, 1]
        b_norm = np.clip(b_norm, -1, 1)
        s_norm = np.clip(s_norm, -1, 1)
        temp_norm = np.clip(temp_norm, -1, 1)
        ph_norm = np.clip(ph_norm, -1, 1)
        
        # Legendre Polynomials
        # L0 = 1
        # L1 = x
        # L2 = 0.5 * (3*x^2 - 1)
        
        b_L1 = b_norm
        s_L1 = s_norm
        T_L1 = temp_norm
        pH_L1 = ph_norm
        
        b_L2 = 0.5 * (3 * b_norm**2 - 1)
        s_L2 = 0.5 * (3 * s_norm**2 - 1)
        T_L2 = 0.5 * (3 * temp_norm**2 - 1)
        pH_L2 = 0.5 * (3 * ph_norm**2 - 1)
        
        result = (p[0] 
                  + p[1] * b_L1 
                  + p[2] * s_L1 
                  + p[3] * T_L1 
                  + p[4] * pH_L1 
                  + p[5] * b_L2 
                  + p[6] * s_L2 
                  + p[7] * T_L2 
                  + p[8] * pH_L2 
                  + p[9] * T_L1 * pH_L1)
                  
        return result
