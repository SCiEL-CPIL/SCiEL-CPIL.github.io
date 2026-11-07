import numpy as np

def equation(b, s, temp, pH, params):
        # Unpack
        p0 = params[0]
        p1 = params[1]
        p2 = params[2]
        p3 = params[3]
        p4 = params[4]
        p5 = params[5]
        p6 = params[6]
        p7 = params[7]
        p8 = params[8]
        p9 = params[9]
        
        # Positive transforms
        # Using softplus: log(1 + exp(x))
        # To avoid overflow in exp for large x, use np.logaddexp(0, x)
        Ks = np.logaddexp(0, p1)
        sigma_T = np.logaddexp(0, p3)
        sigma_pH = np.logaddexp(0, p5)
        Kb = np.logaddexp(0, p6)
        
        # Substrate term
        # s is in [1, 5]. Ks > 0.
        term_s = (s / (s + Ks)) ** p8
        
        # Temperature term
        # temp in [1, 40]
        diff_T = temp - p2
        term_T = np.exp(- (diff_T ** 2) / (2 * sigma_T ** 2))
        
        # pH term
        # pH in [2, 12]
        diff_pH = pH - p4
        term_pH = np.exp(- (diff_pH ** 2) / (2 * sigma_pH ** 2))
        
        # Density term
        # b in [1, 5]
        term_b = 1.0 / (1.0 + (b / Kb) ** 2)
        
        # Combine
        y = p0 * term_s * term_T * term_pH * term_b + p7
        
        # Optional: use p9 as a global scale or something? 
        # Let's make p9 a small additive correction or power for temp?
        # Let's use p9 to adjust the width of the density inhibition?
        # Or maybe p9 is a linear term for temp? 
        # Let's stick to 8 active params and set p9 to 0 or use it for a slight flexibility.
        # Actually, let's use p9 to modulate the substrate power? No, p8 is there.
        # Let's use p9 to add a small linear component to temperature to help if the Gaussian is too restrictive?
        # y = p0 * ... + p9 * temp ? 
        # Mixing multiplicative and additive can be tricky.
        # Let's keep it purely multiplicative + offset for interpretability and stability.
        # I will use p9 as a scale for the whole thing? No, p0 is scale.
        # Let's leave p9 unused or set it to 0 effectively, or use it for a very subtle correction.
        # Better: Use p9 to control the exponent of the density term.
        term_b = 1.0 / (1.0 + (b / Kb) ** (1.0 + p9**2)) # p9 adjusts shape
        
        return p0 * term_s * term_T * term_pH * term_b + p7
