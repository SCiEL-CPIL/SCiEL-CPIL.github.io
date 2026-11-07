import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a fully flexible 4th-order polynomial response surface.
    
    This model uses a linear combination of all monomials up to degree 4 in the 
    normalized inputs. The 10 parameters are fitted by the evaluator using BFGS.
    This approach avoids the rigid structure of RBF centers and allows the 
    optimizer to discover complex interactions between temperature, pH, and 
    nutrients directly.
    
    Inputs are normalized to [0, 1] to improve numerical stability during optimization.
    """
    # Normalize inputs to [0, 1] based on known ranges
    # b: [1, 5] -> [0, 1]
    # s: [1, 5] -> [0, 1]
    # temp: [1, 40] -> [0, 1]
    # pH: [2, 12] -> [0, 1]
    
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    temp_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    # Clip to [0, 1] to handle any minor numerical drift or out-of-bounds inputs
    b_norm = np.clip(b_norm, 0.0, 1.0)
    s_norm = np.clip(s_norm, 0.0, 1.0)
    temp_norm = np.clip(temp_norm, 0.0, 1.0)
    ph_norm = np.clip(ph_norm, 0.0, 1.0)
    
    # Construct a 4th-order polynomial feature vector.
    # We select 10 specific monomials that capture the most likely biological dynamics:
    # 1. Constant term
    # 2. Linear terms for each input
    # 3. Key quadratic terms (self-interactions and cross-interactions)
    # 4. Key cubic terms (e.g., temperature effect often has a peak, so cubic is useful)
    # 5. Key quartic terms (to capture sharp boundaries or high-order effects)
    
    # Feature 1: Constant
    f1 = np.ones_like(b_norm)
    
    # Feature 2: b (nutrient 1)
    f2 = b_norm
    
    # Feature 3: s (nutrient 2)
    f3 = s_norm
    
    # Feature 4: temp (temperature)
    f4 = temp_norm
    
    # Feature 5: pH
    f5 = ph_norm
    
    # Feature 6: temp^2 (quadratic temperature effect, often parabolic)
    f6 = temp_norm ** 2
    
    # Feature 7: pH^2 (quadratic pH effect, often parabolic)
    f7 = ph_norm ** 2
    
    # Feature 8: b * s (nutrient interaction)
    f8 = b_norm * s_norm
    
    # Feature 9: temp * pH (temperature-pH interaction)
    f9 = temp_norm * ph_norm
    
    # Feature 10: temp^3 (cubic temperature effect to better capture the peak)
    f10 = temp_norm ** 3
    
    # Stack features into a matrix of shape (N, 10)
    features = np.stack([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10], axis=-1)
    
    # Compute prediction as a linear combination of features and parameters
    # params is of shape (10,)
    y = np.sum(features * params[None, :], axis=1)
    
    return y
