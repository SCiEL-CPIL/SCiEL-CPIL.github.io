import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        params = np.asarray(params, dtype=np.float64).reshape(-1)
        
        # Map x to [-1, 1] for Chebyshev
        # x in [0.1, 6]
        # t = (2x - (min+max)) / (max-min)
        x_min, x_max = 0.1, 6.0
        t = (2.0 * x - (x_min + x_max)) / (x_max - x_min)
        
        # Chebyshev polynomials T_0 to T_6
        # T_0 = 1
        # T_1 = t
        # T_2 = 2t^2 - 1
        # T_3 = 4t^3 - 3t
        # T_4 = 8t^4 - 8t^2 + 1
        # T_5 = 16t^5 - 20t^3 + 5t
        # T_6 = 32t^6 - 48t^4 + 18t^2 - 1
        
        T1 = t
        T2 = 2 * t**2 - 1
        T3 = 4 * t**3 - 3 * t
        T4 = 8 * t**4 - 8 * t**2 + 1
        T5 = 16 * t**5 - 20 * t**3 + 5 * t
        T6 = 32 * t**6 - 48 * t**4 + 18 * t**2 - 1
        
        # Auxiliary
        inv_x = 1.0 / x
        exp_x = np.exp(x)
        log_x = np.log(x)
        
        # Build Basis
        # Column 0: Bias
        # Columns 1-6: Chebyshev T1-T6
        # Column 7: 1/x
        # Column 8: exp(x)
        # Column 9: log(x)
        
        basis = np.column_stack([
            np.ones_like(x),
            T1, T2, T3, T4, T5, T6,
            inv_x,
            exp_x,
            log_x
        ])
        
        # Standardize columns 1-9
        # Keep column 0 as is (bias)
        
        means = basis[:, 1:].mean(axis=0)
        stds = basis[:, 1:].std(axis=0)
        
        # Avoid division by zero
        stds[stds < 1e-10] = 1.0
        
        basis_std = np.copy(basis)
        basis_std[:, 1:] = (basis[:, 1:] - means) / stds
        
        pred = basis_std @ params
        
        return pred
