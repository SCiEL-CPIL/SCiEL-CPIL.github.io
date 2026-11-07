import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).flatten()
        y = np.asarray(y, dtype=np.float64).flatten()
        p = params
        
        # Precompute trigonometric values
        # Frequencies scaled to [-1, 1] domain.
        # Using pi * x means the first harmonic has a half-period of 2 (covering the domain).
        
        c1x = np.cos(np.pi * x)
        s1x = np.sin(np.pi * x)
        c1y = np.cos(np.pi * y)
        s1y = np.sin(np.pi * y)
        
        c2x = np.cos(2.0 * np.pi * x)
        s2x = np.sin(2.0 * np.pi * x)
        c2y = np.cos(2.0 * np.pi * y)
        s2y = np.sin(2.0 * np.pi * y)
        
        # Interaction term
        c_inter = np.cos(np.pi * (x + y))
        
        basis = np.column_stack((
            np.ones_like(x),
            c1x,
            c1y,
            s1x,
            s1y,
            c2x,
            c2y,
            s2x,
            s2y,
            c_inter
        ))
        
        pred = basis @ p
        return pred
