"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

# Cache for basis matrix to avoid recomputation during optimization steps
_cache_key = None
_cache_B = None

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    global _cache_key, _cache_B
    
    # Check if we need to recompute the basis
    # Using a tuple of shapes and a hash of the first few elements as a cheap key
    # or just array_equal since N is small (200)
    if _cache_key is None or not (np.array_equal(x, _cache_key[0]) and np.array_equal(y, _cache_key[1])):
        _cache_key = (x, y)
        
        # Basis functions
        # 1. Constant
        f0 = np.ones_like(x)
        # 2. Linear x
        f1 = x
        # 3. Linear y
        f2 = y
        # 4. Quadratic x
        f3 = x**2
        # 5. Quadratic y
        f4 = y**2
        # 6. Interaction xy
        f5 = x * y
        # 7. Sin x
        f6 = np.sin(x)
        # 8. Sin y
        f7 = np.sin(y)
        # 9. Cos x
        f8 = np.cos(x)
        # 10. Cos y
        f9 = np.cos(y)
        
        _cache_B = np.column_stack((f0, f1, f2, f3, f4, f5, f6, f7, f8, f9))
    
    return _cache_B @ params
# EVOLVE-BLOCK-END
