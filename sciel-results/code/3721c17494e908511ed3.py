"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
_T0 = time.time()
_DEADLINE = 630.0 - 30.0  # explicit wall-clock guard with safety margin


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def construct():
    """Return the 12 positive double-root locations for the refined
    Cohn-Goncalves Laguerre construction (m=12, n=1).

    Strategy: Use the zeros of the generalized Laguerre polynomial L_12^{(1)}(x)
    instead of the standard L_12(x). The generalized Laguerre polynomial
    L_n^{(α)}(x) has orthogonality with weight x^α e^{-x}. For α=1, the roots
    are shifted to larger values compared to α=0, which may better align with
    the Fourier transform's decay profile for the uncertainty bound.
    """
    if _time_left() <= 1.0:
        return [4.0 * (i + 1) for i in range(12)]
    try:
        # Compute zeros of generalized Laguerre polynomial L_12^{(1)}(x)
        # Using numpy's laggauss which computes for α=0, so we need to use
        # scipy or manual computation for α=1
        
        # Alternative: use scipy.special.roots_genlaguerre if available
        from scipy.special import roots_genlaguerre
        roots, _ = roots_genlaguerre(12, 1)
        roots = [float(x) for x in roots]
        roots.sort()
        
        # Verify constraints
        if (len(roots) == 12
                and all(r > 0.0 for r in roots)
                and all(b > a for a, b in zip(roots, roots[1:]))):
            return roots
    except Exception:
        pass
    
    # Fallback: if scipy not available or fails, use standard Laguerre nodes
    try:
        nodes, _ = np.polynomial.laguerre.laggauss(12)
        roots = [float(x) for x in nodes]
        roots.sort()
        if (len(roots) == 12
                and all(r > 0.0 for r in roots)
                and all(b > a for a, b in zip(roots, roots[1:]))):
            return roots
    except Exception:
        pass
    
    return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
