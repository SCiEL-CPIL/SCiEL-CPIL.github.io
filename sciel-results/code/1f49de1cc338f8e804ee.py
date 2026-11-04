"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.special import roots_genlaguerre

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 540.0  # safety margin before 600s

    m = 12
    n = 1

    # Primary: roots of the ordinary Laguerre polynomial L_m(x).
    # These are the canonical optimal double-root locations for the
    # Cohn-Goncalves Fourier sign-uncertainty construction with weight e^{-x}.
    try:
        res = roots_genlaguerre(m, 0.0)
        if isinstance(res, tuple):
            x = np.asarray(res[0], dtype=float)
        else:
            x = np.asarray(res, dtype=float)
        x = np.sort(x)
        x = x[x > 0]
        if len(x) == m and np.all(np.isfinite(x)):
            return [float(r) for r in x]
    except Exception:
        pass

    # Secondary: optimize a generalized-Laguerre parameter alpha using a
    # proxy for the C4 bound. The C4 bound for the double-root construction
    # is dominated by the tail behavior; we minimize a combination of the
    # largest root and the sum of roots (proxy for the moment that controls C4).
    best_roots = None
    best_proxy = np.inf
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, -0.25, -0.5]:
        if time.time() > deadline:
            break
        try:
            res = roots_genlaguerre(m, alpha)
            if isinstance(res, tuple):
                x = np.asarray(res[0], dtype=float)
            else:
                x = np.asarray(res, dtype=float)
            x = np.sort(x)
            x = x[x > 0]
            if len(x) != m or not np.all(np.isfinite(x)):
                continue
            # Proxy: C4 bound ~ (sum of roots) * (largest root) as a rough
            # measure of the moment and tail extent.
            proxy = float(np.sum(x) * x[-1])
            if proxy < best_proxy:
                best_proxy = proxy
                best_roots = x
        except Exception:
            continue

    if best_roots is not None and len(best_roots) == m:
        return [float(r) for r in best_roots]

    # Final fallback: seed
    return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
