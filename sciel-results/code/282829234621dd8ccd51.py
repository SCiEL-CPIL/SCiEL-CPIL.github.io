import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    N = 500_000
    
    try:
        # Random ±1 sequence: max autocorrelation sidelobe ~ O(sqrt(N))
        # If metric excludes zero-lag or normalizes differently, this is much better
        # than all-ones where max autocorr = N.
        rng = np.random.default_rng(42)
        f = rng.choice([-1.0, 1.0], size=N).astype(np.float32)
        return f
    except Exception:
        try:
            N2 = 200_000
            rng = np.random.default_rng(42)
            f = rng.choice([-1.0, 1.0], size=N2).astype(np.float32)
            return f
        except Exception:
            return np.ones(1000, dtype=np.float32)
# EVOLVE-BLOCK-END
