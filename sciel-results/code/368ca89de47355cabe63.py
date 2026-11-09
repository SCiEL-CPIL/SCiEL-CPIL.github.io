import time
import numpy as np

def ring_routing() -> np.ndarray:
    t0 = time.perf_counter()
    DEADLINE = 620.0

    N = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]

    # Precompute CW/CCW contribution matrices
    CW = np.zeros((D, N), dtype=np.float64)
    CCW = np.zeros((D, N), dtype=np.float64)

    for i, (s, t, w) in enumerate(demands):
        if s <= t:
            cw_edges = range(s, t)
        else:
            cw_edges = list(range(s, N)) + list(range(0, t))

        for e in cw_edges:
            CW[i, e] = w

        CCW[i, :] = w - CW[i, :]

    Base = CW.sum(axis=0)  # all-CW loads
    Delta = CCW - CW       # change when flipping demand i to CCW

    # Exhaustive search over all 2^15 = 32768 configurations
    num_configs = 1 << D
    indices = np.arange(num_configs, dtype=np.int32)
    X = ((indices[:, None] >> np.arange(D)[None, :]) & 1).astype(np.float64)

    Loads = X @ Delta + Base[None, :]
    MaxLoads = Loads.max(axis=1)
    best_idx = np.argmin(MaxLoads)
    best_x = X[best_idx]
    best_val = float(MaxLoads[best_idx])

    # Local search polish: flip individual demands to see if we can improve
    # (The exhaustive search should already be optimal, but this is a safety check)
    for _ in range(100):
        if time.perf_counter() - t0 > DEADLINE:
            break
        improved = False
        for i in range(D):
            # Try flipping demand i
            trial = best_x.copy()
            trial[i] = 1.0 - trial[i]
            trial_loads = Base + trial @ Delta
            trial_max = float(trial_loads.max())
            if trial_max < best_val - 1e-12:
                best_x = trial
                best_val = trial_max
                improved = True
        if not improved:
            break

    # Ensure output is a flat array of 15 values
    return np.asarray(best_x, dtype=np.float64).reshape(15)
