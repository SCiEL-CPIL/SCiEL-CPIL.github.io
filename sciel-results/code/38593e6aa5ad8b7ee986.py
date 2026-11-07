import numpy as np
import time

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 290.0

    try:
        prizes = np.asarray(prizes, dtype=float)
        weights = np.asarray(weights, dtype=float)
        
        # Basic shape check; if wrong, fall through to exception handler
        if prizes.shape != (5, 100) or weights.shape != (5, 100):
            raise ValueError("Unexpected shape")

        n_constraints, n_items = 5, 100
        capacity = 25.0
        epsilon = 1e-9

        # 1. Ratios
        safe_w = weights + epsilon
        ratios = prizes / safe_w
        
        # 2. Harmonic Mean of Ratios
        # This is the core "efficiency" metric.
        inv_ratios = 1.0 / (ratios + epsilon)
        sum_inv = np.sum(inv_ratios, axis=0)
        h_mean = n_constraints / (sum_inv + epsilon)
        
        # 3. Geometric Slack
        # Reward items that leave more room.
        slack = 1.0 - weights / capacity
        slack = np.clip(slack, epsilon, 1.0)
        log_slack = np.log(slack)
        mean_log_slack = np.mean(log_slack, axis=0)
        geo_slack = np.exp(mean_log_slack)
        
        # 4. Composite Base Score
        # Combine efficiency (harmonic mean) and slack (geometric mean)
        # The 1.5 exponent on h_mean and 0.5 on geo_slack was found to be effective
        # in amplifying the influence of cross-constraint efficiency.
        score = np.power(h_mean + epsilon, 1.5) * np.power(geo_slack + epsilon, 0.5)
        
        # 5. Competitive Exclusion / Opportunity Cost Component
        # This idea from Inspiration 2 adds a "dominance" factor.
        # Items that have a high share of the total prize in a constraint are "dominant".
        # We calculate a competitive score: prize_share * ratio.
        total_prize_per_constraint = np.sum(prizes, axis=1)  # (5,)
        prize_share = prizes / (total_prize_per_constraint[:, np.newaxis] + epsilon)  # (5, 100)
        
        # Weighted share: efficiency modulated by dominance
        weighted_share = ratios * prize_share  # (5, 100)
        
        # Geometric mean of the competitive scores across constraints
        log_ws = np.log(weighted_share + epsilon)
        geo_mean_log = np.mean(log_ws, axis=0)
        h_competitive = np.exp(geo_mean_log)  # (100,)
        
        # 6. Blend the two components
        # We need to normalize them to be on comparable scales before blending.
        def normalize_positive(arr):
            arr = np.where(arr > 0, arr, epsilon)
            mn = np.min(arr)
            mx = np.max(arr)
            if mx > mn:
                return (arr - mn) / (mx - mn)
            return np.ones_like(arr)
        
        h1 = normalize_positive(score)
        h2 = normalize_positive(h_competitive)
        
        # Blend: 0.7 weight to the proven harmonic/slack score, 0.3 to competitive
        alpha = 0.7
        blended = alpha * h1 + (1 - alpha) * h2
        
        # 7. Bottleneck Penalty (from Current Program)
        # Penalize items that contribute heavily to the most constrained resources.
        loads = np.sum(weights, axis=1)
        pressure = np.maximum(0.0, loads - capacity) / capacity
        item_pressure = np.sum(weights * pressure, axis=0)
        penalty_factor = 1.0 + item_pressure
        blended = blended / penalty_factor
        
        # 8. Final Sharpening
        # Apply a power to sharpen the differences, making ACO more selective.
        # 1.2 is a mild sharpening.
        heuristic = np.power(blended + epsilon, 1.2)
        
        # 9. Cleanup
        heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, epsilon)
        heuristic = heuristic.reshape(n_items)
        
        if time.time() > deadline:
            return heuristic
            
        return heuristic

    except Exception:
        return np.full((100,), 1e-9)
