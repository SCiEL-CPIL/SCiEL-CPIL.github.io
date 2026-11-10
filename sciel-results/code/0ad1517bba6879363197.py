import numpy as np
import time

_start_time = None

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    if n == 1:
        return 0

    # Safety check
    if time.time() - _start_time > 130:
        w = np.asarray(remaining_weights, dtype=float)
        v = np.asarray(remaining_values, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v / w
        densities[np.where(w == 0)] = 0
        return int(np.argmax(densities))

    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    c = float(remaining_capacity)
    
    # Handle zero/negative weights
    min_w = float(np.min(w))
    if min_w <= 0:
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v / np.where(w > 0, w, 1.0)
        densities[np.where(w == 0)] = float('inf')
        return int(np.argmax(densities))
    
    # Choose scale for DP
    scale = max(1, min(200, 10000 // max(1, int(c))))
    C_scaled = int(c * scale)
    if C_scaled < 1:
        C_scaled = 1
    w_scaled = np.round(w * scale).astype(int)
    w_scaled = np.maximum(w_scaled, 1)
    w_scaled = np.minimum(w_scaled, C_scaled)
    
    # Compute full DP: dp[j] = max value achievable with capacity j using all items
    dp = np.zeros(C_scaled + 1, dtype=float)
    for i in range(n):
        wi = w_scaled[i]
        vi = v[i]
        if wi <= C_scaled:
            for j in range(C_scaled, wi - 1, -1):
                candidate = dp[j - wi] + vi
                if candidate > dp[j]:
                    dp[j] = candidate
    
    # For each candidate item i, estimate total value if picked first:
    # est[i] = v[i] + dp[c_scaled - w_scaled[i]]
    # This is an upper bound (since dp includes item i itself potentially)
    # but serves as a good ranking heuristic
    
    best_score = -1.0
    best_idx = -1
    
    # Compute densities for tie-breaking
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v / w
    densities[np.where(w == 0)] = 0
    
    for i in range(n):
        wi = w_scaled[i]
        if wi > C_scaled:
            continue
        remaining_cap = C_scaled - wi
        est_total = v[i] + dp[remaining_cap]
        # Add small density bonus for tie-breaking
        score = est_total + 1e-9 * densities[i]
        if score > best_score:
            best_score = score
            best_idx = i
    
    if best_idx == -1:
        best_idx = int(np.argmax(densities))
    
    return int(best_idx)
