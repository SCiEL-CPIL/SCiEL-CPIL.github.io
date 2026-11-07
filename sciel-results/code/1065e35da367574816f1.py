import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Heuristic matrix for ant colony orienteering.
    
    Combines prize magnitude, detour cost, and remaining budget slack.
    The heuristic favors edges that lead to high-prize nodes while
    preserving enough path budget to return to the origin.
    
    eta[i,j] = (prize[j] + eps)^alpha * slack[i,j]^gamma / (detour[i,j] + eps)^beta
    where:
      - detour[i,j] = d[i,j] + d[j,0]
      - slack[i,j] = max(0, max_len - detour[i,j])
    
    This blends the direct prize/detour ratio with a slack-weighted
    component to better respect the feasibility constraint.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    alpha = 2.0
    beta = 2.0
    gamma = 1.0
    eps = 1e-9
    
    # detour[i, j] = distance_matrix[i, j] + distance_matrix[j, 0]
    # distance_matrix[j, 0] is shape (N,), broadcast along rows
    detour = distance_matrix + distance_matrix[:, 0][None, :]
    
    # slack[i, j] = max_len - detour[i, j], clipped to non-negative
    slack = np.maximum(max_len - detour, 0.0)
    
    # prize_term: (N,) broadcast to (N, N) along axis 0
    prize_term = (prizes + eps) ** alpha
    
    # detour_term: (N, N)
    detour_term = (detour + eps) ** beta
    
    # slack_term: (N, N)
    slack_term = slack ** gamma
    
    # Combine: prize^alpha * slack^gamma / detour^beta
    # This gives high values for high-prize nodes with large remaining budget
    heuristic = prize_term[None, :] * slack_term / detour_term
    
    # Check deadline
    if time.time() > deadline:
        # Return a simple fallback heuristic
        fallback = np.ones_like(distance_matrix)
        return fallback
    
    # Ensure validity: non-positive or non-finite values become 1e-9
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
