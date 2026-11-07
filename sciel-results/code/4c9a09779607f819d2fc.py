import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    timeout = 300.0
    
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n = len(prizes)
    
    if time.time() - start_time > timeout - 1.0:
        return np.ones((n, n)) * 1e-9

    d_to_depot = distance_matrix[:, 0]
    
    # Slack: remaining budget after moving i->j and returning j->0
    slack = max_len - distance_matrix - d_to_depot[None, :]
    
    # Saturation function: tanh maps (-inf, inf) to (-1, 1)
    # We want positive values. Slack can be negative (infeasible).
    # For infeasible moves, heuristic should be low.
    # tanh(negative) is negative. We want positive heuristic.
    # Use np.tanh(slack / scale) ?
    # Or simply max(slack, 0) ** gamma ?
    
    # Let's use a soft plus: x + sqrt(x^2 + eps) ? No, too complex.
    # Let's use np.tanh(slack). If slack < 0, tanh < 0.
    # We can clamp to 0 or use abs? No, infeasible should be bad.
    # The ant selection logic already filters infeasible moves.
    # So heuristic for infeasible moves doesn't matter much, but should be small.
    # Let's use np.maximum(slack, 0) for the utility part.
    
    # Non-linear saturation: 
    # If slack is very large, utility shouldn't grow infinitely.
    # tanh(slack) does this.
    # But we need to handle negative slack.
    # Let's define utility = np.tanh(slack / 0.5) # Scale factor 0.5
    # utility is in (-1, 1).
    # We want (0, 1).
    # utility_pos = np.maximum(utility, 0) ?
    
    # Alternative: 
    # utility = slack / (slack + 1) if slack > 0 else 0
    # Or simply: utility = np.exp(-slack) ? No, that decreases with slack.
    
    # Let's stick to the directive's "saturation function".
    # f(slack) = 1 - exp(-slack) ? (Saturates at 1)
    # If slack is negative, exp(-slack) > 1, so f < 0.
    # Let's use f(slack) = np.tanh(slack)
    # And then take max(f, 0) to ensure positive contribution to attractiveness?
    # Or just use f(slack) and let the prize/step handle the rest?
    # If f is negative, heuristic is negative -> becomes 1e-9.
    
    scale = 0.5
    sat = np.tanh(slack / scale)
    
    # Ensure sat is positive for feasible moves (slack > 0)
    # tanh(positive) is positive.
    # tanh(negative) is negative.
    
    # Heuristic: Prize * Saturation / Step_Distance
    # Step distance is d[i,j]
    d_step = np.maximum(distance_matrix, 1e-9)
    
    p = np.maximum(prizes, 1e-9)
    
    heuristic = (p[None, :] * sat) / d_step
    
    # If sat is negative (infeasible), heuristic is negative -> 1e-9
    # If sat is positive, heuristic is positive.
    
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
