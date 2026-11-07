"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Ensure non-negative and finite
    prizes = np.clip(prizes, 0, None)
    distance_matrix = np.clip(distance_matrix, 0, None)
    
    n = len(prizes)
    
    # The constraint for a move i -> j is:
    # length_so_far + d[i, j] + d[j, 0] <= max_len
    #
    # We want a heuristic h[i, j] that encourages:
    # 1. High prize at j.
    # 2. Low "commitment" cost d[i, j] + d[j, 0] (saves space in the budget).
    # 3. High "slack" remaining after the move. Since length_so_far varies,
    #    we approximate the "value" of a move by how much budget it leaves 
    #    relative to the maximum possible remaining budget.
    #
    # Structural Change: Instead of just prize/cost, we use a "slack-weighted" prize.
    # Let cost_ij = d[i, j] + d[j, 0].
    # The "slack" available for future moves after taking this step is roughly (max_len - cost_ij) 
    # assuming we start with full budget. But since we are at i, we have already used some budget.
    # However, a static heuristic cannot know length_so_far.
    # 
    # A better proxy for "good move" in orienteering is often "prize per unit distance" 
    # but penalizing the return distance heavily because it is a "dead cost" (you must pay it to finish).
    #
    # Let's try a different framing: 
    # h[i, j] = prize[j] * slack_factor[i, j] / (1 + d[i, j])
    # where slack_factor reflects how "cheap" it is to include j in the tour.
    #
    # Actually, the inspiration "Slack weighting prioritizes moves that leave remaining budget capacity" 
    # suggests we should reward moves with lower total commitment.
    #
    # Let's define:
    # commitment[i, j] = d[i, j] + d[j, 0]
    #
    # If commitment is small, it leaves more room for other nodes.
    # We can model the attractiveness as:
    # h[i, j] = prize[j]^alpha * (1 + (max_len - commitment[i, j]) / max_len)^beta / (1 + d[i, j])^gamma
    #
    # This rewards:
    # - High prize (alpha)
    # - Low commitment (high slack) (beta)
    # - Low transition distance (gamma)
    
    alpha = 2.0
    beta = 1.0
    gamma = 1.0
    eps = 1e-9
    
    # commitment[i, j] = d[i, j] + d[j, 0]
    # d[i, j] is (N, N)
    # d[j, 0] is the 0-th column of distance_matrix, shape (N,).
    # We need to add d[j, 0] to each column j of distance_matrix.
    # So we broadcast (1, N) against (N, N).
    return_dist = distance_matrix[:, 0] # shape (N,)
    commitment = distance_matrix + return_dist[None, :] # shape (N, N)
    
    # Slack factor: 
    # If commitment is 0, slack is max_len. Factor = 1 + 1 = 2.
    # If commitment is max_len, slack is 0. Factor = 1 + 0 = 1.
    # We want to amplify the difference, so maybe use a power or a ratio.
    # Let's use: slack_term = (max_len - commitment) / max_len
    # This is in [0, 1].
    # If we multiply by this, high commitment nodes get penalized to 0.
    # But we need positive values.
    # Let's use: 1 + slack_term.
    # Or better: (1 + slack_term)^beta.
    
    slack_term = (max_len - commitment) / max_len
    slack_term = np.clip(slack_term, 0, 1)
    
    # Prize term
    prize_term = prizes[None, :] ** alpha
    
    # Distance penalty term: penalize the transition distance d[i, j]
    # We use (1 + d[i, j])^-gamma to keep it positive and bounded.
    dist_term = (1.0 + distance_matrix) ** (-gamma)
    
    h = prize_term * (1.0 + slack_term) ** beta * dist_term
    
    # Ensure positive and finite
    h = np.where(np.isfinite(h), h, 1e-9)
    h = np.maximum(h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
