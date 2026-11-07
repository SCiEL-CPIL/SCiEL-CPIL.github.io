import numpy as np
from scipy.stats import norm
import time

START_TIME = time.time()
TIME_LIMIT = 300.0
MARGIN = 10.0

def acquisition(mu, sigma, f_best):
    # Check time
    if time.time() - START_TIME > TIME_LIMIT - MARGIN:
        # Fallback to pure EI if time is running out (though unlikely)
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # 1. Calculate EI
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Calculate UCB
    # For minimization, we maximize -mu + beta * sigma
    beta = 2.5
    ucb = -mu + beta * sigma
    
    # 3. Rank them
    # Rank 0 is best (highest value)
    # np.argsort(-values) gives indices of sorted values (descending)
    # We need the rank of each original index.
    
    def get_ranks(values):
        # values: shape (N,)
        # We want rank[i] = 0 for the max value
        order = np.argsort(-values)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(values))
        return ranks

    rank_ei = get_ranks(ei)
    rank_ucb = get_ranks(ucb)
    
    # 4. Pareto Criterion
    # Minimize the maximum rank? Or Maximize the minimum rank?
    # "maximizes the minimum of its normalized EI and UCB ranks"
    # If rank is 0 for best, then "minimum rank" is the best.
    # So we want to MINIMIZE min(rank_ei, rank_ucb).
    # The candidate with min(rank) = 0 is the best.
    # The candidate with min(rank) = 10 is worse.
    
    # The prompt says: "returns the score ... higher meaning evaluate next"
    # So we need to map "goodness" (low min-rank) to "high score".
    
    min_rank = np.minimum(rank_ei, rank_ucb)
    
    # Invert to make higher score better
    scores = -min_rank
    
    return scores
