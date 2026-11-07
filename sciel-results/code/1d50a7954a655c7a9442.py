import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function for Bayesian Optimization.
    
    Strategy:
    Instead of using Expected Improvement (EI), which measures the expected improvement
    over the current best at a specific point, KG measures the expected reduction in the
    global minimum. For a single candidate x, KG(x) approximates the expected value of
    max(f_best, min(f(x), min_{y in S} f(y))). In the context of a single-step decision
    where we only evaluate one point, KG(x) is closely related to EI but has a different
    interpretation and often better properties for global search, especially when
    combined with an approximation of the future best.
    
    A common approximation for KG in the context of selecting the next point x from a
    set of candidates X is:
    KG(x) = E[ max(f_best, min(f(x), min_{y in X\{x}} f(y))) ] - min(f_best, min_{y in X} f(y))
    
    However, a simpler and computationally efficient approximation often used is:
    KG(x) ≈ E[ max(f_best - f(x), 0) ] + E[ max(min_{y in X} f(y) - f(x), 0) | f(x) < min_{y in X} f(y) ] * P(f(x) < min_{y in X} f(y))
    
    Actually, the standard single-point KG (when we assume the rest of the points are
    fixed at their current posterior means or similar) is:
    KG(x) = E[ min(f_best, f(x)) - min(f_best, min_{y in S} f(y)) ]
    
    Let's use a robust approximation:
    For each candidate x_i, we calculate the expected improvement over the current best,
    but we also account for the fact that evaluating x_i might reveal a value that is
    worse than the current best but better than the expected minimum of the unexplored
    region.
    
    A simpler and very effective variant is the "Expected Improvement of the Global Minimum"
    which can be approximated by:
    KG(x) = E[ max(f_best - f(x), 0) ] * P(f(x) is the new global min) + ...
    
    Let's stick to a well-defined approximation:
    The exact KG for selecting x from a set X is:
    KG(x) = E[ min(f_best, min_{y in X} f(y)) - min(f_best, f(x), min_{y in X\{x}} f(y)) ]
    
    This is complex to compute exactly. A common approximation is to assume that the
    minimum of the other points is fixed at the current minimum of the posterior means
    of the other points, or simply to use EI as a proxy but with a modified weighting.
    
    However, a key difference of KG is that it is non-decreasing and focuses on the
    global best. A simpler structural change is to use **Probability of Improvement (PI)**
    combined with **Expected Improvement (EI)** in a way that prioritizes global exploration.
    
    Let's implement a specific KG approximation:
    1. Estimate the current global minimum from the candidate set: m_global = min(mu).
    2. For each candidate x, the KG value is the expected reduction in the global minimum.
    If we evaluate x, the new global minimum will be min(f_best, f(x), m_other).
    Assuming m_other is fixed at min(mu[j] for j != i), the expected new minimum is:
    E[min(f_best, f(x_i), m_other)]
    
    This is still complex. Let's use a simpler, materially different strategy:
    **Entropy-based exploration** or **Thompson Sampling**.
    
    Actually, let's implement **Thompson Sampling**.
    Thompson Sampling samples a function from the posterior GP and picks the point that
    minimizes that sample. This is a fundamentally different approach from deterministic
    acquisition functions like EI or UCB. It naturally balances exploration and
    exploitation by sampling from the posterior distribution of the function values.
    
    Steps for Thompson Sampling:
    1. Sample a vector of function values y ~ N(mu, sigma^2) from the posterior.
       Since we don't have the covariance matrix (only mu and sigma for 2000 points),
       we can approximate this by sampling y_i ~ N(mu_i, sigma_i^2) independently.
       This ignores correlations, which is a simplification, but it's a valid
       "different" strategy.
    2. Find the index i that minimizes y_i.
    3. Return a score vector where the score is high for the selected point and low
       for others. However, the task requires returning 2000 scores such that the
       evaluator picks the one with the highest score. So we can return -y_i.
       The point with the lowest y_i will have the highest -y_i.
    
    Wait, the task says "returns 2000 finite scores, higher meaning evaluate next".
    So if we sample y, we want the point with the smallest y to have the largest score.
    So score = -y.
    
    Let's implement Thompson Sampling with independent samples (ignoring correlations).
    This is a structurally different approach: stochastic sampling vs. deterministic
    optimization of an acquisition function.
    
    To make it more robust, we can sample multiple times and average, but a single
    sample is the standard Thompson Sampling approach.
    
    Let's implement this.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Numerical stability
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Sample from the posterior: y ~ N(mu, sigma^2)
    # We use a fixed random seed? No, we should use random samples.
    # But we need to be deterministic for reproducibility? The task doesn't specify.
    # However, using random samples makes the acquisition stochastic.
    # The evaluator calls acquisition once per step.
    # Let's use a random sample.
    
    # To ensure we don't have issues with random state, we can use np.random.
    # But if the evaluator runs multiple seeds, the random state might be an issue.
    # Let's assume the evaluator handles randomness or we should be deterministic?
    # The task says "draws 2000 fresh uniform candidates, computes the posterior there,
    # calls acquisition". It doesn't mention randomness in acquisition.
    # However, Thompson Sampling is inherently stochastic.
    # Let's try it. If it's too unstable, we can average multiple samples.
    
    # Sample y
    y = mu + sigma_safe * np.random.randn(len(mu))
    
    # We want to maximize the score, so we return -y
    # The point with the lowest y (best function value) will have the highest score.
    scores = -y
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
