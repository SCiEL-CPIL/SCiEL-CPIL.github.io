import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Strategy:
    1. Sample a value from the posterior distribution N(mu, sigma^2) for each candidate.
    2. Adjust the sampled value by the cost. Since we want to minimize f(x) and cost is 
       a penalty on the budget, we can interpret the "effective value" as the sampled 
       function value. However, standard TS for minimization just picks the min of the samples.
       
    The cost term `cost` represents the budget consumption. In a cost-aware setting, 
    we might prefer points that give good function values for low cost. 
    A common heuristic is to score based on the expected improvement or regret per unit cost, 
    but for Thompson Sampling, we can incorporate cost by scaling the uncertainty or 
    biasing the sample.
    
    A more principled approach for cost-aware TS:
    We want to maximize the "value" of evaluation. 
    If we assume the goal is to minimize f(x), and each evaluation consumes `cost`, 
    we can define a score for a sampled value $y \sim N(\mu, \sigma^2)$ as:
    $Score = -y / cost$ ? 
    Or simply pick the min of $y$? 
    If we just pick min(y), we ignore cost. 
    If we pick min(y / cost), we heavily favor low-cost points even if their y is slightly worse.
    
    Let's look at the previous successful strategy (LCB). It used `(beta * sigma - mu) / cost`.
    This is equivalent to maximizing `(beta * sigma - mu) / cost`.
    Note that `-mu` is the optimistic mean (lower is better), and `beta * sigma` is the uncertainty bonus.
    So it maximizes `(Uncertainty Bonus - Mean) / Cost`.
    
    Thompson Sampling equivalent:
    Sample $y \sim N(\mu, \sigma^2)$.
    We want to minimize $y$. So we want the smallest $y$.
    To incorporate cost, we can define the "effective regret" or "effective cost".
    If we treat the problem as maximizing a reward $R = -f(x)$, then we want to maximize $R$.
    $R_{sample} = -y$.
    Score = $R_{sample} / cost = -y / cost$.
    This favors low cost and high reward (low f).
    
    However, pure TS can be too exploitative or too exploratory depending on the sample.
    To make it robust, we can add a small exploration bonus to the sample or use a 
    "noisy" cost.
    
    Let's try: 
    1. Generate a random seed based on the current state to ensure reproducibility if needed, 
       or just use the global RNG. The task implies a single call per loop iteration.
       Using `np.random.default_rng()` inside might be non-deterministic across runs unless seeded.
       The task says "seeded instances", but the acquisition function itself doesn't get a seed.
       However, standard TS uses the current RNG state.
       
    2. Sample $y = \mu + \sigma * z$, where $z \sim N(0, 1)$.
    3. Compute score = $-y / cost$.
    4. Return scores.
    
    Wait, the previous program (score 1.6359) used a deterministic UCB. 
    TS is stochastic. If the evaluator runs multiple times, TS might vary. 
    However, the task says "Keep the function names...". It doesn't forbid randomness.
    But to be safe and deterministic (which is often preferred for scoring stability), 
    we can use a "deterministic" Thompson Sampling approximation or just rely on the 
    inherent randomness of `np.random` if the environment doesn't reset the seed between calls.
    
    Actually, a better "different family" that is deterministic is **Probability of Improvement (PoI)** 
    combined with **Probability of Significance (PoS)** or a **Bayesian Regret** approximation.
    
    Let's stick to **Thompson Sampling** but make it slightly more robust by 
    blending the sample with the mean to avoid extreme outliers from the normal distribution 
    dominating the selection due to a single bad random draw.
    
    Alternative: **Softmax Selection** based on EI? No, that's refinement.
    
    Let's implement **Thompson Sampling** with a cost adjustment.
    To avoid the variance of a single sample being too high, we can use the 
    **Expected Value of the Sample** (which is mu) but that's back to UCB/EI.
    
    Let's try a **Hybrid**: 
    Score = $-(\mu + \lambda \sigma z) / cost$
    Where $\lambda$ controls the exploration.
    
    Actually, the most distinct "different family" from UCB/EI is **Knowledge Gradient (KG)**.
    But KG is hard to approximate without coordinates.
    
    Let's go with **Thompson Sampling**. It is a fundamentally different algorithmic paradigm (sampling vs bounding).
    
    To ensure stability, we will use a fixed random seed for the acquisition function's internal 
    random number generation? No, that would make it deterministic and identical every call, 
    which is bad. We should use the global RNG.
    
    One issue: If `sigma` is very small, the sample is close to `mu`. 
    If `sigma` is large, the sample can be very far.
    
    Let's refine the cost handling. 
    $cost = 1 + mean(x_{scaled})$. 
    Low cost is better.
    We want to maximize "value per cost".
    Value = Improvement. 
    In TS, we sample $f(x)$. We want $f(x)$ to be low.
    So we want to maximize $-f(x)$.
    Score = $-f_{sampled} / cost$.
    
    To prevent the score from being dominated by the noise in the sample (variance of TS),
    we can regularize the sample.
    
    Let's implement:
    1. $z \sim N(0, 1)$
    2. $y = \mu + \sigma \cdot z$
    3. $score = -y / cost$
    
    This is simple and different from UCB (which uses $-\mu + \beta \sigma$).
    UCB is deterministic and upper-bounds the value. TS samples the value.
    
    Potential downside: TS can pick a point with a very high (bad) sample if the variance is high, 
    but it can also pick a point with a very low (good) sample. 
    On average, it explores well.
    
    To improve upon naive TS, we can use **Noisy Thompson Sampling** or add a small 
    deterministic exploration term to the sample to ensure we don't miss the best region 
    due to a bad random draw.
    
    Let's add a small UCB-like term to the sample? No, that mixes families.
    
    Let's try **Pure Thompson Sampling** first. If it's too unstable, we might need to 
    average multiple samples? No, that's expensive.
    
    We will use `np.random.normal` for the sampling.
    """
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-6)
    
    # Generate random standard normal samples
    # Using global RNG for stochasticity
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Sample from posterior
    # y ~ N(mu, sigma^2)
    y_samples = mu + sigma * z
    
    # We want to minimize f(x), so we want low y_samples.
    # We also want low cost.
    # Score = -y_samples / cost
    # This rewards low function values and low costs.
    
    scores = -y_samples / cost
    
    # Handle non-finite values
    scores[~np.isfinite(scores)] = -1e9
    
    # Optional: If we want to be more robust, we can add a tiny bit of 
    # "optimism" to the sample to avoid being purely exploitative in the 
    # stochastic sense, or purely exploitative in the deterministic sense.
    # But pure TS is the distinct strategy.
    
    return scores
