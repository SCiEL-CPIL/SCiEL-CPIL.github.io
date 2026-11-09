import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 320.0  # Safety margin before 330s hard kill

def acquisition(mu, sigma, f_best, cost):
    """
    Structural Change: Knowledge Gradient (KG) + Thompson Sampling hybrid.
    
    Rationale:
    1. EI is myopic; it only looks at immediate improvement over f_best.
    2. KG approximates the expected reduction in uncertainty about the global minimum.
       For a batch of candidates, the KG for point i is roughly:
       E[ min_j(mu_j - sigma_j * Z_j) | x_i ] - min_j(mu_j - sigma_j * Z_j)
       However, computing exact KG is expensive. A common approximation for minimization
       is to look at the "probability of being the best" combined with the expected value.
       
    3. A more robust and faster alternative that captures exploration/exploitation
       balance better than EI/cost is a "Probability of Improvement" (POI) style
       metric but adjusted for the global context, or simply a Thompson Sampling
       approach where we sample from the posterior and pick the minimum.
       
    4. Since we return scores for 2000 points, we can compute a "Thompson Score":
       Sample a value from N(mu, sigma^2) for each point. The score is -sampled_value.
       But this ignores cost. We need cost-awareness.
       
    5. Let's use a **Cost-Adjusted Thompson Sampling** score.
       Score_i = - (mu_i + sigma_i * Z_i) / cost_i
       Where Z_i is a random standard normal. This is equivalent to sampling from
       the posterior and dividing by cost. This naturally balances exploration (high sigma
       leads to high variance in score, sometimes very low sampled value) and exploitation.
       
    6. However, pure Thompson Sampling can be too greedy or too random. 
       A better structural change is **Expected Improvement with a Gamma prior on cost**
       or **UCB (Upper Confidence Bound)** for minimization.
       
    7. Let's try **UCB (Upper Confidence Bound) for minimization**.
       UCB_i = mu_i - kappa * sigma_i / cost_i
       This directly penalizes high cost and rewards high uncertainty (exploration)
       by subtracting a confidence bound. This is structurally different from EI
       which integrates the tail. UCB is linear in sigma, EI is non-linear.
       
    8. To make it even more distinct and potentially better for 4D Powell:
       The problem is multimodal-ish (shifted minima). 
       EI tends to cluster near the current best. UCB spreads out more.
       
    9. Let's implement a **Hybrid UCB-EI** or just **UCB** with a dynamic kappa.
       But the directive asks for a MATERIAL change. UCB is a different family.
       
    10. Another option: **Probability of Being the Global Minimum (PBG)**
        Approximation: 
        For each point i, the probability it is the global min among the 2000 is:
        P(min_j X_j = X_i) where X_j ~ N(mu_j, sigma_j^2).
        This is hard to compute exactly. 
        Approximation: 1 / (1 + sum_{j != i} exp(-(mu_j - mu_i)^2 / (sigma_j^2 + sigma_i^2)))
        This is computationally O(N^2) which is 4 million ops. Doable in NumPy? 
        2000x2000 matrix is 4M elements. Yes, fast enough.
        
        Let's try **Cost-Weighted Probability of Improvement (POI)**
        POI_i = P(X_i < f_best) = Phi((f_best - mu_i) / sigma_i)
        Score = POI_i / cost_i
        
        POI is simpler than EI. EI = (f_best - mu) * POI + sigma * phi(z).
        EI is dominated by the first term if f_best >> mu.
        POI is bounded [0,1]. EI is unbounded.
        Dividing by cost makes POI/cost very small for high cost.
        
        Let's go with **UCB (Upper Confidence Bound) for minimization**.
        Score = (mu - beta * sigma) / cost
        This is a linear combination. It is structurally different from the non-linear EI.
        It encourages exploration proportional to uncertainty, scaled by cost.
        
        To add a "jump" potential, we can use a **Thompson Sampling** approach:
        Sample theta_i ~ N(mu_i, sigma_i^2)
        Score = -theta_i / cost_i
        This is equivalent to UCB with random beta. It provides stochastic exploration.
        
        Let's implement **Thompson Sampling** with a cost penalty.
        This is a different optimization method (sampling-based vs. expected-value-based).
        
        Implementation:
        1. Check time.
        2. Generate random Z ~ N(0,1) of shape (2000,)
        3. Sampled values = mu + sigma * Z
        4. Score = -Sampled values / cost
        5. Handle infinities.
        
        Why this might be better:
        EI is deterministic given mu, sigma. It always picks the same point if the GP doesn't change much.
        Thompson Sampling introduces randomness, which can help escape local optima in the acquisition function landscape, potentially finding better global structures in the 4D space.
        
        Also, we can add a small UCB component to ensure stability:
        Score = -(mu + sigma * Z) / cost - alpha * sigma / cost
        Actually, pure TS is fine.
        
        Let's refine: 
        The cost is 1 + mean(x). x in [-4,5]. Scaled to [0,1].
        Cost is between 1 and 2.
        
        Let's stick to **Thompson Sampling**.
        
        Wait, the directive says "Do not micro-tune constants". 
        Using TS is a structural change.
        
        One risk: TS can be too random. 
        Let's add a small exploitation term:
        Score = -(mu + sigma * Z) / cost
        
        This is mathematically sound.
        
        Let's verify the time budget.
        Generating 2000 random numbers is negligible.
        Operations are vectorized.
        
        Code structure:
        - Global time check.
        - Vectorized TS.
        - Return scores.
    """
    current_time = time.time()
    if current_time - _start_time > _TIMEOUT:
        # Return a safe default if we are running out of time
        # This should rarely happen as the function is O(N)
        return np.zeros(2000)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Thompson Sampling: Sample from posterior
    # Z ~ N(0, 1)
    z = np.random.randn(len(mu))
    sampled_values = mu + sigma_safe * z
    
    # Minimize sampled_values, penalize by cost
    # Higher score = better
    scores = -sampled_values / cost_safe
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)
    
    return scores
