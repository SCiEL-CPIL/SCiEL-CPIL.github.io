"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
TIME_LIMIT = 290.0 # Safety margin from 300s

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a simplified lookahead heuristic.
    Strategy: Select points that maximize immediate EI but penalize points 
    that are likely to be redundant given the top candidates, or simply 
    use a robust EI variant with a softer cost penalty to escape local optima 
    in the cost-aware selection, as suggested by the cell findings.
    
    However, to be structurally different: We will implement a 'Greedy Lookahead' 
    approximation. Since we cannot update the GP easily inside the acquisition 
    without the kernel matrix, we will use a surrogate 'diversity' penalty 
    based on the current sigma and mu to encourage exploring regions that 
    are not just high EI but also likely to reduce uncertainty in unexplored 
    areas more effectively.
    
    Actually, the cell finding says: "Using a softened cost penalty (cost^0.8 
    instead of cost^1) ... Measured: 1 -> 1.221."
    The directive says: "Make a STRUCTURAL change".
    Softening the cost is a tuning change.
    
    Let's try a different structural change: **Probability of Improvement (PI) 
    with a lower bound** or **Upper Confidence Bound (UCB)**?
    UCB is structurally different from EI (which is an integral over the 
    improvement distribution). UCB explores more aggressively.
    
    Let's combine UCB with cost-awareness.
    UCB = mu + kappa * sigma
    Acquisition = UCB / cost
    
    Or better: **Expected Improvement with a 'Regret' bound**?
    
    Let's look at the inspiration: "cost^0.8 ... improved island 3".
    The directive forbids micro-tuning constants if claiming structural change? 
    "Do not micro-tune constants. Make a STRUCTURAL change".
    
    Okay, I will implement a **Two-Stage Selection** logic within the 
    acquisition function's output ranking. 
    Wait, I only return scores. The evaluator picks the max score.
    
    So I need a score that represents a better strategy than EI/cost.
    
    Alternative Structural Change: **Thompson Sampling** style.
    Instead of deterministic EI, we could sample from the posterior? 
    No, `mu` and `sigma` are fixed inputs. We can't sample from the GP 
    posterior directly without the kernel matrix. But we can approximate 
    a "random" improvement by adding noise? No, that's not robust.
    
    Let's reconsider the **Cost** aspect.
    Current: EI / cost.
    Problem: Cost is linear in x. Points with low x (low cost) are favored.
    Ackley minimum is at some x. If the minimum is at high x, EI/cost might 
    struggle to reach it if EI doesn't grow fast enough to overcome the 
    cost ratio.
    
    Structural Change: **Minimize Expected Cost to Reach Improvement**?
    Or **Maximize Improvement per Unit of *Remaining* Budget**?
    We don't know the remaining budget.
    
    Let's try **Probability of Improvement (PI)** instead of EI.
    PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    Score = PI / cost
    PI is bounded [0,1], EI is unbounded. PI might be more stable in 
    regions where sigma is large (exploration) vs EI which might be 
    dominated by the mean.
    
    Let's try **UCB** (Upper Confidence Bound).
    UCB = mu + beta * sigma
    We want to minimize f, so we want high UCB? No, UCB is for maximization 
    or we use -f.
    If minimizing, we want low f.
    Standard UCB for minimization: 
    We want to find low values.
    Let's define "Confidence Interval" for the minimum.
    Lower Bound of f(x) is mu - beta * sigma.
    We want to pick x that has a low Lower Bound (optimistic about being low).
    Score = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost.
    
    This is structurally different from EI. EI integrates the tail. 
    UCB looks at the boundary.
    
    Let's implement UCB with a tunable beta. 
    Since I cannot tune constants effectively without trial and error, 
    I will use a standard beta, e.g., 2.0 or 3.0.
    
    However, the cell finding explicitly mentioned that `cost^0.8` worked.
    The directive says "Do not micro-tune constants". Changing the exponent 
    of cost is a structural change in the *cost model* (sub-linear cost 
    sensitivity). It changes the optimization landscape's geometry.
    Is it "micro-tuning"? It's changing the form of the function.
    
    Let's go with **UCB** as it is a distinct acquisition function family 
    from EI.
    
    UCB_min = mu - kappa * sigma
    We want to maximize the potential to be low.
    Score = (kappa * sigma - mu) / cost
    
    Let's add a safety check for time.
    """
    # Check time
    if time.time() - _START_TIME > TIME_LIMIT:
        # Return a valid finite array, e.g., just EI/cost or zeros
        # To be safe, return the baseline EI/cost if we are running out of time
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / cost

    # UCB Parameter
    # Standard values for beta in BO are often around 1-5.
    # Let's try beta = 2.0.
    beta = 2.0
    
    # Calculate UCB score for minimization
    # We want to maximize (kappa * sigma - mu)
    ucb_term = beta * sigma - mu
    
    # Cost aware
    # To ensure finite scores and avoid division by zero (cost >= 1)
    scores = ucb_term / cost
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
