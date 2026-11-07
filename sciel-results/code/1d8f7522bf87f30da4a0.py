# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import time

def acquisition(mu, sigma, f_best):
    """
    Stateful Local Search Acquisition Function.
    
    Strategy:
    1. Compute a hybrid acquisition score (weighted EI/LCB) to identify promising regions.
    2. Instead of just returning these scores (which the evaluator uses to pick the max),
       we perform a local optimization around the top few candidates to find a better point
       within the same "basin" of uncertainty/improvement.
    3. We return the original scores, but we modify the score of the locally optimized 
       point to be the highest, effectively forcing the evaluator to pick our refined point.
       Wait, the evaluator picks the candidate with the highest score from the 2000 provided.
       We cannot add new points. We can only score the 2000 provided.
       
    Correction on Strategy:
    The evaluator calls acquisition(mu, sigma, f_best) where mu/sigma are for 2000 points.
    It then evaluates the point with the max score.
    We cannot inject a new point. We can only influence which of the 2000 is picked.
    
    Therefore, a "local search" that generates a new point is impossible unless we can 
    modify the input arrays, which we can't.
    
    Alternative Structural Strategy:
    The current program uses a "smoothstep" blend of LCB and EI.
    A different design family: **Thompson Sampling (TS)** approximation or **Probability of Improvement (PI)** with a dynamic margin.
    
    Let's look at the "Stateful" aspect. We can store state in a global variable.
    Since the evaluator runs the loop for 30 steps, we can track the history of f_best 
    or the number of calls to adapt the acquisition function dynamically in a way that 
    isn't just a function of the current mu/sigma.
    
    However, the most robust structural change for BO when the surrogate is fixed (GP) 
    and we have a small budget (30 evals) is often **Expected Improvement (EI)** 
    with a well-tuned kappa, or **UCB** with a decaying beta.
    
    The current program uses a max(sigma) based switch. This is fragile.
    
    New Strategy: **Adaptive UCB with Stateful Beta Decay**.
    We will track the number of calls in a global variable.
    Beta will decay from a high exploratory value to a low exploitative value 
    over the 30 steps. This is a standard, robust BO strategy that is structurally 
    different from the "sigma-based heuristic switch" of the parent.
    
    Additionally, to handle the "greediness" of UCB/EI, we can use a **Multi-Start 
    Local Search on the Surrogate** IF we could pick a point not in the 2000. 
    But we can't.
    
    Let's try a different angle: **Gaussian Process Based Direct Search (GP-DS)**.
    No, that's too complex.
    
    Let's stick to **Stateful Adaptive UCB**.
    Why? Because the parent's "smoothstep" on max_sigma is a proxy for time that can 
    be noisy. Explicit time tracking (call count) is more reliable.
    
    Also, we can use a **Hybrid EI/UCB** where the mix is determined by the call count, 
    not the sigma magnitude.
    
    Let's implement:
    1. Global counter `call_count`.
    2. If `call_count < 10`, use UCB with high beta (exploration).
    3. If `call_count >= 10`, use EI (exploitation).
    4. Transition smoothly.
    
    Actually, a more powerful structural change is **Expected Hypervolume Improvement (EHVI)** 
    or **Probability of Improvement (PI)**. PI is very greedy. EI is standard.
    
    Let's try **Pure EI with a dynamic kappa** that starts high and decays.
    Standard EI uses kappa=0.
    "PI" is basically EI with a margin.
    
    Let's go with **Stateful UCB with Explicit Time Decay**.
    This is a distinct optimization strategy (time-based adaptation vs uncertainty-magnitude-based adaptation).
    """
    
    # Global state
    if not hasattr(acquisition, '_call_count'):
        acquisition._call_count = 0
        
    current_step = acquisition._call_count
    acquisition._call_count += 1
    
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety check
    if time.time() - start_time > 325:
        return -mu

    # Total budget is 30 evaluations.
    # Steps 0-4 are the initial 5 random points.
    # Steps 5-29 are the optimization steps (25 steps).
    
    # We define a decay schedule for the exploration parameter.
    # UCB: -mu + beta * sigma
    # EI: (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # If we are in the first few optimization steps (5-9), explore heavily.
    # If we are in the later steps (25-29), exploit heavily.
    
    # Total optimization steps = 25.
    # Let t = (current_step - 5) / 25.0 for step >= 5.
    # For step < 5, it's random init, but acquisition is still called?
    # "starts from 5 uniform random points, then at each step... calls acquisition"
    # So acquisition is called for steps 5, 6, ..., 29.
    # That is 25 calls.
    
    if current_step < 5:
        # This shouldn't happen if the evaluator is strict, but if it does,
        # just return random-ish scores or high uncertainty scores.
        beta = 3.0
        lcb_scores = -mu + beta * sigma
        return np.nan_to_num(lcb_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize step to [0, 1] over the 25 optimization steps
    # Step 5 -> 0.0, Step 29 -> 1.0
    opt_step = current_step - 5
    t = opt_step / 25.0
    
    # Decay beta from 2.0 to 0.0
    # We can use a cosine decay or linear.
    # Linear: beta = 2.0 * (1 - t)
    # Cosine: beta = 1.0 + 1.0 * cos(pi * t)  -> 2.0 to 0.0
    
    beta = 1.0 + 1.0 * np.cos(np.pi * t)
    
    # Calculate UCB scores
    ucb_scores = -mu + beta * sigma
    
    # Calculate EI scores (kappa=0)
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Blend UCB and EI.
    # Early on (t=0, beta=2): We want UCB.
    # Late on (t=1, beta=0): We want EI.
    # We can just use UCB with decaying beta. As beta->0, UCB becomes -mu.
    # But -mu is not quite EI. EI is better for exploitation because it accounts for the 
    # non-linearity of the improvement.
    # So we should blend UCB and EI.
    
    # Weight for UCB: w_ucb = 1 - t
    # Weight for EI: w_ei = t
    
    # However, UCB and EI have different scales.
    # UCB is in objective units. EI is in objective units (expected improvement).
    # They are comparable.
    
    # To ensure smooth transition, we can normalize them or just blend.
    # Blending raw values can be tricky if scales differ significantly.
    # Let's try blending.
    
    w_ucb = 1.0 - t
    w_ei = t
    
    final_scores = w_ucb * ucb_scores + w_ei * ei_scores
    
    # Ensure finite values
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Final time check
    if time.time() - start_time > 328:
        return ucb_scores # Fallback to fast UCB
        
    return final_scores
# EVOLVE-BLOCK-END
