"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function based on Expected Improvement (EI) with a small exploration
    bonus to encourage diversity in early stages, while maintaining exploitation
    efficiency later.
    
    We use a hybrid approach:
    1. Standard Expected Improvement (EI)
    2. A small UCB-like term scaled by the remaining budget (implicit via sigma decay)
    
    However, to differentiate from the base EI, we implement a 'Thompson Sampling'
    inspired variant or a modified EI that handles the standardized GP properly.
    
    Actually, let's stick to a robust EI but with a small epsilon-greedy exploration
    component injected via the score. Since we don't know the step count, we rely on
    the fact that sigma shrinks as the GP learns.
    
    A stronger alternative is the 'Probability of Improvement' (PI) combined with EI,
    or 'Upper Confidence Bound' (UCB). UCB often explores better in high-dimensional
    spaces or when the prior is wide.
    
    Let's try UCB (Upper Confidence Bound) for minimization.
    UCB = -mu + kappa * sigma
    Since we want to minimize f, we want high UCB scores to mean "likely to be low f".
    So score = -mu + kappa * sigma.
    
    The choice of kappa is critical. A common heuristic is to use a time-varying kappa,
    but we don't have step info. A fixed kappa around 2.0-3.0 is standard.
    However, EI is generally superior for minimization when the goal is finding the
    global minimum, not just exploring.
    
    Let's look at the specific failure mode of EI: it can get stuck in local minima
    if sigma is very small but mu is not f_best. UCB explores more aggressively.
    
    Let's try a hybrid: 0.5 * EI + 0.5 * UCB (normalized).
    Or better, just use EI with a slight modification to handle the 'standardized targets'
    aspect more robustly, or switch to a 'Knowledge Gradient' approximation.
    
    Actually, the most robust improvement over vanilla EI in BO for smooth functions
    like RBF mixtures is often **Expected Improvement with a small constant exploration
    bonus** or **UCB**.
    
    Let's implement UCB with a dynamically adjusted kappa based on the spread of sigma.
    If sigma is large, explore more. If sigma is small, exploit.
    kappa = 2.5 * (sigma / (sigma + 0.1))  # This makes kappa ~ 2.5 when sigma >> 0.1, and ~ 2.5*sigma/0.1 when small.
    Wait, that increases kappa when sigma is large.
    
    Let's try: score = -mu + kappa * sigma
    With kappa = 2.0.
    
    Let's compare EI vs UCB.
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    UCB = -mu + kappa * sigma
    
    In the region where mu < f_best (exploitative), EI is positive. UCB is also positive (if -mu is large enough).
    In the region where mu > f_best (exploratory), EI is small/zero. UCB is still positive if sigma is large enough.
    
    Given the task is a "fixed seeded random RBF mixture", the landscape is smooth but multimodal.
    EI is known to be strong for this. However, the initial program already uses EI.
    To get a jump, we need a different mechanism.
    
    Let's try **Thompson Sampling**.
    We can't sample from the GP directly in the acquisition function easily without the kernel matrix,
    but we have mu and sigma. We can approximate Thompson Sampling by adding a random noise
    to the predicted mean? No, that's not standard.
    
    A better alternative: **Maximizing the Probability of Improvement (PI)** is too greedy.
    
    How about **Expected Simple Regret**? That's basically EI.
    
    Let's try **UCB** with a specific kappa.
    The seed program uses EI. UCB is a fundamentally different family (optimistic vs. improvement-based).
    
    Let's use UCB with kappa = 3.0.
    score = -mu + 3.0 * sigma
    
    To make it more robust, we can normalize sigma by its max or mean to make kappa scale-invariant?
    The GP is standardized, so mu and sigma are in objective units.
    If the objective values are around -2.5, and sigma starts around 1.0 and drops to 0.1,
    kappa=3.0 might be too high initially (over-exploration) or just right.
    
    Let's try a **dynamic UCB** where kappa depends on the number of evaluations?
    We don't have N.
    
    Let's try **EI with a small UCB component**.
    score = EI + 0.1 * (-mu + 2.0 * sigma)
    
    Or simply switch to **UCB** with kappa=2.0.
    
    Let's go with **UCB** (Upper Confidence Bound) for minimization.
    It is known to outperform EI in many multimodal landscapes by forcing exploration of
    high-uncertainty regions that EI might ignore if the current best is already decent.
    """
    # Clip sigma to avoid division by zero issues in other contexts, though not needed for UCB
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB for minimization: we want to maximize (-mu + kappa * sigma)
    # Higher score means more likely to find a lower f.
    # kappa controls the exploration-exploitation trade-off.
    # A value of 2.5 is a common starting point.
    kappa = 2.5
    
    scores = -mu + kappa * sigma_safe
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
