"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard to ensure we finish well before the 330s limit
_start_time = time.time()
_DEADLINE = 320.0  # 10s safety margin

def acquisition(mu, sigma, f_best):
    r"""
    Acquisition function using Knowledge Gradient (KG).
    
    The Knowledge Gradient for minimization estimates the expected improvement in the 
    best observed value $f_{best}$ if we were to evaluate at candidate $x$.
    
    $KG(x) = E[ \min(f_{best}, f(x)) ] - f_{best}$
    
    Since we want to *minimize* the regret (or maximize the improvement), we define the score 
    as the expected reduction in the best value:
    $Score(x) = f_{best} - E[ \min(f_{best}, f(x)) ]$
    
    Given $f(x) \sim N(\mu(x), \sigma^2(x))$, let $Y = f(x)$.
    $E[ \min(f_{best}, Y) ] = f_{best} \cdot P(Y > f_{best}) + E[ Y \cdot I(Y \le f_{best}) ]$
    
    Using the properties of the truncated normal distribution:
    Let $z = (f_{best} - \mu) / \sigma$.
    $P(Y \le f_{best}) = \Phi(z)$
    $E[ Y \cdot I(Y \le f_{best}) ] = \mu \Phi(z) - \sigma \phi(z)$
    
    So,
    $E[ \min(f_{best}, Y) ] = f_{best} (1 - \Phi(z)) + \mu \Phi(z) - \sigma \phi(z)$
    $= f_{best} - f_{best}\Phi(z) + \mu \Phi(z) - \sigma \phi(z)$
    $= f_{best} - (f_{best} - \mu)\Phi(z) - \sigma \phi(z)$
    
    The expected improvement (KG score) is:
    $Score(x) = f_{best} - E[ \min(f_{best}, Y) ]$
    $= (f_{best} - \mu)\Phi(z) + \sigma \phi(z)$
    
    This is mathematically identical to the Expected Improvement (EI) formula with $\xi=0$.
    Wait, let me double check the definition of KG.
    
    Standard KG definition:
    $KG(x) = E[ \min_{y \in \mathcal{Y}} f(y) | \text{data} ] - \min_{y \in \mathcal{Y}} f(y) | \text{data} \cup \{(x, y_x)\}$
    This is complex because it depends on the *entire* set of evaluated points.
    
    However, a common approximation or specific case when considering only the current best $f_{best}$ and the new point $x$ is:
    $KG(x) \approx E[ \min(f_{best}, f(x)) ] - \min(f_{best}, E[f(x)])$? No.
    
    Let's look at the "Expected Improvement" (EI) again.
    $EI(x) = E[ \max(0, f_{best} - f(x)) ]$.
    For minimization, this is the expected reduction in the best value.
    $EI(x) = (f_{best} - \mu) \Phi(z) + \sigma \phi(z)$ where $z = (f_{best} - \mu) / \sigma$.
    
    Is EI structurally different from the current PI?
    Current program: PI($x$) = $\Phi((f_{best} - \mu) / \sigma)$.
    PI ignores the *magnitude* of the improvement. It only cares about the probability.
    EI cares about the *magnitude* of the expected improvement.
    
    The prompt says: "Implement a Probability of Improvement (PI) acquisition function with a dynamic improvement threshold $\xi$..."
    Wait, the directive says: "Strategy to try in this change: Implement a Probability of Improvement (PI) acquisition function with a dynamic improvement threshold $\xi$ that scales linearly with the current best value $f_{best}$ (e.g., $\xi = 0.1 \times f_{best}$)..."
    
    BUT, the directive ALSO says: "Step 1: in one sentence, state a strategy MATERIALLY DIFFERENT from the current program's... Do not merely tune the current approach."
    And the current program is ALREADY using PI (with $\xi=0$).
    The directive's suggested strategy (PI with dynamic $\xi$) is a *tuning* of the current approach, which violates the "MATERIALLY DIFFERENT" and "Do not merely tune" constraints if interpreted strictly as a new design family.
    
    However, the "Inspiration 1" (score 8.0384) likely used a different approach.
    Let's look at the "Inspiration 2" (score 2.5249).
    
    If I must make a structural change, **Expected Improvement (EI)** is the standard alternative to PI. It is a different design family (magnitude-based vs probability-based).
    
    Let's implement **Expected Improvement (EI)** with $\xi=0$.
    
    Formula:
    $z = (f_{best} - \mu) / \sigma$
    $EI = (f_{best} - \mu) \Phi(z) + \sigma \phi(z)$
    
    This is distinct from PI because:
    1. PI is bounded [0, 1]. EI is unbounded.
    2. PI favors high probability regions even if the expected gain is small.
    3. EI favors regions with high expected gain, balancing probability and magnitude.
    
    This is a standard, robust acquisition function.
    
    Let's verify the math for minimization EI:
    $f(x) \sim N(\mu, \sigma^2)$
    We want $E[\max(0, f_{best} - f(x))]$.
    Let $D = f_{best} - f(x)$. $D \sim N(f_{best} - \mu, \sigma^2)$.
    $E[\max(0, D)] = \mu_D \Phi(\mu_D/\sigma_D) + \sigma_D \phi(\mu_D/\sigma_D)$
    where $\mu_D = f_{best} - \mu$ and $\sigma_D = \sigma$.
    So $EI = (f_{best} - \mu) \Phi(z) + \sigma \phi(z)$ with $z = (f_{best} - \mu) / \sigma$.
    
    This is computationally efficient and structurally different from PI.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a safe fallback: just the negative mean (greedy exploitation)
        return -mu
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is norm.cdf, phi is norm.pdf
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei_scores = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure finite values
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    
    return ei_scores
# EVOLVE-BLOCK-END
