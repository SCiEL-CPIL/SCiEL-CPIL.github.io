"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Adjusted Expected Improvement (CAEI).
    
    Strategy:
    1. Instead of a static additive penalty, we adjust the reference value $f_{ref}$ 
       used in the EI calculation.
       Standard EI uses $f_{ref} = f_{best}$.
       Here, we set $f_{ref} = f_{best} + \lambda \cdot (c_{max} - cost)$.
       
       - For expensive points (cost close to $c_{max}$), $f_{ref} \approx f_{best}$.
         They must show improvement over the global best.
       - For cheap points (cost close to $c_{min}$), $f_{ref}$ is significantly higher 
         (worse) than $f_{best}$. 
         This means even a point with $mu < f_{best}$ but not by much might have 
         high EI if it is very cheap, OR more importantly, a point with $mu > f_{best}$ 
         might still have positive EI if it is cheap enough and the GP is uncertain.
         Wait, EI is defined as $E[max(f_{ref} - f(x), 0)]$.
         If $f_{ref}$ is higher, the expected improvement is larger.
         So cheap points get a "boost" in their acquisition score because the 
         threshold they need to beat is lower (i.e., they are allowed to be worse 
         than $f_{best}$ in expectation if they are cheap enough? No, 
         $f_{ref}$ is the target. If $f_{ref}$ is high, it's easier to improve upon it.
         
         Let's re-evaluate the logic.
         We want to minimize cost.
         If we use $f_{ref} = f_{best} + \lambda (c_{max} - cost)$:
         - Expensive point: $f_{ref} \approx f_{best}$. Must beat global best.
         - Cheap point: $f_{ref} = f_{best} + \lambda (c_{max} - c_{min})$. 
           The "target" is worse than the global best. 
           This implies that a cheap point with $mu$ slightly worse than $f_{best}$ 
           might still yield positive EI if the uncertainty is high, or if 
           $f_{ref}$ is sufficiently above $mu$.
           
         Actually, standard EI measures the expected amount by which $f(x)$ beats $f_{ref}$.
         If we raise $f_{ref}$ for cheap points, we are effectively saying:
         "For this cheap point, I am willing to accept a value that is worse than 
         my current best, provided the expected gain relative to this adjusted 
         threshold is high."
         
         This is a form of cost-awareness where cheapness relaxes the exploitation 
         pressure.
         
    2. We use a small lambda to ensure the adjustment is gentle.
       $c_{max}$ is estimated as the max cost in the current candidate set, 
       or a fixed upper bound if we know the domain. 
       Since we don't have coordinates, we can estimate $c_{min}$ and $c_{max}$ 
       from the `cost` array passed in, or assume a range. 
       The cost is $1 + \text{mean}(x)$. Since $x \in [0,1]^3$, mean(x) $\in [0,1]$. 
       So cost $\in [1, 2]$.
       Let's use fixed bounds $c_{min}=1.0, c_{max}=2.0$ for stability, 
       or use the min/max of the provided cost array for normalization.
       Using the provided cost array's min/max is safer as it adapts to the 
       specific sampling distribution of the 2000 candidates.
       
    3. Compute EI with the adjusted reference.
       $z = (f_{ref} - mu) / sigma$
       $EI = (f_{ref} - mu) \Phi(z) + sigma \phi(z)$
       
    4. Return the EI scores.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 280:
        # Fallback: simple exploitation with cost penalty
        scores = -mu / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(scores), scores, -np.inf)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Normalize cost to [0, 1] range for the penalty term
    # Using min/max of the current candidate set
    c_min = np.min(cost)
    c_max = np.max(cost)
    if c_max - c_min < 1e-9:
        c_norm = np.zeros_like(cost)
    else:
        c_norm = (cost - c_min) / (c_max - c_min)
    
    # Lambda: weight of the cost adjustment.
    # If lambda is too high, we might ignore the global best.
    # If lambda is too low, it behaves like standard EI.
    # Let's try lambda = 0.5 * (f_best - min_mu) to scale with the problem scale?
    # Or just a fixed small value relative to the function range.
    # Hartmann 3d range is roughly -3.8 to 0.
    # Let's use a fixed lambda.
    lambda_cost = 0.5
    
    # Adjusted reference value
    # f_ref = f_best + lambda * (c_max - cost)
    # Note: (c_max - cost) is large for cheap points, small for expensive.
    # So f_ref is higher (worse) for cheap points.
    f_ref = f_best + lambda_cost * (c_max - cost)
    
    # Compute z-score
    z = (f_ref - mu) / sigma_safe
    
    # Compute Expected Improvement
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_ref - mu) * Phi_z + sigma_safe * phi_z
    
    # Clip EI to be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite values
    scores = np.where(np.isfinite(ei), ei, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
