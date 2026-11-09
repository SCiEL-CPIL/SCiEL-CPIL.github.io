"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
import time
import random
import math
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 600
    
    # Base roots from Laguerre polynomial L_12
    x_base, _ = roots_laguerre(12)
    x_norm = x_base / np.max(x_base)
    
    # We assume the evaluator returns a score where higher is better (as per "score is seed/value")
    # or a value where lower is better (as per "evaluator measures raw value (lower is better)").
    # The prompt says: "score is seed/value... 1.0 equals the seed program and above 1.0 is an improvement"
    # This implies the *returned* value from construct() is used to compute the score.
    # Wait, construct() returns a list of roots. The evaluator takes these roots and computes a value.
    # The prompt says: "The evaluator measures the raw value (lower is better)."
    # So, if I return roots that lead to a lower C4 bound, the score increases.
    # My optimization target is to MINIMIZE the C4 bound.
    
    # Since I cannot call the evaluator directly inside construct() to get the C4 value 
    # (construct() is called once, and the evaluator runs afterwards on the returned list),
    # I must construct the best possible roots *a priori* based on the "inspirations" and knowledge.
    #
    # HOWEVER, looking at the inspirations:
    # Inspiration 1 (score 33.3): "local optimization search ... that directly minimizes a surrogate"
    # Inspiration 2 (score 1.0): Initial program.
    #
    # The current program uses a surrogate. The stagnation directive says "Make a STRUCTURAL change".
    # The proposed strategy is: "Model the 12 double-root locations as the critical points... 
    # Optimize these low-dimensional moment parameters".
    #
    # Let's try a different structural approach:
    # Instead of scaling the Laguerre roots, let's construct the roots from the 
    # critical points of a polynomial defined by its moments.
    # 
    # Actually, a simpler and more robust structural change given the "black box" nature:
    # The previous attempts converged on Scale ~ 48000 and Shape ~ 1.0 (Laguerre shape).
    # Score 33.3 is quite high. Is it possible to go higher?
    # The relationship V ~ S^(-0.436) suggests that increasing S further might help, 
    # but the previous program capped it or the surrogate penalized it.
    #
    # Let's try a "Multi-start Local Refinement" strategy on the *actual* geometry 
    # if we assume the evaluator is deterministic and smooth. 
    # But we can't call it.
    #
    # Let's re-read the "INSPIRATIONS". 
    # Inspiration 1 used Nelder-Mead on a surrogate. 
    # The current program uses Differential Evolution on a surrogate.
    #
    # NEW STRATEGY: 
    # Abandon the surrogate model entirely. 
    # Assume the "value" is related to the condition number of the Vandermonde matrix 
    # formed by the roots, or the spread. 
    # But wait, if I can't call the evaluator, I can't do black-box optimization on the *real* function.
    #
    # Let's look at the "Knowledge": 
    # "Combines the structural shape of generalized Laguerre roots (alpha=11) ... 
    # with a large scale factor (targeting max~48000) ... Measured: 0.4869 -> 33.31."
    #
    # This suggests that the specific *shape* of the roots matters, and alpha=11 (which is m-1) 
    # is better than alpha=0 (standard Laguerre). 
    # Standard Laguerre L_m corresponds to weight e^-x x^0.
    # Generalized Laguerre L_m^(alpha) has weight e^-x x^alpha.
    # Roots of L_12^(11) are different from L_12^(0).
    #
    # Let's compute the roots for L_12^(11) and scale them.
    # Also, let's try a range of alphas and scales to see which configuration 
    # is most likely to be optimal, based on the "measured" data point.
    #
    # The knowledge explicitly says: "strategy of a program that improved island 3" 
    # used alpha=11 and max~48000 to get 33.31.
    # The current program (score 33.3095) is essentially doing this but maybe with 
    # slight variations or a less precise fit.
    #
    # To make a structural change, let's try to optimize the *relative spacing* 
    # more aggressively using a different basis: Chebyshev nodes scaled and shifted, 
    # or Gauss-Laguerre nodes with a different alpha, or even a mix.
    #
    # Let's try:
    # 1. Roots of Generalized Laguerre L_12^(alpha) for alpha in [0, 11, 22, 44].
    # 2. Scale them so the max root is S.
    # 3. Try S in [1000, 10000, 100000, 1000000].
    #
    # Since I can't evaluate, I have to guess. 
    # However, the "STAGNATION" directive says "Make a STRUCTURAL change: a different design or modelling family".
    #
    # Let's try a **Moment-Based Construction**.
    # We want roots r_1 < ... < r_12.
    # Let's parameterize them as r_i = A * i^B + C.
    # This allows for power-law spacing.
    # Laguerre roots are roughly linear in i for large i, but curved for small i.
    # Power law might be more flexible.
    #
    # Let's optimize A (scale), B (exponent), and C (offset) using a grid search 
    # over a plausible range, assuming the "value" decreases with scale A and 
    # improves with a specific B.
    #
    # Based on the surrogate in the current program, B=1 was favored. 
    # But let's explore B in [0.5, 1.0, 1.5, 2.0].
    # And A in [1e3, 1e4, 1e5, 1e6].
    #
    # I will select the configuration that maximizes the "spread" (max-min) 
    # while keeping the roots well-conditioned (not too close to 0).
    # Why? Because uncertainty bounds often involve the range of support.
    #
    # Actually, let's look at the "Cohn-Goncalves" construction. 
    # It involves a polynomial P(x) = prod (x - r_i)^2.
    # The bound on C4 might be related to the coefficients of P(x).
    # If we scale roots by S, the coefficients scale by powers of S.
    #
    # Let's try a specific structural change: 
    # Use **Gauss-Legendre** roots instead of Laguerre, scaled to [0, S].
    # Gauss-Legendre roots are on [-1, 1]. Shift to [0, 1] and scale.
    # This changes the "shape" fundamentally from Laguerre (which are clustered near 0) 
    # to Legendre (which are more uniform).
    #
    # Let's test this hypothesis: 
    # If the Laguerre shape is suboptimal, Legendre might be better.
    #
    # Plan:
    # 1. Generate roots for Legendre (shifted) and Laguerre (standard and alpha=11).
    # 2. Scale each set to max=48000 (the known good scale).
    # 3. Also try max=100000 and max=10000.
    # 4. Since I can't evaluate, I will return the Laguerre alpha=11 scaled to 48000 
    #    BUT I will refine the *last few roots* to be slightly more spread out, 
    #    as uncertainty bounds are often sensitive to the tails.
    #
    # Wait, the "STAGNATION" directive says "do not micro-tune constants".
    # So I shouldn't just tweak the last roots.
    #
    # Let's try a **Genetic Algorithm** approach on the *parameters* of a parametric family, 
    # using a **multi-objective surrogate** that penalizes clustering (small gaps) 
    # and rewards large scale.
    #
    # Family: r_i = Scale * (i / 12)^Power
    # This is a simple parametric family.
    # Let's use Differential Evolution to optimize Scale and Power 
    # with a surrogate that is:
    # Value = (Scale)^-0.436 * (1 + Penalty(Clustering))
    # Penalty(Clustering) = sum(1/gap_i^2)
    #
    # This is structurally different from using the actual Laguerre roots as the base shape.
    # It uses a synthetic power-law shape.
    #
    # Let's implement this.
    
    def surrogate_value(params):
        log_scale = params[0]
        power = params[1]
        
        scale = np.exp(log_scale)
        
        # Generate roots: r_i = scale * (i/12)^power for i=1..12
        # Ensure r_1 is not 0. i starts at 1.
        i_vals = np.arange(1, 13)
        roots = scale * (i_vals / 12.0) ** power
        
        # Check validity
        if not np.all(np.diff(roots) > 1e-6):
            return 1e10
        
        # Calculate clustering penalty
        gaps = np.diff(roots)
        # Penalize small gaps
        # Normalize gaps by scale to make penalty scale-invariant for shape
        norm_gaps = gaps / scale
        # If any gap is very small, penalty is high
        min_gap = np.min(norm_gaps)
        if min_gap < 1e-3:
             return 1e10
             
        clustering_penalty = 1.0 / (min_gap ** 0.5)
        
        # Scale penalty
        # We want large scale.
        # V ~ scale^-0.436
        # Let's use a steeper decay to encourage larger scales, 
        # assuming the previous model was too conservative.
        scale_factor = scale ** -0.5 
        
        value = scale_factor * clustering_penalty
        
        # Penalty for too large scale (numerical issues)
        if scale > 1e8:
            value += (scale / 1e8) ** 2
            
        return value

    bounds = [
        (np.log(1000), np.log(1e9)),  # log_scale
        (0.5, 2.0)                     # power
    ]
    
    # Use Differential Evolution
    try:
        result = differential_evolution(
            surrogate_value, 
            bounds, 
            maxiter=200, 
            popsize=20, 
            seed=42, 
            tol=1e-8,
            polish=True
        )
        best_log_scale = result.x[0]
        best_power = result.x[1]
    except Exception:
        best_log_scale = np.log(48000.0)
        best_power = 1.0

    # Check time
    if time.time() > deadline - 60:
        # Fallback to known good
        x_alpha11, _ = roots_laguerre(12, alpha=11)
        scale = 48000.0 / np.max(x_alpha11)
        roots = x_alpha11 * scale
        return [float(r) for r in roots]

    final_scale = np.exp(best_log_scale)
    i_vals = np.arange(1, 13)
    final_roots = final_scale * (i_vals / 12.0) ** best_power
    
    # Ensure positive and sorted
    if not np.all(final_roots > 0) or not np.all(np.isfinite(final_roots)):
        # Fallback
        x_alpha11, _ = roots_laguerre(12, alpha=11)
        scale = 48000.0 / np.max(x_alpha11)
        final_roots = x_alpha11 * scale
    else:
        final_roots = np.sort(final_roots)
        
    return [float(r) for r in final_roots]

from scipy.optimize import differential_evolution
# EVOLVE-BLOCK-END
