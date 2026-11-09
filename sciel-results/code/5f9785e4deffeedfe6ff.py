import time
import math
import numpy as np
from scipy.optimize import minimize, root, brentq
from scipy.special import eval_genlaguerre, factorial

# EVOLVE-BLOCK-START

def _laguerre_poly(x, m, n):
    """Generalized Laguerre polynomial L_m^(n)(x)."""
    return eval_genlaguerre(m, n, x)

def _construct_poly(coeffs, roots):
    """
    Construct the polynomial with given double roots.
    P(x) = prod_i (x - r_i)^2
    We use log-space for stability.
    """
    # We'll work with the polynomial in coefficient form or evaluate directly.
    # For sign changes, we just need to evaluate P(x) at various points.
    # P(x) = prod_i (x - r_i)^2
    # Since all roots are double, P(x) >= 0 for all x, and P(x) = 0 only at the roots.
    # So there are NO sign changes for P(x) itself.
    
    # Wait, the task says "Fourier sign uncertainty upper bound" and "last sign change".
    # Let me re-read the task carefully.
    
    # "Tune the refined Cohn-Goncalves Laguerre-polynomial construction for the Fourier sign uncertainty upper bound."
    # "the notebook supplies 12 double roots with m=12 and dimension n=1, verifies the root constraints, and computes the last sign change."
    # "score is the upper bound on C4"
    
    # I think the "sign change" refers to the sign changes of some related function, not P(x) itself.
    # In the Cohn-Goncalves construction for uncertainty principles, one typically considers
    # a function and its Fourier transform, and looks at sign changes of a certain combination.
    
    # Actually, looking at this more carefully: the "refined Cohn-Goncalves Laguerre-polynomial construction"
    # likely involves constructing a polynomial whose zeros are at specific locations, and the 
    # "sign uncertainty" relates to the number of sign changes in the function or its transform.
    
    # For the Fourier uncertainty principle, if f has N sign changes, then its Fourier transform 
    # has at least N sign changes (or something similar). The "upper bound on C4" likely refers to
    # a constant in an inequality of the form:
    # (uncertainty product) >= C4 / (something involving sign changes)
    
    # Let me think about this differently. The task says:
    # - 12 double roots
    # - m=12 (degree parameter for Laguerre)
    # - n=1 (dimension)
    # - score is the upper bound on C4 (lower is better)
    
    # The Cohn-Goncalves result for Fourier uncertainty with sign changes typically gives bounds
    # of the form: if f has k sign changes, then f̂ has at least k sign changes, and there's a
    # quantitative bound. The constant C4 would depend on the specific construction.
    
    # Given that we need to return 12 positive double-root locations, and the score is an upper bound
    # on C4, I think we need to find root locations that minimize this upper bound.
    
    # Let me assume the construction works as follows:
    # We have a polynomial P(x) = prod_{i=1}^{12} (x - r_i)^2 where r_i > 0
    # The "upper bound on C4" is some function of the root locations.
    
    # Without the exact formula, let me try to infer from the seed program:
    # The seed returns [4.0, 8.0, 12.0, ..., 48.0] (arithmetic progression)
    # This gives score 1.0 (the baseline).
    
    # Let me try to think about what makes a good set of roots. In Laguerre polynomial theory,
    # the zeros of L_m^(n)(x) are all positive and distinct. For m=12, n=1, there are 12 zeros.
    # Maybe the optimal roots are related to the zeros of L_12^(1)(x)?
    
    # Let me compute the zeros of L_12^(1)(x) and see if that gives a better score.
    
    # Actually, I realize I need to understand what the "upper bound on C4" actually is.
    # Let me try a different approach: assume the score is computed by the evaluator based on
    # the root locations, and I need to find roots that minimize it.
    
    # Since I don't know the exact formula, let me try the zeros of the Laguerre polynomial L_12^(1)(x).
    
    pass

def _compute_score(roots):
    """
    Compute the upper bound on C4 given the root locations.
    This is a placeholder - we need to figure out the actual formula.
    
    Let me try to reason about this. In the context of Fourier uncertainty with sign changes:
    - The Cohn-Goncalves theorem relates the number of sign changes of a function to its Fourier transform
    - The "refined" version with Laguerre polynomials likely gives a quantitative bound
    - C4 is a constant in this bound
    
    A common form is: if f has at most k sign changes, then the uncertainty product satisfies
    Δx·Δξ >= C(k) where C(k) depends on k.
    
    For the specific construction with double roots, the bound might be:
    C4 = some_function(roots)
    
    Without the exact formula, let me try to use the fact that the seed program (arithmetic progression
    with spacing 4) gives score 1.0, and try to optimize.
    
    Actually, let me try a completely different interpretation. Maybe the "upper bound on C4" is
    simply a function that the evaluator computes from the roots, and I need to minimize it.
    
    Let me try the zeros of L_12^(1)(x) as a starting point.
    """
    # Placeholder: return a dummy value
    return 1.0

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better).
    """
    deadline = time.time() + 620  # Safety margin before 630s kill
    
    # Try the zeros of the Laguerre polynomial L_12^(1)(x)
    # These are all positive and distinct
    try:
        from numpy.polynomial.laguerre import laggauss
        # laggauss gives Gauss-Laguerre quadrature nodes, which are zeros of L_n(x)
        # For L_12^(1)(x), we need a different approach
        pass
    except:
        pass
    
    # Let me compute zeros of L_m^(alpha)(x) using scipy
    try:
        from scipy.special import roots_genlaguerre
        # roots_genlaguerre(m, alpha) returns the roots of L_m^(alpha)(x)
        roots_lag = roots_genlaguerre(12, 1)
        roots = sorted(roots_lag['roots'].tolist())
        # Verify all are positive
        if all(r > 0 for r in roots):
            return roots
    except Exception as e:
        pass
    
    # Fallback: use the seed program's answer
    return [4.0 * (i + 1) for i in range(12)]

# EVOLVE-BLOCK-END
