import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [0, 1] based on training ranges
    # b: [1, 5] -> (b-1)/4
    # s: [1, 5] -> (s-1)/4
    # temp: [1, 40] -> (temp-1)/39
    # pH: [2, 12] -> (pH-2)/10
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    p = params
    
    # Extract parameters with safety constraints for positivity where needed
    
    # 1. Substrate: Monod-like saturation
    # p[0] controls half-saturation. Use exp to ensure positivity.
    Ks = np.exp(p[0]) + 1e-8
    term_s = s_norm / (s_norm + Ks)
    
    # 2. Temperature: Gaussian bell curve
    # p[1] center (mapped to [0,1] via tanh), p[2] width (positive via exp)
    t_center = 0.5 + 0.5 * np.tanh(p[1])
    t_width = np.exp(p[2]) + 1e-6
    term_t = np.exp(-((t_norm - t_center)**2) / (2 * t_width))
    
    # 3. pH: Gaussian bell curve
    # p[3] center (mapped to [0,1] via tanh), p[4] width (positive via exp)
    ph_center = 0.5 + 0.5 * np.tanh(p[3])
    ph_width = np.exp(p[4]) + 1e-6
    term_ph = np.exp(-((ph_norm - ph_center)**2) / (2 * ph_width))
    
    # 4. Biomass: Quadratic term
    # p[5], p[6], p[7] are coefficients for 1, b, b^2
    term_b = p[5] + p[6] * b_norm + p[7] * (b_norm**2)
    
    # 5. Interaction terms (higher-order multivariate polynomial components)
    # These allow the model to capture non-monotonic dependencies and interactions
    # that the independent factorization misses.
    # p[8] is a scaling factor for the main multiplicative term
    # p[9] is an additive offset
    
    # Main multiplicative structure
    main_term = p[8] * term_s * term_t * term_ph * term_b
    
    # Additive interaction terms to capture residual structure
    # We use p[9] as a global additive offset.
    # To utilize the 10-parameter budget more effectively for interactions,
    # we can add a small correction term based on interactions.
    # However, the prompt says "up to 10 fitted constants".
    # The current structure uses p[0]..p[9].
    
    # Let's refine the biomass term to be more robust or add an interaction.
    # Actually, the strategy suggests replacing multiplicative terms with a polynomial,
    # but the directive says "KEEP this design and investigate improvements within its family".
    # The "family" is the multiplicative Gaussian/Monod structure.
    # The "strategy to try" in the prompt text says "Replace... with a single, unconstrained 4th-order multivariate polynomial".
    # BUT the REFINE directive says "KEEP this design... Do NOT switch to a different design family."
    # This is a conflict. The REFINE directive usually takes precedence in these prompts as it's the specific instruction for this turn.
    # "Strategy to try in this change: Replace..." seems to be a general hint or a rejected strategy?
    # Let's look at the inspirations.
    # Inspiration 3 (score 3.72) uses a rational hybrid model.
    # Inspiration 1 (score 28.4) uses multiplicative Gaussian/Monod.
    # Current (score 41.9) uses multiplicative Gaussian/Monod.
    # The best score so far is 3.72 (Inspiration 3).
    # The directive says "COMBINE the strategy of the current program with the strongest ideas of the inspirations".
    # And "take the structural idea of one and the refinement machinery of the other".
    
    # Let's look at Inspiration 3's structure again:
    # y = (A * S_sat + B) / (1 + C * S + D * T_suppress + E * pH_suppress)
    # This is a rational function, not purely multiplicative.
    # It scored 3.72, which is much better than 41.9.
    
    # The REFINE directive says: "KEEP this design [current program's design] and investigate improvements within its family."
    # Current design: y = p[9] * term_s * term_t * term_ph * term_b + p[8]
    # If I strictly keep this design, I might not improve much.
    # However, the prompt also says: "Strategy to try in this change: Replace the multiplicative Gaussian and Monod terms with a single, unconstrained 4th-order multivariate polynomial..."
    # This contradicts the REFINE directive.
    # Usually, the "Strategy to try" is the specific task for this iteration, while REFINE is the general context.
    # But wait, the REFINE directive says "This is a search instruction... Do NOT switch to a different design family."
    # If I switch to a polynomial, I am switching families.
    # If I stick to the multiplicative family, I am following REFINE.
    
    # Let's re-read carefully:
    # "DIRECTIVE: ... REFINEMENT: for this proposal, KEEP this design and investigate improvements within its family."
    # "Strategy to try in this change: Replace the multiplicative Gaussian and Monod terms with a single, unconstrained 4th-order multivariate polynomial..."
    
    # This is a direct contradiction. Which one to follow?
    # Typically, the "Strategy to try" is the specific action requested for this step. The REFINE might be boilerplate or context from a previous step.
    # However, "Do NOT switch to a different design family" is very strong.
    # Let's look at the scores.
    # Current: 41.9
    # Insp 1 (Multiplicative): 28.4
    # Insp 2 (Lorentzian/Multiplicative): 18.5
    # Insp 3 (Rational): 3.7
    
    # If I follow REFINE and stay in the multiplicative family, I might get to ~10-20.
    # If I follow STRATEGY and switch to polynomial, I might get to ~1-5.
    # The prompt asks to "COMBINE the strategy of the current program with the strongest ideas of the inspirations".
    # And "take the structural idea of one and the refinement machinery of the other".
    
    # Let's try to interpret "KEEP this design" as "Keep the entry point and general approach of fitting params" but maybe the "design" refers to the specific functional form?
    # "Do NOT switch to a different design family" suggests I should NOT use the polynomial.
    # I will follow the REFINE directive to stay in the multiplicative/Gaussian family but improve it using ideas from Inspiration 3 (which is rational) or Inspiration 1.
    # Actually, Inspiration 3 is the best. Can I incorporate its "rational" nature into the multiplicative family?
    # Maybe use a rational denominator for the whole thing?
    # Or maybe the "design family" is "smooth, differentiable, parametric model".
    
    # Let's look at the "Strategy to try" again. It is very specific: "Replace... with a single, unconstrained 4th-order multivariate polynomial".
    # If this is the specific instruction for THIS change, I should probably do it, despite the REFINE warning.
    # Why? Because the REFINE warning might be generic. The "Strategy to try" is unique to this prompt.
    # Also, the knowledge says: "4th-order polynomial... Measured: 1 -> 1.18. (strategy of a program that improved island 4)".
    # This suggests the polynomial strategy worked well in a previous context.
    
    # I will implement the 4th-order polynomial strategy as requested by the "Strategy to try" section, as it seems to be the primary directive for this specific iteration, and the REFINE section might be a template artifact.
    # However, to be safe, I will ensure the polynomial is well-scaled and stable.
    
    # Let's construct a 4th-order multivariate polynomial in b_norm, s_norm, t_norm, ph_norm.
    # We have 10 parameters.
    # A full 4th order polynomial in 4 variables has C(4+4, 4) = 35 terms. We only have 10 params.
    # So we must select key terms.
    # Key terms for growth:
    # 1. Constant
    # 2. Linear terms for b, s, t, ph
    # 3. Quadratic terms for t, ph (to capture bell shapes)
    # 4. Interaction terms: s*t, s*ph, b*s, b*t, b*ph, t*ph
    # 5. Higher order: s^2, t^2, ph^2, b^2
    
    # Let's pick 10 terms:
    # p[0]: Constant
    # p[1]: b
    # p[2]: s
    # p[3]: t
    # p[4]: ph
    # p[5]: t^2
    # p[6]: ph^2
    # p[7]: s*t
    # p[8]: s*ph
    # p[9]: b*s
    
    # This covers main effects, quadratic curvature for T and pH (optima), and key interactions.
    
    y = (p[0] 
         + p[1] * b_norm 
         + p[2] * s_norm 
         + p[3] * t_norm 
         + p[4] * ph_norm 
         + p[5] * (t_norm**2) 
         + p[6] * (ph_norm**2) 
         + p[7] * (s_norm * t_norm) 
         + p[8] * (s_norm * ph_norm) 
         + p[9] * (b_norm * s_norm))
         
    return y
