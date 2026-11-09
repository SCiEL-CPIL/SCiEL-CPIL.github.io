"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import sympy as sp
from sympy import Rational, pi, symbols, solve, Poly, lcm, Rational as R
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns 12 double-root locations z_1..z_12 (list of floats).
    Strategy: Use nodes from a weighted orthogonal polynomial family.
    We choose z_i as the roots of the generalized Laguerre polynomial L_n^(alpha)(x)
    for a suitable n and alpha, or a shifted/scaled version thereof, to ensure
    good distribution and positivity.
    
    Specifically, we use the roots of L_12^(alpha)(x) for alpha = -1/2?
    Wait, the task says g(x) uses L_2k(x) with alpha = -1/2 implicitly in the definition of L_d?
    "Let L_d be the generalized Laguerre polynomial L_d^(alpha)(x)" with alpha = -1/2.
    So the basis functions are L_0^(-1/2), L_2^(-1/2), ..., L_50^(-1/2).
    
    The roots z_i are variables.
    The system is:
    g(0) = 0
    g'(0) = 1
    g(z_i) = 0
    g'(z_i) = 0
    
    This is a linear system for coefficients c_k.
    For a unique non-trivial solution (or rather, a unique solution for c given z),
    the matrix must be invertible.
    
    To minimize B, we want the largest root of q(x) = g(x)/(x * prod (x-z_i)^2) to be as small as possible.
    q(x) is a polynomial of degree 50 - 1 - 24 = 25.
    
    If we choose z_i to be the roots of a specific orthogonal polynomial, say L_12^(alpha)(x) with alpha = -1/2,
    these are positive and distinct.
    Let's try that first.
    """
    
    # Time guard
    start_time = time.time()
    deadline = 600.0
    
    # Candidate 1: Roots of L_12^(-1/2)(x)
    # sympy.laguerre(n, alpha, x)
    x = symbols('x')
    n = 12
    alpha = Rational(-1, 2)
    
    # Get the polynomial
    L_n = sp.laguerre(n, alpha, x)
    
    # Find roots
    roots = sp.nroots(L_n, n=50)
    
    # Filter for positive real roots
    pos_roots = []
    for r in roots:
        if abs(sp.im(r)) < 1e-10:
            real_part = float(sp.re(r))
            if real_part > 1e-10:
                pos_roots.append(real_part)
    
    # We need exactly 12 distinct positive roots.
    # L_n has n distinct positive roots for alpha > -1.
    if len(pos_roots) != 12:
        # Fallback: geometric progression
        pos_roots = [0.5 * (1.1 ** i) for i in range(12)]
    else:
        # Sort them
        pos_roots = sorted(pos_roots)
        
        # Check time
        if time.time() - start_time > deadline - 60:
            return pos_roots
    
    # Let's try to improve by shifting/scaling.
    # The score is B / (2*pi). We want B small.
    # B is the largest real root of q(x).
    # q(x) depends on z_i.
    
    # Since we can't easily optimize this in 600s with exact arithmetic,
    # let's try a few simple transformations of the Laguerre nodes.
    # 1. Identity (current)
    # 2. Shifted: z_i + c
    # 3. Scaled: c * z_i
    
    # Let's evaluate the score for the current set to see if it's valid.
    # But the task says "Each evaluation takes seconds to about a minute".
    # We have 600s. So we can do maybe 5-10 evaluations.
    
    # Let's try to compute B for the current pos_roots.
    # If it's too slow, we just return the nodes.
    
    # Actually, the prompt says "The evaluator measures the raw value...".
    # The construct() function just returns the z_i's.
    # The evaluator does the rest.
    # So I just need to return good z_i's.
    
    # Heuristic: Roots of Laguerre polynomials are clustered near 0.
    # Maybe scaling them up or down changes B.
    # Let's try a few scales and pick one that seems "stable" or just return the base ones.
    # Without running the expensive check here, I'll return the Laguerre nodes.
    
    # One more idea: The task mentions "Spectral Interpolation with Laguerre Orthogonality Constraints".
    # Using the roots of L_12^(-1/2) is exactly using Laguerre orthogonality nodes.
    
    return pos_roots

# EVOLVE-BLOCK-END
