"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_legendre
import time


def construct():
    """
    Constructs an 8-node Gauss-Legendre quadrature rule on the interval [0, 1].
    
    Strategy:
    The luminosity distance integral, after the substitution x = z'/z (where z' is the integration variable from 0 to z),
    becomes an integral over the fixed domain [0, 1]:
    D_L = (c/H0) * (1+z) * z * ∫_0^1 dx / E(z*x)
    
    Here, the nodes are in the variable z' space? No, let's re-read the contract.
    "dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i), where a_i=1/(1+z*nodes[i])"
    
    Wait, the formula in the prompt is:
    dL/(c/H0) = (1+z)*z * sum_i weights[i]/E(a_i)
    where a_i = 1/(1 + z * nodes[i])
    
    Let's check the standard luminosity distance formula.
    d_L = (1+z) * c/H0 * ∫_0^z dz'/E(z')
    
    Let u = z'/z, so z' = z*u, dz' = z du.
    ∫_0^z dz'/E(z') = ∫_0^1 z du / E(z*u) = z * ∫_0^1 du / E(z*u)
    
    So d_L/(c/H0) = (1+z) * z * ∫_0^1 du / E(z*u)
    
    The prompt says:
    dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    where a_i = 1/(1+z*nodes[i])
    
    This implies that the quadrature is approximating ∫_0^1 du / E(z*u).
    The nodes in the prompt's formula are used to compute a_i = 1/(1+z*nodes[i]).
    This means nodes[i] corresponds to the variable u in my derivation.
    So the integral is over u in [0,1].
    The integrand is f(u) = 1/E(z*u).
    
    So we need a quadrature rule for ∫_0^1 f(u) du.
    The nodes are in [0,1] and weights sum to 1.
    
    Gauss-Legendre quadrature on [0,1] is the optimal choice for smooth functions.
    It integrates polynomials of degree 2n-1 exactly. For n=8, degree 15.
    The function 1/E(z*u) is smooth for reasonable z and standard cosmologies.
    
    We compute the standard Gauss-Legendre nodes and weights on [-1, 1] and map them to [0, 1].
    """
    # Check time budget
    start_time = time.time()
    deadline = 60.0  # seconds
    
    # Compute Gauss-Legendre nodes and weights on [-1, 1]
    # roots_legendre(n) returns nodes x and weights w such that
    # ∫_{-1}^{1} f(x) dx ≈ sum_i w_i f(x_i)
    x, w = roots_legendre(8)
    
    # Map from [-1, 1] to [0, 1]
    # u = (x + 1) / 2
    # du = dx / 2
    # ∫_0^1 f(u) du = ∫_{-1}^1 f((x+1)/2) (1/2) dx
    nodes = (x + 1.0) / 2.0
    weights = w / 2.0
    
    # Ensure nodes are strictly increasing and in [0, 1]
    # roots_legendre returns nodes in increasing order on [-1, 1]
    # So nodes will be in increasing order on [0, 1]
    
    # Ensure weights are positive and sum to 1
    # weights from roots_legendre are positive
    # Sum of weights on [-1, 1] is 2, so sum on [0, 1] is 1
    
    # Check if time is running out
    elapsed = time.time() - start_time
    if elapsed > deadline - 1.0:
        # Return the current result as a safety measure
        pass
    
    # Verify constraints
    # Nodes strictly increasing in [0, 1]
    if not np.all(np.diff(nodes) > 0):
        raise ValueError("Nodes not strictly increasing")
    if not np.all(nodes >= 0) or not np.all(nodes <= 1):
        raise ValueError("Nodes not in [0, 1]")
    
    # Weights positive and sum to 1
    if not np.all(weights > 0):
        raise ValueError("Weights not positive")
    if abs(np.sum(weights) - 1.0) > 1e-12:
        raise ValueError("Weights do not sum to 1")
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
