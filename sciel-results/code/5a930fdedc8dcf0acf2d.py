import numpy as np
from math import pi, cos

def construct():
    # Use the 12 smallest positive zeros of J_{1/2}(x).
    # J_{1/2}(x) = sqrt(2/(pi*x)) * sin(sqrt(x)), so zeros are at x = (k*pi)^2.
    # This exploits asymptotic orthogonality of Laguerre polynomials to the Bessel
    # measure, potentially confining the quotient q(x)'s largest root B more
    # effectively than finite-dimensional orthogonal nodes.
    
    # Base Bessel nodes: (k*pi)^2 for k = 1, 2, ..., 12
    k = np.arange(1, 13)
    base_nodes = (k * pi) ** 2  # [pi^2, 4*pi^2, ..., 144*pi^2] ~ [9.87, 39.48, ..., 1421.2]
    
    # The largest Bessel node is ~1421, which is very large.
    # Scaling down may help reduce B, but risks singularity for small scales.
    # Let's try a moderate scale factor to bring nodes into a reasonable range.
    # If B scales linearly with s (as analyzed in the original code),
    # we want s small, but not too small to avoid ill-conditioning.
    
    # Try s = 0.01: nodes in [0.0987, 14.21]
    # Try s = 0.001: nodes in [0.00987, 1.421] - might be too clustered near 0
    # Try s = 0.05: nodes in [0.493, 71.06]
    
    # Based on the Chebyshev success with nodes in [0.1, 15],
    # let's try s such that the largest node is around 15-20.
    # 144*pi^2 * s ≈ 15-20 => s ≈ 15/(144*pi^2) to 20/(144*pi^2)
    # s ≈ 0.0106 to 0.0141
    
    # Let's try s = 0.012
    s = 0.012
    nodes = s * base_nodes
    
    # Ensure nodes are positive and distinct
    nodes = sorted([float(z) for z in nodes])
    
    # Verify: all positive, distinct
    if len(set(nodes)) != 12 or any(z <= 0 for z in nodes):
        # Fallback to Chebyshev nodes on [0.1, 15.0] which scored 0.4540
        n = 12
        L, R = 0.1, 15.0
        k = np.arange(n)
        nodes = 0.5 * (L + R) + 0.5 * (R - L) * np.cos(np.pi * (2 * k + 1) / (2 * n))
        nodes = sorted([float(z) for z in nodes])
    
    return nodes
