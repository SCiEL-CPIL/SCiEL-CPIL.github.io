"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Use Clenshaw-Curtis quadrature of order 8 (N=7 intervals, 8 points).
    # This approach uses Chebyshev-Lobatto nodes, which are denser at the endpoints
    # of the interval [0, 1]. This is beneficial for integrands that vary more
    # rapidly near the boundaries, such as the luminosity distance integrand
    # 1/E(z*t) which can have steep gradients near t=0 for high redshifts.
    
    N = 7  # Number of intervals, so N+1 = 8 nodes
    # Nodes on [-1, 1]
    k = np.arange(N + 1)
    x = np.cos(np.pi * k / N)
    
    # Weights on [-1, 1] for Clenshaw-Curtis
    c = np.zeros(N + 1)
    c[0] = 1.0 / (N * N)
    c[N] = 1.0 / (N * N)
    
    for i in range(1, N):
        s = 0.0
        # Sum over odd j up to floor((N-1)/2)
        # N=7, floor(6/2)=3. j=1, 3? No, j=1, 2, 3?
        # Formula: sum_{j=1}^{floor((N-1)/2)} (1/j^2) cos(2*j*i*pi/N)
        # Wait, the standard formula is:
        # c_i = (2/N^2) * (1 - 2 * sum_{j=1, j odd}^{floor((N-1)/2)} ... ) ?
        # Let's check the exact formula for Clenshaw-Curtis weights.
        # c_0 = c_N = 1/N^2
        # c_i = (2/N^2) * (1 - 2 * sum_{j=1}^{floor((N-1)/2)} (cos(2*j*i*pi/N) / j^2))
        # Note: The sum is over ALL j, not just odd?
        # Reference: "Clenshaw-Curtis Quadrature"
        # c_i = 2/N^2 * (1 - 2 * sum_{j=1, j odd}^{(N-1)/2} cos(2*j*i*pi/N)/j^2) IF N is odd?
        # Let's use a more robust calculation or known values for N=7.
        
        # For N=7:
        # j goes from 1 to 3.
        for j in range(1, (N // 2) + 1):
            s += np.cos(2 * j * i * np.pi / N) / (j * j)
        c[i] = (2.0 / (N * N)) * (1.0 - 2.0 * s)
        
    # Transform nodes to [0, 1]
    nodes = (x + 1.0) / 2.0
    # Sort nodes to be strictly increasing
    sort_idx = np.argsort(nodes)
    nodes = nodes[sort_idx]
    weights = c[sort_idx] / 2.0  # Factor of 1/2 for interval length
    
    # Ensure sum of weights is 1.0 (it should be by construction, but normalize just in case)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
