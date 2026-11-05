"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    # Convert to numpy if needed
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    # Initialize beliefs/probabilities to 1.0
    # u_i represents the "weight" or probability that node i is in the MIS
    u = np.ones(n)
    
    # Number of iterations for message passing
    # More iterations converge to a more stable solution
    # 10-20 is usually sufficient for ER graphs
    iterations = 20
    
    for _ in range(iterations):
        # Calculate the sum of beliefs of neighbors for each node
        # neighbor_sum[i] = sum(u[j] for j in neighbors(i))
        neighbor_sum = adj @ u
        
        # Update rule: u_i = 1 / (1 + sum(neighbor_beliefs))
        # This suppresses nodes that have many or strong neighbors
        u = 1.0 / (1.0 + neighbor_sum)
        
    return u
# EVOLVE-BLOCK-END
