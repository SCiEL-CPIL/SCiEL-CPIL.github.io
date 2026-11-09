"""Rewritten program using MODEL-AND-OPTIMIZE approach."""
import time

N = 100
DEADLINE_LIMIT = 145  # Safety margin from 150s limit

# EVOLVE-BLOCK-START
def construct():
    """
    Model-and-Optimize approach for OneMax (100 bits).
    
    Model: Each bit $i$ is generated with probability $p_i$ of being '1'.
    Objective: Maximize the actual count of ones $J = \sum_{i=1}^N b_i$.
    
    Optimization:
    The objective is linear and separable. The maximum for each term $b_i$ 
    is 1, achieved when the underlying parameter $p_i$ is set to 1.0.
    The global optimum is therefore $p_i = 1.0$ for all $i \in [1, N]$.
    This solution is found in a single step (exact local/global solve).
    
    Returns:
        A binary string of length N consisting entirely of '1's.
    """
    # The optimization converges immediately to the boundary of the parameter space
    # where all probabilities are 1.0.
    # Construct the result directly from the optimized parameters.
    return '1' * N
# EVOLVE-BLOCK-END
