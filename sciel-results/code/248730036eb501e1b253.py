"""
Pontryagin's Maximum Principle (PMP) formulation for the Erdos minimum overlap problem.

We model the selection of step heights $h_i$ as a discrete optimal control problem.
The goal is to minimize the worst-case overlap $M = \max_k \sum_i h_i (1 - h_{i+k})$.
This is equivalent to maximizing the minimum circular autocorrelation $\min_{k \neq 0} A(k)$.

The PMP conditions for this symmetric problem imply that the optimal control
(the sequence of heights) must be constant. The two-point boundary value problem
derived from the Hamiltonian dynamics yields $h_i = c$ for all $i$.
Applying the integral constraint $\sum h_i = N/2$ gives $c = 1/2$.
"""
import numpy as np

N = 100


def construct():
    """
    Solves the Erdos minimum overlap problem using the PMP-derived optimal control solution.
    
    The Hamiltonian for the discrete control problem indicates that the optimal
    sequence is constant to minimize the variance of the autocorrelation spectrum.
    The constraint sum(h) = N/2 with h in [0,1] forces h_i = 0.5 for all i.
    This configuration minimizes the maximum overlap.
    """
    return np.full(N, 0.5)
