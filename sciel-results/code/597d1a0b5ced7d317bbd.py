import numpy as np
from scipy.optimize import linprog

def _get_distribution(distribution_spec):
    # ... same logic ...
    pass

def _precompute_ic_matrix():
    # Returns A_ic, b_ic for the 16x16 IC constraints
    # p[i] - p[j] - v[i]·a[i] + v[i]·a[j] <= 0
    # Variables: a[0..31], p[0..15]
    # ...
    pass

def construct_auction(distribution_spec, support, constraints):
    # ...
    pass
