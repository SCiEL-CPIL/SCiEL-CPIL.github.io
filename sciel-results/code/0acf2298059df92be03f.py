"""Cap set greedy construction with improved static priority heuristic."""
import time
import math
import random
from itertools import product

# Deadline guard
_START_TIME = time.time()
TIME_LIMIT = 280  # seconds, with 20s safety margin before 300s limit

def priority(vector, n):
    """
    Static priority function for greedy cap set construction.
    
    Prioritize vectors that are balanced in digit distribution,
    with tie-breakers that encourage diversity.
    """
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    target = n / 3.0
    
    # Primary: minimize deviation from balanced distribution
    penalty = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    
    # Secondary: sum of digits (moderate weight)
    digit_sum = sum(vector)
    
    # Tertiary: sum of squared digits
    sq_sum = sum(v * v for v in vector)
    
    # Quaternary: lexicographic tie-breaker using base-3 encoding
    # This gives a unique value for each vector
    lex_val = 0.0
    for i in range(n):
        lex_val = lex_val * 3.0 + vector[i]
    
    # Combine: primary dominates, then secondary, then tertiary, then quaternary
    return -penalty * 100000.0 + digit_sum * 1000.0 + sq_sum * 100.0 + lex_val * 0.01
