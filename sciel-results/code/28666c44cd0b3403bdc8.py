"""Cap set greedy construction with a refined static priority heuristic."""
import time
import math
import random
from itertools import product


def priority(vector, n):
    """
    Static priority function for cap set greedy construction.
    
    Primary: Balance of digit counts (0s, 1s, 2s) - lower variance = higher priority.
    Secondary: Position-weighted sum to break ties structurally.
    Tertiary: Raw sum for final tie-breaking.
    """
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    target = n / 3.0
    penalty = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    
    # Position-weighted sum: encourages spreading non-zero digits across positions
    pos_weighted = sum(vector[i] * (i + 1) for i in range(n))
    
    # Raw sum
    s = sum(vector)
    
    # Combine: primary dominates, then position-weighted, then raw sum
    # Scale factors ensure proper hierarchy
    return -penalty * 10000.0 + pos_weighted * 10.0 + s
