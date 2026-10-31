"""Rewritten program: dual-view construction via Sidon-like difference sets."""
import numpy as np
from itertools import combinations

def construct():
    """
    Construct a finite set U of nonnegative integers containing 0 to maximize
    the lower bound 1 + log(|U-U|/|U+U|)/log(2*max(U)+1).
    
    Strategy: Use a "dual-view" approach where we start with a Sidon-like structure
    for the difference set, then select elements that minimize additive energy
    while maintaining large difference set size.
    """
    # We'll try several candidate constructions and pick the best one.
    # The score is 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)
    
    def compute_score(U):
        if not U or 0 not in U:
            return 0.0
        U_set = set(U)
        max_u = max(U_set)
        if max_u == 0:
            return 0.0
        # Compute |U+U| and |U-U|
        sum_set = set()
        diff_set = set()
        for a in U_set:
            for b in U_set:
                sum_set.add(a + b)
                diff_set.add(a - b)
        # We want nonnegative differences? The problem says U-U, which includes negatives.
        # But |U-U| counts all differences. Let's keep all.
        # Actually, for the ratio |U-U|/|U+U|, we use the sizes.
        n_diff = len(diff_set)
        n_sum = len(sum_set)
        if n_sum == 0:
            return 0.0
        import math
        log_denom = math.log(2 * max_u + 1)
        if log_denom <= 0:
            return 0.0
        return 1.0 + math.log(n_diff / n_sum) / log_denom
    
    # Candidate 1: Small Sidon set
    # A Sidon set has all pairwise sums distinct (or differences distinct up to sign)
    # Classic small Sidon sets:
    # {0, 1, 4, 10, 18, 23, ...} - let's use known small ones
    
    candidates = []
    
    # Candidate: {0, 1, 4, 10}
    candidates.append([0, 1, 4, 10])
    
    # Candidate: {0, 1, 4, 10, 18}
    candidates.append([0, 1, 4, 10, 18])
    
    # Candidate: {0, 1, 4, 10, 18, 23}
    candidates.append([0, 1, 4, 10, 18, 23])
    
    # Candidate: {0, 1, 4, 10, 18, 23, 31}
    candidates.append([0, 1, 4, 10, 18, 23, 31])
    
    # Candidate: {0, 1, 4, 10, 18, 23, 31, 40}
    candidates.append([0, 1, 4, 10, 18, 23, 31, 40])
    
    # Candidate: {0, 1, 4, 10, 18, 23, 31, 40, 50}
    candidates.append([0, 1, 4, 10, 18, 23, 31, 40, 50])
    
    # Also try arithmetic progression-like sets which have small |U+U|
    # {0, 1, 2, ..., n-1} has |U+U| = 2n-1, |U-U| = 2n-1, ratio = 1, score = 1
    # So APs give score ~1. We need |U-U|/|U+U| > 1 for improvement.
    # Sidon sets have |U-U| ~ n^2 and |U+U| ~ n^2, but the constants matter.
    
    # Let's also try some structured sets
    # {0, 1, 3, 7, 15, 31} - powers of 2 minus 1
    candidates.append([0, 1, 3, 7, 15, 31])
    
    # {0, 1, 3, 7, 15, 31, 63}
    candidates.append([0, 1, 3, 7, 15, 31, 63])
    
    # {0, 1, 2, 5, 10, 17} - quadratic
    candidates.append([0, 1, 2, 5, 10, 17])
    
    # {0, 1, 2, 5, 10, 17, 26}
    candidates.append([0, 1, 2, 5, 10, 17, 26])
    
    # {0, 1, 3, 6, 10, 15} - triangular numbers
    candidates.append([0, 1, 3, 6, 10, 15])
    
    # {0, 1, 3, 6, 10, 15, 21}
    candidates.append([0, 1, 3, 6, 10, 15, 21])
    
    # Evaluate all candidates
    best_score = 0.0
    best_U = [0, 1, 3]
    
    for U in candidates:
        score = compute_score(U)
        if score > best_score:
            best_score = score
            best_U = U
    
    return best_U

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
