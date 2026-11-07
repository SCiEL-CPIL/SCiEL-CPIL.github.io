"""Heuristic via WASTE-COMPATIBILITY: squared-waste potential field that strongly
repels ants from loose fits, combined with complementarity structure from two-pointer
sweep. The squared term creates a sharper contrast between tight and loose pairings."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _two_pointer_complements(demands, capacity):
    """
    For each item, find its ideal partner using a two-pointer sweep.
    Returns an array where ideal_partner[i] is the index (in original order)
    of the item that best complements item i to fill a bin, or -1 if none.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=float)
    
    # Sort indices by demand
    sort_idx = np.argsort(d)
    sorted_d = d[sort_idx]
    
    # ideal_partner[i] will store the original index of the best complement
    ideal_partner = np.full(n, -1, dtype=int)
    
    # Two-pointer: for each item i (in sorted order), find the largest item
    # that fits with it (i.e., d[i] + d[j] <= capacity)
    # We want j such that d[i] + d[j] is as close to capacity as possible
    # but still <= capacity.
    
    # For each i in sorted order, use binary search or two-pointer to find
    # the best complement.
    # Since sorted_d is sorted, for a given i, we want the largest j > i
    # such that sorted_d[i] + sorted_d[j] <= capacity.
    # If no such j exists, try j < i.
    
    for i in range(n):
        target = capacity - sorted_d[i]
        # Find the largest index j such that sorted_d[j] <= target
        # Use binary search
        # We want the rightmost position where sorted_d <= target
        pos = np.searchsorted(sorted_d, target, side='right') - 1
        if pos >= 0:
            # pos is the index in sorted order of the best complement
            # Map back to original index
            ideal_partner[sort_idx[i]] = sort_idx[pos]
    
    return ideal_partner


def _build_complementarity_matrix(demands, capacity):
    """
    Build an attractiveness matrix based on complementarity scores.
    
    For each pair (i, j):
    - If i's ideal partner is j (or j's ideal partner is i), give high score.
    - Otherwise, score decreases with deviation from the ideal pairing.
    - Only pairs that fit (d[i] + d[j] <= capacity) get positive scores.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=float)
    
    # Find ideal partners
    ideal_partner = _two_pointer_complements(demands, capacity)
    
    # Pair sums
    pair_sum = d[:, None] + d[None, :]
    fits = pair_sum <= capacity
    
    # Base score: waste compatibility metric
    # Use 1/(empty^2 + eps) as a base - squared waste creates sharper contrast
    empty = capacity - pair_sum
    base = np.full((n, n), 1e-9)
    base[fits] = 1.0 / (empty[fits] ** 2 + 1.0)
    
    # Complementarity bonus: for each item i, boost the score for its ideal partner
    # and penalize pairs that are far from the ideal structure
    
    # Create a matrix where M[i, j] gets a bonus if j is i's ideal partner
    # or if i is j's ideal partner
    comp_bonus = np.zeros((n, n))
    for i in range(n):
        ip = ideal_partner[i]
        if ip >= 0:
            # Boost the pair (i, ip) and (ip, i)
            comp_bonus[i, ip] = 1.0
            comp_bonus[ip, i] = 1.0
    
    # Deviation penalty: for each i, penalize j that is far from i's ideal partner
    # in terms of demand difference
    # deviation[i, j] = |d[j] - d[ideal_partner[i]]|
    # This encourages picking items close in demand to the ideal complement
    
    # Build a deviation matrix
    dev = np.zeros((n, n))
    for i in range(n):
        ip = ideal_partner[i]
        if ip >= 0:
            # Deviation from ideal complement's demand
            dev[i, :] = np.abs(d - d[ip])
        else:
            # No ideal partner; penalize all equally (or use a default)
            dev[i, :] = capacity  # Maximum deviation
    
    # Combine: final score = base * (1 + alpha * comp_bonus) * exp(-beta * dev / capacity)
    # alpha controls how much we boost ideal partners
    # beta controls how sharply we penalize deviation
    # With squared waste base, we can use slightly stronger complementarity boost
    alpha = 3.0
    beta = 1.5
    
    M = base * (1.0 + alpha * comp_bonus) * np.exp(-beta * dev / capacity)
    
    # Ensure only fitting pairs have positive scores
    M[~fits] = 1e-9
    
    # Ensure all values are positive and finite
    M[~np.isfinite(M)] = 1e-9
    M[M <= 0] = 1e-9
    
    return M


def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    capacity = float(capacity)
    n = len(demands)
    
    # Deadline guard: return a simple heuristic if we're running out of time
    start_time = time.time()
    
    try:
        M = _build_complementarity_matrix(demands, capacity)
    except Exception:
        # Fallback: squared waste tight-fit heuristic
        pair = demands[:, None] + demands[None, :]
        ok = pair <= capacity
        empty = capacity - pair
        M = np.full((n, n), 1e-9)
        M[ok] = 1.0 / (empty[ok] ** 2 + 1.0)
    
    # Final safety checks
    M[~np.isfinite(M)] = 1e-9
    M[M <= 0] = 1e-9
    
    return M
# EVOLVE-BLOCK-END
