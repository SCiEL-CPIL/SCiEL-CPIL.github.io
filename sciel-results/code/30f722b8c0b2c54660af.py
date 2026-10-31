import math
import random
import time

# Global state for farthest-point sampling
SELECTED_VECTORS = {}  # n -> list of selected vectors
START_TIME = time.time()
DEADLINE = 300.0  # Safety margin for 330s limit

def _hamming_distance(v1, v2):
    """Compute Hamming distance between two vectors."""
    dist = 0
    for i in range(len(v1)):
        if v1[i] != v2[i]:
            dist += 1
    return dist

def _min_hamming_to_selected(vector, n):
    """Compute minimum Hamming distance to all selected vectors for dimension n."""
    selected = SELECTED_VECTORS.get(n, [])
    if not selected:
        return float('inf')
    
    min_dist = float('inf')
    for sv in selected:
        d = _hamming_distance(vector, sv)
        if d < min_dist:
            min_dist = d
            if min_dist == 0:
                break
    return min_dist

def _is_balanced(vector, n, threshold_factor=1.0):
    """
    Check if a vector is sufficiently balanced.
    Uses chi-squared distance from ideal distribution (n/3 of each).
    """
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
            
    ideal = n / 3.0
    chi_sq = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    
    return chi_sq < (n / 3.0) * 0.5

def priority(vector, n):
    """
    Prioritizes vectors using farthest-point sampling:
    1. Penalize unbalanced vectors heavily
    2. Maximize minimum Hamming distance to already-selected vectors
    3. Use balance as tiebreaker
    """
    # Deadline guard
    if time.time() > DEADLINE:
        c0 = sum(1 for v in vector if v == 0)
        c1 = sum(1 for v in vector if v == 1)
        c2 = n - c0 - c1
        ideal = n / 3.0
        return -((c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2)

    # Compute balance
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
            
    ideal = n / 3.0
    chi_sq_self = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    
    # If very unbalanced, deprioritize
    if chi_sq_self > n:
        return -1000.0 - chi_sq_self

    # Get or initialize selected vectors for this n
    if n not in SELECTED_VECTORS:
        SELECTED_VECTORS[n] = []
    
    selected = SELECTED_VECTORS[n]
    
    # Compute minimum Hamming distance to selected vectors
    min_dist = _min_hamming_to_selected(vector, n)
    
    # The key insight: prioritize vectors that are FAR from already selected ones
    # This forces spatial spreading and reduces the chance of forming APs
    
    # If no vectors selected yet, use balance as primary
    if not selected:
        # Mark this vector as selected
        SELECTED_VECTORS[n].append(vector)
        return -chi_sq_self
    
    # Score: primarily based on minimum distance (higher = better),
    # with balance as secondary
    # Normalize distance: max possible is n
    dist_score = min_dist / n  # in [0, 1]
    balance_score = -chi_sq_self / max(n, 1)  # normalize balance
    
    # Weight: distance should dominate
    # Use a smooth combination: dist_score * (1 + balance_score)
    # This rewards both distance and balance, with distance as primary
    score = dist_score * 10.0 + balance_score * 1.0
    
    # Mark as selected for future queries
    SELECTED_VECTORS[n].append(vector)
    
    return score
