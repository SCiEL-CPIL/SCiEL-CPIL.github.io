"""Odd covering system with moduli dividing 51975; minimize uncovered residues."""
import time
import numpy as np

BASE = 51975
DEADLINE = 110.0  # safety margin under the 120 s wall-clock limit

# All odd divisors of 51975 = 3^3 * 5^2 * 7 * 11, excluding 1.
def _divisors():
    divs = []
    for a in range(4):
        for b in range(3):
            for c in range(2):
                for d in range(2):
                    m = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
                    if m > 1:
                        divs.append(m)
    return sorted(divs)

DIVS = _divisors()
N = len(DIVS)


def _score(covered):
    # covered is a numpy array of 0s and 1s
    return BASE - int(np.sum(covered))


def _greedy(order, t0):
    """
    Run greedy algorithm with given order of indices.
    Returns (score, chosen_list, covered_array)
    """
    covered = np.zeros(BASE, dtype=np.uint8)
    chosen = [None] * N
    
    for i in order:
        m = DIVS[i]
        best_r = 0
        best_g = -1
        
        # Loop through residues to find the one covering the most uncovered points
        for r in range(m):
            # Count how many points in this residue class are currently uncovered (0)
            g = np.sum(covered[r::m] == 0)
            if g > best_g:
                best_g = g
                best_r = r
                 
        chosen[i] = best_r
        covered[best_r::m] = 1
        
        # Time check: if we are running out of time, we stop and return what we have
        # However, we must ensure the result is valid. 
        # If we break early, some chosen[i] will be None.
        # Given the speed of NumPy, this loop should finish in < 1 second.
        if (time.time() - t0) > DEADLINE - 5:
             # Fill remaining with 0 to ensure validity, though suboptimal
             for j in range(i+1, N):
                 if chosen[j] is None:
                     chosen[j] = 0
             break

    return _score(covered), chosen, covered


def construct():
    t0 = time.time()
    
    # Strategy 1: Descending moduli (large moduli first)
    order_desc = sorted(range(N), key=lambda i: -DIVS[i])
    score_desc, chosen_desc, covered_desc = _greedy(order_desc, t0)
    
    # Strategy 2: Ascending moduli (small moduli first)
    # Check if we have enough time for the second strategy
    if (time.time() - t0) < DEADLINE - 10:
        order_asc = sorted(range(N), key=lambda i: DIVS[i])
        score_asc, chosen_asc, covered_asc = _greedy(order_asc, t0)
        
        # Pick the better of the two
        if score_desc <= score_asc:
            best_score = score_desc
            best_chosen = chosen_desc
        else:
            best_score = score_asc
            best_chosen = chosen_asc
    else:
        # Not enough time for second greedy, use the first
        best_score = score_desc
        best_chosen = chosen_desc

    # Construct the final list of classes
    final_classes = []
    for i in range(N):
        m = DIVS[i]
        r = best_chosen[i]
        if r is None:
            r = 0
        final_classes.append([m, r])
        
    return final_classes
