"""Odd covering system with moduli dividing 51975; minimize uncovered residues."""
import time
import numpy as np
import random

BASE = 51975
DEADLINE = 115.0  # safety margin under the 120 s wall-clock limit (evaluator kills at 150s)

def _divisors():
    """Generate all odd divisors of 51975 = 3^3 * 5^2 * 7 * 11, excluding 1."""
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
N_DIVS = len(DIVS)
N = BASE

def _best_residue(cover_count, m):
    """
    Find the residue r in [0, m) that maximizes the marginal gain
    (newly covered - becomes uncovered) for modulus m.
    Uses reshape trick for vectorization.
    """
    # Reshape cover_count to (m, N//m)
    # Since N is divisible by m, this works perfectly.
    # The element at [r, k] is cover_count[r + k*m]
    cc_2d = cover_count.reshape(m, N // m)
    
    # Count zeros and ones in each row (each residue class)
    # newly_covered: count of 0s
    # lost_coverage: count of 1s
    newly_covered = np.sum(cc_2d == 0, axis=1)
    lost_coverage = np.sum(cc_2d == 1, axis=1)
    
    gains = newly_covered - lost_coverage
    best_r = np.argmax(gains)
    best_gain = gains[best_r]
    
    return best_r, best_gain

def _greedy(order, t0, deadline):
    """
    Run greedy algorithm with given order of indices.
    Returns (score, chosen_list, cover_count)
    """
    cover_count = np.zeros(BASE, dtype=np.int32)
    chosen = [0] * N_DIVS

    for i in order:
        m = DIVS[i]
        best_r, _ = _best_residue(cover_count, m)
        chosen[i] = best_r
        cover_count[best_r::m] += 1

        if (time.time() - t0) > deadline:
            break

    score = BASE - int(np.sum(cover_count > 0))
    return score, chosen, cover_count

def _local_search(cover_count, chosen, t0, deadline, max_no_improve=200):
    """
    Stochastic Local Search (Hill Climbing with Random Restarts/Kicks).
    """
    score = BASE - int(np.sum(cover_count > 0))
    no_improve_steps = 0
    max_no_improve = max_no_improve
    
    # Precompute indices for moduli? No, reshape is faster for evaluation.
    # But for applying the move, we need slices.
    
    while (time.time() - t0) < deadline:
        # Pick a random modulus index
        i = np.random.randint(0, N_DIVS)
        m = DIVS[i]
        current_r = chosen[i]
        
        # Find best residue for this modulus
        best_r, best_gain = _best_residue(cover_count, m)
        
        if best_r == current_r:
            # No improvement found for this modulus
            no_improve_steps += 1
        else:
            # Calculate actual gain considering current state
            # _best_residue returns gain relative to "do nothing" for this modulus?
            # No, it returns gain = (zeros in new) - (ones in new).
            # But we are *replacing* current_r.
            # Gain of switching from current_r to best_r:
            # Newly covered by best_r: zeros in best_r slice
            # Lost from current_r: ones in current_r slice
            # The _best_residue function calculated:
            # gain_new = zeros(best_r) - ones(best_r)
            # We need: zeros(best_r) - ones(current_r)
            
            # Let's re-calculate properly using the 2D view for accuracy
            cc_2d = cover_count.reshape(m, N // m)
            zeros_all = np.sum(cc_2d == 0, axis=1)
            ones_all = np.sum(cc_2d == 1, axis=1)
            
            # best_r is the argmax of (zeros - ones).
            # But we are removing current_r.
            # If best_r != current_r:
            #   Gain = zeros[best_r] - ones[current_r]
            #   (We cover zeros[best_r] new points. We lose ones[current_r] points that were only covered by current_r.)
            #   Note: If a point is in both best_r and current_r? Impossible, residues are distinct mod m.
            
            if best_r != current_r:
                gain = zeros_all[best_r] - ones_all[current_r]
                
                if gain > 0:
                    # Apply move
                    cover_count[current_r::m] -= 1
                    cover_count[best_r::m] += 1
                    chosen[i] = best_r
                    score -= gain
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                no_improve_steps += 1
        
        # If stuck, perform a random kick to escape local minimum
        if no_improve_steps > max_no_improve:
            # Pick a random modulus and a random residue
            i_rand = np.random.randint(0, N_DIVS)
            m_rand = DIVS[i_rand]
            r_rand = np.random.randint(0, m_rand)
            
            if r_rand != chosen[i_rand]:
                # Calculate delta for random move
                cc_2d = cover_count.reshape(m_rand, N // m_rand)
                zeros_all = np.sum(cc_2d == 0, axis=1)
                ones_all = np.sum(cc_2d == 1, axis=1)
                
                old_r = chosen[i_rand]
                delta = zeros_all[r_rand] - ones_all[old_r]
                
                # Apply random move regardless of delta (kick)
                cover_count[old_r::m_rand] -= 1
                cover_count[r_rand::m_rand] += 1
                chosen[i_rand] = r_rand
                score -= delta
                
            no_improve_steps = 0

    return score, chosen, cover_count

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)

    best_score = BASE
    best_chosen = None
    best_cover_count = None

    # Strategy 1: Descending moduli (large first)
    # Large moduli cover few points, small moduli cover many.
    # Usually, covering with small moduli first is better for greedy?
    # Let's try both.
    
    order_desc = sorted(range(N_DIVS), key=lambda i: -DIVS[i])
    score, chosen, cc = _greedy(order_desc, t0, DEADLINE - 80)
    if score < best_score:
        best_score = score
        best_chosen = chosen.copy()
        best_cover_count = cc.copy()

    # Strategy 2: Ascending moduli (small first)
    if (time.time() - t0) < DEADLINE - 70:
        order_asc = sorted(range(N_DIVS), key=lambda i: DIVS[i])
        score, chosen, cc = _greedy(order_asc, t0, DEADLINE - 70)
        if score < best_score:
            best_score = score
            best_chosen = chosen.copy()
            best_cover_count = cc.copy()

    # Strategy 3: Random order
    if (time.time() - t0) < DEADLINE - 60:
        order_rand = list(range(N_DIVS))
        np.random.shuffle(order_rand)
        score, chosen, cc = _greedy(order_rand, t0, DEADLINE - 60)
        if score < best_score:
            best_score = score
            best_chosen = chosen.copy()
            best_cover_count = cc.copy()

    # Local Search to refine the best initialization
    if (time.time() - t0) < DEADLINE - 5 and best_chosen is not None:
        best_score, best_chosen, best_cover_count = _local_search(
            best_cover_count, best_chosen, t0, DEADLINE - 2
        )

    # Construct the final list of classes
    final_classes = []
    for i in range(N_DIVS):
        m = DIVS[i]
        r = best_chosen[i] if best_chosen is not None else 0
        if r is None or r >= m:
            r = 0
        final_classes.append([m, r])

    return final_classes
