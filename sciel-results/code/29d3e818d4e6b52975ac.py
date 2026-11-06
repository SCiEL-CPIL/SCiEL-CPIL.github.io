"""Rewritten program with efficient evaluation and simulated annealing."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

N = 256

def _evaluate_max_abs_sum(signs):
    """
    Compute the maximum absolute signed sum on any arithmetic progression in [1, N].
    signs: numpy array of shape (N,) with values -1 or 1.
    Returns: integer max absolute sum.
    """
    max_abs = 0
    # Steps from 1 to N//2.
    for s in range(1, N // 2 + 1):
        for j in range(s):
            strand = signs[j::s]
            if strand.size > 1:
                cs = np.cumsum(strand)
                m = np.max(np.abs(cs))
                if m > max_abs:
                    max_abs = m
            # If size is 1, max abs is 1, which is covered if max_abs is 0 initially?
            # No, if N=1, loop doesn't run. But N=256.
            # For s=1, j=0, strand is whole array.
            # For s=N//2, strands are length 2 or 1.
            # Max abs is at least 1.
            if max_abs >= N:
                return max_abs
    # Ensure we return at least 1 if any AP exists
    return max(1, max_abs)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initial candidates
    # Thue-Morse
    tm = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(N)])
    score_tm = _evaluate_max_abs_sum(tm)
    
    # Random
    rng = np.random.RandomState(42)
    rand_signs = rng.choice([-1, 1], N)
    score_rand = _evaluate_max_abs_sum(rand_signs)
    
    # Choose better initial
    if score_tm < score_rand:
        signs = tm.copy()
        current_score = score_tm
    else:
        signs = rand_signs.copy()
        current_score = score_rand
        
    best_signs = signs.copy()
    best_score = current_score
    
    # Simulated Annealing
    temp = 1.0
    cooling = 0.9995
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline - 5.0:
        pos = rng.randint(0, N)
        old_sign = signs[pos]
        signs[pos] = -old_sign
        
        new_score = _evaluate_max_abs_sum(signs)
        
        if new_score < current_score:
            current_score = new_score
            no_improve = 0
            if new_score < best_score:
                best_score = new_score
                best_signs = signs.copy()
        else:
            delta = new_score - current_score
            if delta == 0:
                current_score = new_score
                # Accept neutral move
            else:
                # Accept worse move with prob exp(-delta/temp)
                # Use math.exp for speed
                if random.random() < np.exp(-delta / temp):
                    current_score = new_score
                else:
                    signs[pos] = old_sign
                    no_improve += 1
                    if no_improve > max_no_improve:
                        no_improve = 0
                        # Reset to best known to avoid drifting too far
                        signs = best_signs.copy()
                        current_score = best_score
                        # Reduce temperature
                        temp = max(0.01, temp * 0.5)
        
        temp *= cooling
        
        if time.time() > deadline - 5.0:
            break
    
    final_score = _evaluate_max_abs_sum(best_signs)
    print(f"Final score: {final_score}", flush=True)
    
    return [int(x) for x in best_signs]

# EVOLVE-BLOCK-END
