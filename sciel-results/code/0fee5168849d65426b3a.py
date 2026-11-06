"""Optimized sign assignment for low discrepancy on arithmetic progressions."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 256
DEADLINE = 85.0  # safety margin under 90s kill

# Precompute all arithmetic progressions (start, step, length>=1) within [1, N]
# For efficiency, we focus on progressions that can have large sums.
# A progression with step s and length L covers indices start, start+s, ..., start+(L-1)*s
# The sum over such a progression can be at most L in absolute value.

def _generate_progressions():
    """Generate list of (indices) for all arithmetic progressions.
    We store as lists of 0-based indices for numpy operations."""
    progs = []
    # For each step s from 1 to N-1
    for s in range(1, N):
        # For each start
        for start in range(N - s + 1):  # start from 0 to N-1-s (0-based)
            # Maximum length
            max_len = (N - 1 - start) // s + 1
            # Include all lengths from 1 to max_len? That's too many.
            # Key insight: for a fixed (start, step), the longest progression
            # dominates in terms of potential sum magnitude. But shorter ones
            # matter too. However, the max over ALL progressions will be
            # achieved by some progression. Let's include all but be smart.
            # Actually, for step=1, all consecutive runs matter.
            # For large steps, fewer progressions exist.
            # Let's include all lengths to be safe, but this could be huge.
            # Total count: sum over s of sum over start of max_len
            # For s=1: N starts, each up to N lengths -> ~N^2/2 = 32768
            # For s=2: ~N/2 starts, each up to N/2 -> ~N^2/8
            # Total ~ N^2 * (1/2 + 1/8 + 1/18 + ...) ~ N^2 * pi^2/12 ~ 53600
            # That's manageable!
            for length in range(1, max_len + 1):
                idx = list(range(start, start + length * s, s))
                progs.append(idx)
    return progs

# We'll use a smarter approach: instead of enumerating ALL progressions,
# we focus on the most constraining ones and use local search.

def _compute_max_discrepancy(signs, progs):
    """Compute max absolute sum over all progressions.
    signs: array of -1/+1, length N
    progs: list of index lists
    Returns: max absolute sum."""
    max_disc = 0
    for idx in progs:
        s = signs[idx].sum()
        abs_s = abs(s)
        if abs_s > max_disc:
            max_disc = abs_s
            # Early exit if we already exceeded a known bound
    return max_disc

def _compute_max_discrepancy_fast(signs, progs_np, progs_lens):
    """Faster version using numpy.
    signs: np.array of -1/+1
    progs_np: list of np.array of indices
    progs_lens: list of lengths
    """
    max_disc = 0
    for i, idx in enumerate(progs_np):
        s = signs[idx].sum()
        abs_s = abs(s)
        if abs_s > max_disc:
            max_disc = abs_s
    return max_disc

def construct():
    start_time = time.time()
    
    # Generate progressions
    progs = _generate_progressions()
    # Convert to numpy arrays for speed
    progs_np = [np.array(p, dtype=np.int32) for p in progs]
    progs_lens = [len(p) for p in progs]
    
    # Sort progressions by length descending (longer ones more likely to give high sums)
    # Actually, for local search we need to evaluate all. Let's keep them.
    
    # Initial random assignment
    rng = random.Random(42)
    signs = np.array([rng.choice([-1, 1]) for _ in range(N)], dtype=np.int8)
    
    # Compute initial max discrepancy
    best_disc = _compute_max_discrepancy_fast(signs, progs_np, progs_lens)
    best_signs = signs.copy()
    
    # Local search: flip one sign at a time if it reduces max discrepancy
    # To be efficient, we'll use a hill-climbing approach with occasional restarts
    
    # Precompute for each position, which progressions it belongs to
    # pos_to_progs[i] = list of progression indices that include position i
    pos_to_progs = [[] for _ in range(N)]
    for p_idx, idx in enumerate(progs_np):
        for pos in idx:
            pos_to_progs[pos].append(p_idx)
    
    # For each progression, track its current sum
    prog_sums = np.zeros(len(progs_np), dtype=np.int32)
    for p_idx, idx in enumerate(progs_np):
        prog_sums[p_idx] = signs[idx].sum()
    
    # Current max discrepancy
    cur_max = best_disc
    
    def _get_max_disc(prog_sums):
        return int(np.max(np.abs(prog_sums)))
    
    # Hill climbing with restarts
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        if restart == 0:
            # Already initialized
            pass
        else:
            # Random restart with some perturbation of best
            signs = best_signs.copy()
            # Flip a few random positions
            n_flips = rng.randint(1, 5)
            for _ in range(n_flips):
                pos = rng.randint(0, N - 1)
                signs[pos] *= -1
            # Recompute prog_sums
            for p_idx, idx in enumerate(progs_np):
                prog_sums[p_idx] = signs[idx].sum()
            cur_max = _get_max_disc(prog_sums)
        
        # Hill climbing
        improved = True
        while improved and (time.time() - start_time) < DEADLINE - 3:
            improved = False
            # Try flipping each position
            # Shuffle order to avoid bias
            positions = list(range(N))
            rng.shuffle(positions)
            for pos in positions:
                if time.time() - start_time > DEADLINE - 3:
                    break
                # Compute new max if we flip pos
                # Flipping pos changes prog_sums for all progs containing pos
                affected = pos_to_progs[pos]
                if not affected:
                    continue
                
                # Save old sums for affected progs
                old_sums = []
                for p_idx in affected:
                    old_sums.append((p_idx, int(prog_sums[p_idx])))
                    # Flip: prog_sum changes by -2*signs[pos]
                    prog_sums[p_idx] -= 2 * signs[pos]
                
                new_max = _get_max_disc(prog_sums)
                
                if new_max < cur_max:
                    # Accept
                    signs[pos] *= -1
                    cur_max = new_max
                    improved = True
                    # Update best
                    if cur_max < best_disc:
                        best_disc = cur_max
                        best_signs = signs.copy()
                else:
                    # Revert
                    for p_idx, old_sum in old_sums:
                        prog_sums[p_idx] = old_sum
        
        if best_disc <= 3:
            # Can't do much better; likely optimal or near-optimal
            break
    
    # Return as list of Python integers
    return [int(s) for s in best_signs]

# EVOLVE-BLOCK-END
