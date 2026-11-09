"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from fractions import Fraction
from collections import defaultdict

DENOMS = list(range(2, 49))
N = len(DENOMS)

# Precompute weights as integers for exact comparison
# We want sum(1/d_i) < 1.
# Multiply by L = lcm(2..48).
# Let W_i = L / d_i.
# We want to maximize sum(W_i) subject to sum(W_i) < L.
# Deficit in original space = 1 - sum(1/d_i) = (L - sum(W_i)) / L.
# Minimizing deficit is equivalent to maximizing sum(W_i) subject to sum(W_i) < L.

L = 1
for d in DENOMS:
    L = lcm(L, d)

WEIGHTS = [L // d for d in DENOMS]
# Sort indices by weight descending to help pruning
WEIGHT_IDX = sorted([(w, i) for i, w in enumerate(WEIGHTS)], key=lambda x: -x[0])
SORTED_WEIGHTS = [w for w, i in WEIGHT_IDX]
SORTED_ORIG_IDX = [i for w, i in WEIGHT_IDX]

# Precompute suffix sums of weights for pruning
suffix_sum = [0] * (N + 1)
for i in range(N - 1, -1, -1):
    suffix_sum[i] = suffix_sum[i + 1] + SORTED_WEIGHTS[i]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Beam search: maintain top-K states at each depth
    # Each state: (current_sum, frozenset of selected sorted indices)
    # We use a dict to avoid duplicate states
    
    # Initial greedy to get a good lower bound
    greedy_sum = 0
    greedy_indices = []
    for i in range(N):
        if greedy_sum + SORTED_WEIGHTS[i] < L:
            greedy_sum += SORTED_WEIGHTS[i]
            greedy_indices.append(SORTED_ORIG_IDX[i])
    
    best_sum_w = greedy_sum
    best_indices = list(greedy_indices)
    
    # Beam search parameters
    # Start with moderate beam width, increase if time permits
    beam_widths = [200, 500, 1000, 2000, 5000, 10000]
    
    for bw in beam_widths:
        if time.time() > deadline - 5.0:
            break
        
        # Beam: list of (sum_w, tuple_of_sorted_indices)
        # Use tuple of sorted indices as key for deduplication
        beam = [(0, ())]
        
        for i in range(N):
            if time.time() > deadline - 3.0:
                break
            
            new_beam = []
            seen = set()
            
            for cur_sum, cur_idx in beam:
                # Option 1: Include item i
                new_sum = cur_sum + SORTED_WEIGHTS[i]
                if new_sum < L:
                    new_idx = cur_idx + (i,)
                    if new_idx not in seen:
                        seen.add(new_idx)
                        new_beam.append((new_sum, new_idx))
                        if new_sum > best_sum_w:
                            best_sum_w = new_sum
                            # Convert sorted indices back to original
                            best_indices = [SORTED_ORIG_IDX[j] for j in new_idx]
                
                # Option 2: Exclude item i
                if cur_idx not in seen:
                    seen.add(cur_idx)
                    new_beam.append((cur_sum, cur_idx))
            
            # Keep top beam_width states by sum (descending)
            new_beam.sort(key=lambda x: -x[0])
            beam = new_beam[:bw]
            
            # If beam is empty or we've exhausted all items, break
            if not beam:
                break
        
        # Check if we can continue with wider beam
        if time.time() > deadline - 10.0:
            break
    
    # If best_indices is empty (shouldn't happen as 1/2 < 1), fallback
    if not best_indices:
        best_indices = [0]
    
    return best_indices
# EVOLVE-BLOCK-END
