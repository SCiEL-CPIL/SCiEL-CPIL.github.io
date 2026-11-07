"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    n = demands.shape[0]
    
    # Pair sum matrix
    pair_sum = demands[:, None] + demands[None, :]
    fits = pair_sum <= capacity
    
    # Waste: remaining space after placing i and j in a bin
    waste = np.maximum(capacity - pair_sum, 0.0)
    
    # For each pair (i, j) that fits, compute the maximum size of a third item k
    # that could also fit in the same bin after i and j.
    # remaining after i,j = capacity - pair_sum[i,j]
    # max third item that fits: max d[k] such that d[k] <= remaining[i,j]
    # We can precompute: for each possible remaining value, find max d[k] <= remaining.
    
    # Since demands are integers 20-99 (passed as floats), and capacity is 150,
    # remaining ranges from 0 to 150 - 40 = 110 (min pair sum is 20+20=40, max remaining=110)
    # Actually remaining can be from 0 (if pair_sum=capacity) up to 150-40=110.
    
    # For each unique remaining value r, find max d[k] where d[k] <= r.
    # We'll vectorize: for each (i,j), remaining = waste[i,j].
    # max_third[i,j] = max{d[k] : d[k] <= waste[i,j]}
    
    # Sort demands to make binary search possible, but we need per-element max.
    # Alternative: for each k, it contributes d[k] to all (i,j) where waste[i,j] >= d[k].
    # max_third[i,j] = max over k of d[k] * (waste[i,j] >= d[k])
    
    # Efficient approach: sort demands, then for each threshold, use broadcasting.
    # Since n=120 and demands are in [20,99], we can do:
    # For each possible demand value (20 to 99), check if waste >= that value.
    
    # Actually simpler: max_third[i,j] = max of demands where demand <= waste[i,j]
    # This is equivalent to: sort demands, for each waste value, find the largest demand <= waste.
    
    sorted_d = np.sort(demands)
    # For each waste value, use searchsorted to find index of largest element <= waste
    # searchsorted(sorted_d, waste, side='right') - 1 gives index of last element <= waste
    
    idx = np.searchsorted(sorted_d, waste, side='right') - 1
    # Where idx < 0, no item fits as third
    max_third = np.where(idx >= 0, sorted_d[np.clip(idx, 0, n-1)], 0.0)
    
    # The strategy: reward pairs that close bins.
    # If max_third is 0 (no third item fits), the pair nearly fills the bin -> very attractive.
    # If max_third is very small, also attractive.
    # We want H[i,j] to be large when the pair is "complete" (no room for more items).
    
    # Let's define attractiveness based on how "closed" the bin is after i,j.
    # A pair is maximally attractive if no third item fits (max_third == 0).
    # Next best: smallest possible third item fits.
    
    # We can set H[i,j] = 1 / (max_third + epsilon) for fitting pairs,
    # which gives high values when max_third is small (bin is nearly full).
    
    # But wait: the strategy says "proportional to this value only if it is zero or very small"
    # Actually re-reading: "setting H[i,j] proportional to this value only if it is zero 
    # (indicating the pair exactly fills the bin) or very small"
    
    # Hmm, this is a bit ambiguous. Let me re-read:
    # "computing, for each item j, the maximum size of a third item k that fits in the 
    # remaining space after i and j are placed, and setting H[i,j] proportional to this 
    # value only if it is zero or very small"
    
    # I think the intent is: H[i,j] should be large when max_third is zero or small,
    # because that means the bin is well-filled. So we want H to be INVERSELY related 
    # to max_third (or related to how close max_third is to 0).
    
    # Let's use: H[i,j] = 1 / (1 + max_third[i,j]) for fitting pairs.
    # This gives H=1 when max_third=0 (perfect fill), H=1/2 when max_third=1, etc.
    
    # But we also need to consider that some pairs don't fit at all.
    
    H = np.zeros((n, n), dtype=float)
    
    # For fitting pairs, compute attractiveness
    # Use inverse of (1 + max_third) to reward tight packing
    H[fits] = 1.0 / (1.0 + max_third[fits])
    
    # For non-fitting pairs, set to a very small value
    H[~fits] = 1e-9
    
    # Ensure no zeros or negatives become problematic
    H = np.maximum(H, 1e-9)
    
    # Ensure finite values
    H = np.where(np.isfinite(H), H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
