"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
SEED_TIME_LIMIT = 85  # safety margin for evaluator 90s limit
SEED_COMPARATORS = 153

def _apply_network(comparators, bits):
    """Apply a list of (i,j) comparators to a numpy array of bits."""
    a = bits.copy()
    for i, j in comparators:
        if a[i] > a[j]:
            a[i], a[j] = a[j], a[i]
    return a

def _verify_network(comparators):
    """Verify that the network sorts all 2^N boolean inputs."""
    n = len(comparators)
    if n == 0:
        return True
    # Generate all 2^N boolean inputs
    # 2^18 = 262144
    all_inputs = np.arange(2**N, dtype=np.uint32)
    # Create a matrix of shape (2^N, N) where row x has bit i set if x has bit i
    # We can do this efficiently
    bits_matrix = np.zeros((2**N, N), dtype=np.uint8)
    for k in range(N):
        bits_matrix[:, k] = (all_inputs >> k) & 1
    
    # For each input, apply the network and check if sorted
    # To be efficient, we can process in batches or just loop
    # Given 262144 inputs and up to ~80 comparators, this is ~20M ops which is fine in numpy if vectorized
    # But applying sequentially per comparator is hard to vectorize fully without memory blowup if we store all intermediates.
    # Instead, we can check each input individually but that's slow in Python.
    # Better: use a single pass with numpy operations on the whole matrix per comparator step.
    
    current = bits_matrix.copy()
    for i, j in comparators:
        # Swap if current[:, i] > current[:, j]
        mask = current[:, i] > current[:, j]
        if np.any(mask):
            current[mask, i], current[mask, j] = current[mask, j], current[mask, i]
    
    # Check if every row is sorted (non-decreasing)
    # A row is sorted if for all k < N-1, current[k] <= current[k+1]
    sorted_mask = np.all(current[:, :-1] <= current[:, 1:], axis=1)
    return np.all(sorted_mask)

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + SEED_TIME_LIMIT
    
    # Start with the known good seed network
    best_network = _odd_even_transposition()
    best_len = len(best_network)
    
    # If we have time, try a simple local search / hill climbing to reduce size
    # Strategy: Try removing random comparators and see if it still sorts.
    # Or try replacing a comparator with a different one.
    # Given the time limit, we will do a simple random removal heuristic.
    
    current_network = list(best_network)
    current_len = best_len
    
    # Pre-compute the verification function for speed? 
    # _verify_network is vectorized but has overhead. 
    # Let's cache the inputs matrix to avoid regenerating it every time.
    all_inputs = np.arange(2**N, dtype=np.uint32)
    bits_matrix = np.zeros((2**N, N), dtype=np.uint8)
    for k in range(N):
        bits_matrix[:, k] = (all_inputs >> k) & 1

    def _fast_verify(comparators):
        current = bits_matrix.copy()
        for i, j in comparators:
            mask = current[:, i] > current[:, j]
            if np.any(mask):
                current[mask, i], current[mask, j] = current[mask, j], current[mask, i]
        # Check sorted
        if np.any(current[:, :-1] > current[:, 1:]):
            return False
        return True

    # Simple Hill Climbing: try to remove each comparator and see if it's still valid.
    # If yes, remove it and continue from there.
    # This is a greedy approach.
    # We can also try swapping pairs to different positions, but removal is simpler.
    
    improved = True
    while improved and (time.time() < deadline - 5):
        improved = False
        # Iterate backwards to avoid index issues when removing
        for idx in range(len(current_network) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            # Try removing comparator at idx
            trial = current_network[:idx] + current_network[idx+1:]
            if _fast_verify(trial):
                current_network = trial
                current_len = len(current_network)
                improved = True
                # Reset improved flag? No, we want to keep trying.
                # But we broke out of the inner loop logic? No, we are in the for loop.
                # If we remove one, the indices shift. It's safer to restart the scan.
                break
        
        # Optional: try to reorder? For now, stick to removal only for simplicity and safety.
        # If we remove one, we need to re-scan from the end or start.
        # Breaking the for loop and restarting the while loop effectively restarts the scan.

    if current_len < best_len:
        best_network = current_network
        best_len = current_len
    
    return best_network
# EVOLVE-BLOCK-END
