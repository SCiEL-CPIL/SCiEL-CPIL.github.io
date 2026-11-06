import numpy as np
import time
import random
import math

N = 18

def _batcher_oms(n):
    """
    Generates a Batcher Odd-Merge sorting network for n wires.
    Returns a list of (i, j) pairs.
    """
    if n <= 1:
        return []
    
    # Split into two halves
    n1 = n // 2
    n2 = n - n1
    
    # Recursively sort the two halves
    left = _batcher_oms(n1)
    right = _batcher_oms(n2)
    
    # Shift right network indices by n1
    right = [(i + n1, j + n1) for i, j in right]
    
    # Merge the two sorted halves
    # Batcher's merge for two lists of length n1 and n2 (n1 <= n2)
    # The merge network is constructed recursively.
    # For simplicity and correctness, we can use a standard merge network generator.
    # However, implementing the exact Batcher merge indices is tricky.
    # Alternative: Use Bitonic Sort which is easier to implement recursively.
    # Bitonic sort for N=18:
    # We can pad to 32? No, that increases comparators.
    # Let's implement a generic Bitonic Sort for N wires.
    
    # Actually, let's just use a simple, known-good network generator.
    # Odd-Even Transposition is 153.
    # Let's try to implement Bitonic Sort for N=18.
    
    # Bitonic Sort requires N to be a power of 2 for the standard implementation.
    # For N=18, we can split into 16 and 2? No, that's not standard.
    
    # Let's stick to Odd-Even Transposition as a fallback, but try to improve it.
    # Or, better yet, use a "known" small network if I can hardcode it?
    # No, I must generate it.
    
    # Let's use a recursive Bitonic Sort that handles non-powers of 2 by splitting into powers of 2?
    # No, that's not how it works.
    
    # Let's use the fact that we can construct a network for N=18 by taking a network for N=16 and adding 2 wires?
    # That's complex.
    
    # Let's use a simpler approach: 
    # Generate Odd-Even Transposition (153 comps).
    # Then apply aggressive Greedy Removal.
    # The greedy removal on OET is very effective.
    # I suspect we can get down to < 100 easily.
    
    return _odd_even_transposition_network(n)

def _odd_even_transposition_network(n):
    """Generates the standard Odd-Even Transposition sorting network."""
    network = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            network.append((i, i + 1))
    return network

def _is_valid_network_full(network, inputs):
    """
    Checks if the network sorts all inputs in the provided 'inputs' array.
    'inputs' is a numpy array of shape (2^N, N).
    """
    state = inputs.copy()
    for i, j in network:
        a = state[:, i]
        b = state[:, j]
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    return np.all(state[:, :-1] <= state[:, 1:])

def construct():
    start_time = time.time()
    deadline = 55.0
    
    # Generate inputs
    bits = np.arange(2**N, dtype=np.uint32)
    inputs = np.zeros((2**N, N), dtype=np.uint8)
    for k in range(N):
        inputs[:, k] = (bits >> k) & 1

    # Start with Odd-Even Transposition
    current_network = _odd_even_transposition_network(N)
    
    # Verify initial
    if not _is_valid_network_full(current_network, inputs):
        return current_network

    best_network = list(current_network)
    best_size = len(best_network)
    
    # Greedy Removal Strategy
    # We repeatedly try to remove comparators.
    # To avoid getting stuck in a specific local minimum due to order,
    # we shuffle the network before each pass.
    
    passes = 0
    while time.time() < deadline:
        passes += 1
        # Shuffle the current network to try removals in a different order
        candidate = list(current_network)
        random.shuffle(candidate)
        
        improved = False
        for idx in range(len(candidate) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Try removing candidate[idx]
            test_network = candidate[:idx] + candidate[idx+1:]
            
            # Fast check first? 
            # For N=18, full check is fast enough.
            # But we can do a quick check on a subset to filter out obvious failures.
            # Let's just do full check.
            
            if _is_valid_network_full(test_network, inputs):
                # Valid! Remove it.
                candidate = test_network
                improved = True
                if len(candidate) < best_size:
                    best_size = len(candidate)
                    best_network = list(candidate)
        
        if not improved:
            # No more removals possible in this configuration.
            # Try a random perturbation to escape local minimum.
            # 1. Randomly swap two adjacent comparators
            # 2. Randomly move a comparator
            # 3. Randomly insert a comparator (and then remove others)
            
            # Let's try swapping a random pair of adjacent comparators
            if len(current_network) > 1:
                idx = random.randint(0, len(current_network) - 2)
                perturbed = list(current_network)
                perturbed[idx], perturbed[idx+1] = perturbed[idx+1], perturbed[idx]
                
                # Verify perturbed network
                if _is_valid_network_full(perturbed, inputs):
                    current_network = perturbed
                else:
                    # Try moving a comparator
                    if len(current_network) > 2:
                        idx1 = random.randint(0, len(current_network) - 1)
                        idx2 = random.randint(0, len(current_network) - 1)
                        if idx1 != idx2:
                            perturbed = list(current_network)
                            comp = perturbed.pop(idx1)
                            perturbed.insert(idx2, comp)
                            if _is_valid_network_full(perturbed, inputs):
                                current_network = perturbed
            else:
                # If we are stuck with a very small network, break
                break

        # Update current network to the best candidate found in this pass
        # If we didn't improve, current_network remains the same (or perturbed if valid)
        # We should keep the best valid network we have
        # Actually, if we found a smaller one, we updated best_network.
        # We should set current_network to the best_network if it's smaller,
        # or keep the current one if it's the same size but different structure?
        # Keeping the structure that allowed further removals is key.
        # If 'candidate' was smaller, it is now current_network effectively?
        # No, 'candidate' was local.
        # Let's set current_network to the best_network if it improved, 
        # otherwise keep the one we were working on (which might have been perturbed).
        
        if len(current_network) > best_size:
             # This shouldn't happen if we only accept valid perturbations
             pass
             
        # If we didn't improve in this pass, and we've done many passes, 
        # we might be stuck.
        
        # To ensure we don't loop forever on the same stuck state,
        # if no improvement after a few passes, we can try a more aggressive reset?
        # No, just keep trying perturbations.

    return best_network
