import numpy as np
import time
import random

N = 18
NUM_INPUTS = 1 << N
DEADLINE = time.time() + 85.0

def _verify_network(comparators):
    """
    Verifies if the network sorts all 2^N boolean inputs.
    """
    if not comparators:
        return N <= 1

    # Generate all 2^N inputs
    indices = np.arange(NUM_INPUTS, dtype=np.uint64)
    shifts = np.arange(N-1, -1, -1, dtype=np.uint64)
    inputs = (indices[:, None] >> shifts[None, :]) & 1
    
    state = inputs.copy()
    
    for i, j in comparators:
        a = state[:, i]
        b = state[:, j]
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
        
    # Check if sorted
    diffs = state[:, 1:] - state[:, :-1]
    return np.all(diffs >= 0)

def _generate_seed_network():
    """
    Generates Odd-Even Transposition network.
    """
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    
    # 1. Initialize
    network = _generate_seed_network()
    
    # Verify initial
    if not _verify_network(network):
        return network

    best_network = network
    best_size = len(network)
    
    # 2. Greedy Pruning
    # We iterate until no more changes
    while time.time() < DEADLINE:
        changed = False
        # Iterate backwards to handle index shifts
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > DEADLINE:
                break
                
            candidate = network[:idx] + network[idx+1:]
            if _verify_network(candidate):
                network = candidate
                best_network = network
                best_size = len(network)
                changed = True
        
        if not changed:
            break

    # 3. Local Search (Hill Climbing)
    # We try to move or swap comparators to find a structure that allows
    # further pruning or is simply more compact (though size is fixed in moves).
    # The goal is to reach a state where greedy pruning can remove more.
    
    while time.time() < DEADLINE:
        if not network:
            break
            
        # Try a random swap
        if len(network) > 1:
            i, j = random.sample(range(len(network)), 2)
            candidate = network.copy()
            candidate[i], candidate[j] = candidate[j], candidate[i]
            
            if _verify_network(candidate):
                network = candidate
                best_network = network
                # Try to prune again after a successful swap
                for idx in range(len(network) - 1, -1, -1):
                    if time.time() > DEADLINE:
                        break
                    candidate2 = network[:idx] + network[idx+1:]
                    if _verify_network(candidate2):
                        network = candidate2
                        best_network = network
                        best_size = len(network)
                        break # Restart outer loop? No, just continue local search

        # Try a random move
        if len(network) > 1:
            i = random.randint(0, len(network) - 1)
            j = random.randint(0, len(network) - 1)
            if i != j:
                candidate = network.copy()
                comp = candidate.pop(i)
                candidate.insert(j, comp)
                
                if _verify_network(candidate):
                    network = candidate
                    best_network = network
                    # Try to prune
                    for idx in range(len(network) - 1, -1, -1):
                        if time.time() > DEADLINE:
                            break
                        candidate2 = network[:idx] + network[idx+1:]
                        if _verify_network(candidate2):
                            network = candidate2
                            best_network = network
                            best_size = len(network)
                            break

    return best_network
