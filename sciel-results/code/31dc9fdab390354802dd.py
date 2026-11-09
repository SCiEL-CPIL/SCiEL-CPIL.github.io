import time
import numpy as np

N = 17
_DEADLINE = 85  # seconds, safety margin before 90s kill

def _seed_network():
    """Odd-even transposition sorting network for N wires."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

def _precompute_inputs():
    """Pre-compute all 2^N boolean inputs as a NumPy array of shape (2^N, N)."""
    num_inputs = 1 << N
    x = np.arange(num_inputs, dtype=np.uint32)
    X = np.zeros((num_inputs, N), dtype=np.uint8)
    for k in range(N):
        X[:, k] = (x >> (N - 1 - k)) & 1
    return X

def _verify_fast(net, X):
    """
    Verify if the network sorts all inputs in X.
    X is shape (2^N, N).
    Returns True if sorted.
    """
    state = X.copy()
    
    for i, j in net:
        vi = state[:, i]
        vj = state[:, j]
        state[:, i] = np.minimum(vi, vj)
        state[:, j] = np.maximum(vi, vj)
        
    # Check if sorted: for each row, is it non-decreasing?
    return not np.any(state[:, :-1] > state[:, 1:])

def construct():
    start_time = time.time()
    X = _precompute_inputs()
    
    network = _seed_network()
    best_net = list(network)
    
    # Verify seed
    if not _verify_fast(network, X):
        return best_net

    # Phase 1: Deterministic Greedy Deletion
    improved = True
    while improved:
        if time.time() - start_time > _DEADLINE:
            break
            
        improved = False
        i = 0
        while i < len(network):
            if time.time() - start_time > _DEADLINE:
                break
            
            # Try removing network[i]
            candidate = network[:i] + network[i+1:]
            
            if _verify_fast(candidate, X):
                network = candidate
                best_net = list(network)
                improved = True
                # Do not increment i
            else:
                i += 1
                
    # Phase 2: Randomized Greedy Deletion to escape local minima
    # We perform several passes, each trying to remove comparators in a random order.
    # If a removal is successful, we update the best and restart the pass.
    
    if time.time() - start_time < _DEADLINE - 5:
        rng = np.random.default_rng()
        max_random_passes = 20  # Cap to avoid infinite loops if time is weird
        passes = 0
        
        while passes < max_random_passes:
            if time.time() - start_time > _DEADLINE - 2:
                break
            
            current_net = list(best_net)
            # Shuffle indices
            indices = list(range(len(current_net)))
            rng.shuffle(indices)
            
            removed_any = False
            for idx in indices:
                if time.time() - start_time > _DEADLINE - 2:
                    break
                if idx >= len(current_net):
                    continue
                    
                candidate = current_net[:idx] + current_net[idx+1:]
                if _verify_fast(candidate, X):
                    current_net = candidate
                    removed_any = True
                    if len(current_net) < len(best_net):
                        best_net = list(current_net)
                        
            if not removed_any:
                # No improvement in this random pass, might be stuck
                # But maybe a different random order would work?
                # If we didn't remove anything, the network is locally optimal for this random order.
                # We can try a few more random orders.
                passes += 1
            else:
                # We made progress, so we should continue searching from this new state
                # Reset passes counter? Or just continue?
                # Continuing is fine.
                passes = 0  # Reset counter since we found something new

    return best_net
