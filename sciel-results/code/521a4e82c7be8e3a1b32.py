"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 17

# EVOLVE-BLOCK-START
def _check_network(network, full_check=True, sample_size=1000):
    """
    Check if the network sorts all inputs.
    Uses 0-1 principle.
    """
    if full_check:
        # Generate all 2^N boolean inputs
        # Shape: (2^N, N)
        # This is 131072 x 17 for N=17. Memory ~ 131072 * 17 * 8 bytes ~ 17MB. OK.
        inputs = np.array([list(bin(i)[2:].zfill(N)) for i in range(2**N)], dtype=int)
        # Convert to int matrix
        # Actually, creating the array via list comprehension is slow.
        # Use bit manipulation.
        indices = np.arange(2**N)
        inputs = ((indices[:, None] >> np.arange(N)[None, :]) & 1).astype(int)
        
        # Apply network
        for i, j in network:
            min_val = np.minimum(inputs[:, i], inputs[:, j])
            max_val = np.maximum(inputs[:, i], inputs[:, j])
            inputs[:, i] = min_val
            inputs[:, j] = max_val
            
        # Check if sorted: each row should be non-decreasing
        diffs = np.diff(inputs, axis=1)
        if np.any(diffs < 0):
            return False
        return True
    else:
        # Sample check
        samples = np.random.randint(0, 2, size=(sample_size, N))
        for i, j in network:
            min_val = np.minimum(samples[:, i], samples[:, j])
            max_val = np.maximum(samples[:, i], samples[:, j])
            samples[:, i] = min_val
            samples[:, j] = max_val
        diffs = np.diff(samples, axis=1)
        if np.any(diffs < 0):
            return False
        return True

def _generate_oet_network(n):
    """Generate Odd-Even Transposition Network."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def construct():
    """
    Construct a sorting network for 17 wires using local search optimization.
    Starts with Odd-Even Transposition and tries to remove/move comparators.
    """
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer for overhead
    
    # Start with OET
    network = _generate_oet_network(N)
    current_size = len(network)
    
    # Verify initial network
    if not _check_network(network, full_check=True):
        # Should not happen, but just in case
        return network

    best_network = network.copy()
    best_size = current_size
    
    # Local Search
    # We will try to remove comparators or move them.
    # Removing is the primary way to reduce size.
    # Moving can help rearrange to allow further removals.
    
    # Pre-generate sample inputs for fast rejection
    # We rely on _check_network with full_check=False for quick checks
    
    attempts = 0
    while time.time() - start_time < time_limit:
        attempts += 1
        if not network:
            break
            
        # Randomly choose a move type
        # 80% remove, 20% move
        if len(network) > 1:
            move_type = random.choice(['remove', 'move', 'remove', 'remove', 'remove', 'move'])
        else:
            move_type = 'remove'
            
        original_network = network.copy()
        
        if move_type == 'remove':
            idx = random.randint(0, len(network) - 1)
            new_network = network[:idx] + network[idx+1:]
            new_size = len(new_network)
            
            # Quick check
            if _check_network(new_network, full_check=False):
                # Full check
                if _check_network(new_network, full_check=True):
                    network = new_network
                    current_size = new_size
                    if current_size < best_size:
                        best_size = current_size
                        best_network = network.copy()
                        # Optional: restart or continue
                else:
                    network = original_network
        elif move_type == 'move':
            idx = random.randint(0, len(network) - 1)
            comp = network[idx]
            # Remove it
            new_network = network[:idx] + network[idx+1:]
            # Insert at random position
            new_idx = random.randint(0, len(new_network))
            new_network = new_network[:new_idx] + [comp] + new_network[new_idx:]
            
            # Quick check
            if _check_network(new_network, full_check=False):
                # Full check
                if _check_network(new_network, full_check=True):
                    network = new_network
                    # Size didn't change, but structure did
                    # We don't update best_size, but we keep this state to explore further
                else:
                    network = original_network
                    
        # Periodic restart if stuck? 
        # If we haven't improved in a long time, maybe shuffle?
        # For now, simple hill climbing.
        
    return best_network
# EVOLVE-BLOCK-END
