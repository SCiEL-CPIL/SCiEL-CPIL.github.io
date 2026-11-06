"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
MAX_COMPARATORS = 4096
DEADLINE = 85.0  # seconds, leave 5s safety margin for 90s wall-clock limit

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a sorting network for N=17 wires as a list of (i, j) pairs.
    Uses a depth-based genetic/local search approach:
    1. Start with a known good network (odd-even transposition).
    2. Apply depth-aware local search to minimize comparator count.
    3. Verify using zero-one principle with bit-vector simulation.
    """
    start_time = time.time()
    
    # Baseline: odd-even transposition
    def odd_even_network(n):
        comps = []
        for phase in range(n):
            for i in range(phase % 2, n - 1, 2):
                comps.append((i, i + 1))
        return comps
    
    # Start with the odd-even network
    current = odd_even_network(N)
    
    # Verify it works (should always work)
    if not verify_network_fast(current):
        # This should never happen, but just in case
        return current
    
    current_count = len(current)
    best_count = current_count
    best_network = current[:]
    
    # Local search: try to remove comparators
    # Strategy: iterate through positions, try removing each one
    # If removal works, keep it and restart from the beginning
    # This is a greedy approach that can get stuck, so we also try random mutations
    
    max_iterations = 500  # Maximum number of removal attempts
    
    for iteration in range(max_iterations):
        if time.time() - start_time > DEADLINE:
            break
        
        improved = False
        
        # Try removing each comparator in order
        for idx in range(len(current)):
            if time.time() - start_time > DEADLINE:
                break
            
            # Create a candidate by removing the comparator at idx
            candidate = current[:idx] + current[idx+1:]
            
            # Verify the candidate
            if verify_network_fast(candidate):
                # Accept the removal
                current = candidate
                current_count -= 1
                improved = True
                
                # Update best if this is better
                if current_count < best_count:
                    best_count = current_count
                    best_network = current[:]
                
                # Restart the removal loop from the beginning
                break
        
        if not improved:
            # No single removal works, try a random mutation
            # Replace a random comparator with a different one
            if len(current) > 0:
                idx = random.randint(0, len(current) - 1)
                # Try replacing with a random comparator
                new_i = random.randint(0, N - 2)
                new_j = random.randint(new_i + 1, N - 1)
                candidate = current[:]
                candidate[idx] = (new_i, new_j)
                
                if verify_network_fast(candidate):
                    current = candidate
                    improved = True
            
            # If still no improvement, try removing two comparators
            if not improved and len(current) >= 2:
                idx1 = random.randint(0, len(current) - 1)
                idx2 = random.randint(0, len(current) - 2)
                if idx2 >= idx1:
                    idx2 += 1
                candidate = current[:idx1] + current[idx1+1:idx2] + current[idx2+1:]
                
                if verify_network_fast(candidate):
                    current = candidate
                    current_count -= 2
                    improved = True
                    
                    if current_count < best_count:
                        best_count = current_count
                        best_network = current[:]
            
            # If no improvement after trying all moves, stop
            if not improved:
                break
    
    # Return the best network found
    return best_network


def verify_network_fast(network):
    """
    Faster verification using bit-parallelism.
    Process multiple inputs at once using numpy.
    """
    n = len(network)
    if n == 0:
        return N <= 1
    
    # Process in batches
    batch_size = 2048
    total_inputs = 1 << N
    
    for batch_start in range(0, total_inputs, batch_size):
        batch_end = min(batch_start + batch_size, total_inputs)
        batch = np.arange(batch_start, batch_end, dtype=np.uint32)
        
        # Initialize wires: shape (batch_size, N)
        wires = np.zeros((batch_end - batch_start, N), dtype=np.uint8)
        for i in range(N):
            wires[:, i] = (batch >> i) & 1
        
        # Apply each comparator
        for i, j in network:
            # For each input in batch, compare wires[i] and wires[j]
            # If wires[i] > wires[j], swap
            mask = wires[:, i] > wires[:, j]
            if not np.any(mask):
                continue
            # Swap where mask is True
            temp = wires[mask, i].copy()
            wires[mask, i] = wires[mask, j]
            wires[mask, j] = temp
        
        # Check if sorted
        # For each input, check if all adjacent pairs are non-decreasing
        for k in range(N - 1):
            if np.any(wires[:, k] > wires[:, k + 1]):
                return False
    
    return True


def verify_network(network):
    """
    Verify that the network sorts all inputs using the zero-one principle.
    We only need to check all 2^N Boolean inputs.
    For N=17, that's 131072 inputs.
    Use bit-vector simulation for efficiency.
    """
    return verify_network_fast(network)
# EVOLVE-BLOCK-END
