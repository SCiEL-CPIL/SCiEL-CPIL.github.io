"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time

def _verify_network_fast(network):
    """
    Verify network sorts all 2^N Boolean inputs using the zero-one principle.
    Represents each wire's state across all 65536 inputs as a Python int bitmask.
    """
    n = N
    # Initialize wire states: wire w for input k is bit w of k
    wire_state = []
    for w in range(n):
        # Build bitmask where bit k = (k >> w) & 1 for k in 0..2^n-1
        period = 1 << (w + 1)
        half = 1 << w
        period_mask = 0
        for bit_pos in range(half, period):
            period_mask |= (1 << bit_pos)
        mask = 0
        for rep in range(1 << (n - w - 1)):
            mask |= (period_mask << (rep * period))
        wire_state.append(mask)
    
    # Apply each comparator
    for (i, j) in network:
        a = wire_state[i]
        b = wire_state[j]
        wire_state[i] = a & b  # min for 0/1
        wire_state[j] = a | b  # max for 0/1
    
    # Check sorted: wire_state[k] & ~wire_state[k+1] must be 0 for all k
    for k in range(n - 1):
        if (wire_state[k] & ~wire_state[k + 1]) != 0:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Start with odd-even transposition (120 comparators)
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    if not _verify_network_fast(network):
        return network
    
    best_network = network
    best_count = len(network)
    
    # Iterative pruning
    max_iterations = 50
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        pruned = False
        for idx in range(len(network)):
            if time.time() > deadline:
                break
            
            candidate = network[:idx] + network[idx+1:]
            if _verify_network_fast(candidate):
                network = candidate
                best_network = network
                best_count = len(network)
                pruned = True
                break
        
        if not pruned:
            break
    
    return best_network
# EVOLVE-BLOCK-END
