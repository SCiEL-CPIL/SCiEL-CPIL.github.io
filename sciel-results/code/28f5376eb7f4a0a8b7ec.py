"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 18

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55.0  # Leave 5s buffer for safety
    
    # 1. Input Generation
    inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for col in range(N):
        period = 1 << (col + 1)
        half = 1 << col
        block = np.zeros(period, dtype=np.uint8)
        block[half:] = 1
        repeats = (1 << N) // period
        inputs[:, col] = np.tile(block, repeats)
    
    # 2. Fast Verification Setup
    rng_seed = 42
    rng = np.random.RandomState(rng_seed)
    
    # Use a smaller subset for fast verification
    sample_size = 4096
    hard_indices = rng.choice(1 << N, size=sample_size, replace=False)
    hard_inputs = inputs[hard_indices]
    
    state_buffer_full = np.empty_like(inputs)
    state_buffer_hard = np.empty_like(hard_inputs)
    
    def verify_full(net):
        """Verify against all 2^N inputs. Returns True if sorted."""
        np.copyto(state_buffer_full, inputs)
        for i, j in net:
            mi = state_buffer_full[:, i] & state_buffer_full[:, j]
            ma = state_buffer_full[:, i] | state_buffer_full[:, j]
            state_buffer_full[:, i] = mi
            state_buffer_full[:, j] = ma
        return np.all(state_buffer_full[:, :-1] <= state_buffer_full[:, 1:])
    
    def verify_hard(net):
        """Verify against the subset of hard inputs. Fast filter."""
        np.copyto(state_buffer_hard, hard_inputs)
        for i, j in net:
            mi = state_buffer_hard[:, i] & state_buffer_hard[:, j]
            ma = state_buffer_hard[:, i] | state_buffer_hard[:, j]
            state_buffer_hard[:, i] = mi
            state_buffer_hard[:, j] = ma
        return np.all(state_buffer_hard[:, :-1] <= state_buffer_hard[:, 1:])
    
    def _odd_even_network():
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Start with odd-even transposition as baseline
    best_network = _odd_even_network()
    best_count = len(best_network)
    
    # Verify initial network
    if not verify_full(best_network):
        return best_network
    
    # Local search: try to remove or modify comparators to reduce size
    # Strategy: iteratively try removing each comparator, check if still valid
    # If valid, keep the removal. Otherwise, try swapping with other positions.
    
    current_network = list(best_network)
    current_count = best_count
    
    # Helper: try all possible single-comparator removals
    def try_remove_comparators(net, max_iterations=1000):
        """
        Try to remove comparators one by one.
        For each comparator, check if removing it keeps the network valid.
        If yes, remove it and continue. If no, skip it.
        """
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            for idx in range(len(net) - 1, -1, -1):
                if time.time() > deadline - 3.0:
                    return net, False
                # Try removing this comparator
                candidate = net[:idx] + net[idx+1:]
                if verify_hard(candidate):
                    if verify_full(candidate):
                        net = candidate
                        improved = True
                        break
        return net, improved
    
    # Helper: try swapping two comparators
    def try_swap_comparators(net, max_iterations=500):
        """
        Try swapping pairs of comparators to see if it helps.
        This is more expensive, so use sparingly.
        """
        improved = False
        iterations = 0
        while iterations < max_iterations:
            iterations += 1
            if time.time() > deadline - 3.0:
                return net, False
            # Try random swaps
            if len(net) < 2:
                break
            i = rng.randint(len(net))
            j = rng.randint(len(net))
            if i == j:
                continue
            # Skip if swapping identical comparators
            if net[i] == net[j]:
                continue
            candidate = list(net)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            if verify_hard(candidate):
                if verify_full(candidate):
                    net = candidate
                    improved = True
        return net, improved
    
    # Helper: try replacing a comparator with a different one
    def try_replace_comparator(net, max_iterations=1000):
        """
        Try replacing random comparators with different ones.
        This can help escape local minima.
        """
        improved = False
        iterations = 0
        while iterations < max_iterations:
            iterations += 1
            if time.time() > deadline - 3.0:
                return net, False
            if not net:
                break
            idx = rng.randint(len(net))
            # Generate a random new comparator
            new_i = rng.randint(N)
            new_j = rng.randint(N)
            if new_i == new_j:
                new_j = (new_j + 1) % N
            if new_i > new_j:
                new_i, new_j = new_j, new_i
            # Skip if same as existing
            if (new_i, new_j) == net[idx]:
                continue
            candidate = list(net)
            candidate[idx] = (new_i, new_j)
            if verify_hard(candidate):
                if verify_full(candidate):
                    net = candidate
                    improved = True
        return net, improved
    
    # Main local search loop
    search_phase = 0
    while time.time() < deadline - 2.0:
        search_phase += 1
        
        if search_phase % 10 == 1:
            # Phase 1: Remove comparators
            current_network, improved = try_remove_comparators(current_network)
            if improved:
                best_network = current_network
                best_count = len(current_network)
        
        elif search_phase % 10 == 2:
            # Phase 2: Swap comparators
            current_network, improved = try_swap_comparators(current_network)
            if improved:
                best_network = current_network
                best_count = len(current_network)
        
        else:
            # Phase 3: Replace comparators (more aggressive)
            current_network, improved = try_replace_comparator(current_network)
            if improved:
                best_network = current_network
                best_count = len(current_network)
        
        # Occasionally restart from best to avoid getting stuck
        if search_phase % 50 == 0:
            current_network = list(best_network)
    
    # Final verification
    if not verify_full(best_network):
        best_network = _odd_even_network()
    
    return best_network

# EVOLVE-BLOCK-END
