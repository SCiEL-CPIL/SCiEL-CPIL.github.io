"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
TIME_LIMIT = 60.0  # seconds
SAFETY_MARGIN = 5.0  # seconds to leave before the hard limit

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network on N wires.
    
    Strategy:
    1. Start with a known-correct network (odd-even transposition, 153 comparators).
    2. Use local search to remove redundant comparators while maintaining correctness.
    3. Use simulated annealing to explore the space of networks.
    
    Returns a list of (i, j) pairs where i < j, representing compare-exchange operations.
    """
    start_time = time.time()
    
    # Generate the initial odd-even transposition network
    def odd_even_transposition():
        """Generate odd-even transposition sorting network."""
        comparators = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comparators.append((i, i + 1))
        return comparators
    
    # Check if a network sorts all 2^N boolean inputs
    def is_sorting_network(comparators, timeout=None):
        """Check if the network sorts all boolean inputs."""
        if timeout is not None:
            if time.time() - start_time > timeout:
                return None  # timeout
        
        n = len(comparators)
        if n > MAX_COMPARATORS:
            return False
        
        # For each of the 2^N boolean inputs, check if the network sorts it
        # An input is sorted if the output has all 0s followed by all 1s
        # i.e., output[k] <= output[k+1] for all k
        
        # We can use a smarter check: a network is a sorting network iff
        # it sorts all 0-1 inputs. For a 0-1 input with k ones, the sorted
        # output should have the first N-k entries as 0 and last k as 1.
        
        # Efficient approach: simulate all 2^N inputs using bit operations
        # Represent each wire's value as a bit in an integer
        
        # For N=18, we have 2^18 = 262144 inputs
        # We can represent the state as an 18-bit integer
        
        # Actually, let's use a different approach:
        # A network is sorting iff for every subset S of wires (representing 
        # which wires have value 1), the output has exactly |S| ones, and
        # they are in the highest |S| positions.
        
        # More efficiently: we only need to check that the network preserves
        # the count of 1s and puts them in the right positions.
        
        # Let's simulate: for each input mask, apply the comparators and check
        # if the result is sorted (all 0s then all 1s).
        
        # Optimization: we can represent the state as a tuple of bits or as an integer
        # where bit i represents wire i.
        
        # For each comparator (i, j), if bit i > bit j, swap them.
        # In terms of bits: if bit i is 1 and bit j is 0, swap.
        
        # Let's use integer representation for speed.
        
        # Actually, for N=18, 2^18 = 262144 inputs, and each input requires
        # up to ~150 comparator operations. That's about 39 million operations,
        # which might be slow in Python. Let's use NumPy for vectorization.
        
        # Alternative: use the 0-1 principle more cleverly.
        # We can check correctness by verifying that the network sorts all
        # "threshold" inputs: inputs where wires 0..k-1 are 0 and wires k..N-1 are 1.
        # Wait, that's not sufficient. The 0-1 principle says we need to check all
        # 2^N binary inputs.
        
        # Let's use NumPy to vectorize over all inputs at once.
        
        # State: shape (2^N, N) array of 0s and 1s
        # For each comparator, update all states simultaneously.
        
        # This uses 2^18 * 18 * 8 bytes = 36 MB, which is fine.
        
        # But doing this for each candidate network during search might be slow.
        # Let's optimize: we can represent the state as an array of uint8 or bool.
        
        # Actually, let's use a more efficient representation:
        # Each input is an 18-bit integer. We have 2^18 such integers.
        # We can store them in a uint32 array of size 2^18.
        
        # For a comparator (i, j): 
        #   bit_i = (state >> i) & 1
        #   bit_j = (state >> j) & 1
        #   if bit_i > bit_j (i.e., bit_i=1, bit_j=0), swap
        #   new_state = state with bits i and j swapped if needed
        
        # This can be done with bit operations on the entire array at once.
        
        # Let me think about this more carefully.
        
        # For all states simultaneously:
        # mask_i = 1 << i
        # mask_j = 1 << j
        # bit_i = states & mask_i  # non-zero where wire i is 1
        # bit_j = states & mask_j  # non-zero where wire j is 1
        # 
        # We want to swap bits i and j where bit_i=1 and bit_j=0.
        # Condition: (states & mask_i) != 0 and (states & mask_j) == 0
        # 
        # To swap: clear both bits, then set them swapped.
        # 
        # Let's define:
        # swap_mask = states where (bit_i=1 and bit_j=0) or (bit_i=0 and bit_j=1)
        # Actually, we only swap when bit_i > bit_j, i.e., bit_i=1, bit_j=0.
        
        # Hmm, this is getting complex. Let me just use the straightforward
        # NumPy approach with a 2D array.
        
        # Actually, for the search, we need to be fast. Let me use a different
        # approach: use the fact that we only need to check a subset of inputs
        # if we're confident, or use a randomized check.
        
        # For now, let's use a full check but optimized with NumPy.
        
        # Create all 2^N inputs as a 2D array
        # This is expensive to create each time, so let's precompute.
        
        # Actually, let me reconsider. The full check is 2^18 * num_comparators
        # operations. With ~150 comparators, that's ~40M operations per check.
        # In NumPy, this might take ~0.1-0.5 seconds. For local search with
        # hundreds of iterations, this is feasible within 60s.
        
        # Let's precompute the all-inputs array once.
        
        # For efficiency, let's represent the state as a 2D uint8 array of shape (2^N, N).
        
        # Precompute all inputs
        all_inputs = np.arange(1 << N, dtype=np.uint32)
        # Expand to bits
        # states[i, j] = bit j of input i
        # Use broadcasting: for each bit position j, extract from all_inputs
        bit_positions = np.arange(N, dtype=np.uint32)
        # states has shape (2^N, N)
        # states[i, j] = (all_inputs[i] >> j) & 1
        states = ((all_inputs[:, None] >> bit_positions[None, :]) & 1).astype(np.uint8)
        
        # Now apply each comparator
        for (i, j) in comparators:
            # For each row, if states[row, i] > states[row, j], swap
            # i.e., if states[row, i] == 1 and states[row, j] == 0, swap
            swap_mask = (states[:, i] == 1) & (states[:, j] == 0)
            if np.any(swap_mask):
                states[swap_mask, i] = 0
                states[swap_mask, j] = 1
        
        # Check if all rows are sorted: states[row, k] <= states[row, k+1] for all k
        # i.e., no row has a 1 followed by a 0
        # sorted_check: for each row, all 0s come before all 1s
        # This is equivalent to: for each row, the sum of the first k elements
        # equals the total sum for all k >= total_sum
        # Or more simply: the row is non-decreasing
        
        # Check: states[:, :-1] <= states[:, 1:]
        is_sorted = np.all(states[:, :-1] <= states[:, 1:], axis=1)
        return bool(np.all(is_sorted))
    
    # Start with the odd-even transposition network
    network = odd_even_transposition()
    
    # Verify it's correct (it should be)
    if not is_sorting_network(network):
        # Shouldn't happen, but just in case
        return network
    
    best_network = list(network)
    best_score = len(best_network)
    
    # Local search: try removing each comparator and check if still correct
    # Also try swapping adjacent comparators, etc.
    
    def try_remove(comparator_index):
        """Try removing the comparator at the given index."""
        new_network = network[:comparator_index] + network[comparator_index + 1:]
        if is_sorting_network(new_network):
            return new_network
        return None
    
    def try_swap(comparator_index):
        """Try swapping the comparator at the given index with the next one."""
        if comparator_index >= len(network) - 1:
            return None
        new_network = network.copy()
        new_network[comparator_index], new_network[comparator_index + 1] = \
            new_network[comparator_index + 1], new_network[comparator_index]
        if is_sorting_network(new_network):
            return new_network
        return None
    
    # Simulated annealing
    current_network = list(network)
    current_score = len(current_network)
    best_network = list(current_network)
    best_score = current_score
    
    temperature = 100.0
    cooling_rate = 0.95
    min_temperature = 0.1
    
    random.seed(42)
    
    while temperature > min_temperature:
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Random move
        move_type = random.random()
        
        if move_type < 0.7:
            # Try removing a random comparator
            if len(current_network) > 1:
                idx = random.randint(0, len(current_network) - 1)
                new_network = current_network[:idx] + current_network[idx + 1:]
                if is_sorting_network(new_network):
                    new_score = len(new_network)
                    delta = new_score - current_score
                    if delta < 0 or random.random() < np.exp(-delta / temperature):
                        current_network = new_network
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_network = list(current_network)
        elif move_type < 0.85:
            # Try swapping two adjacent comparators
            if len(current_network) > 1:
                idx = random.randint(0, len(current_network) - 2)
                new_network = current_network.copy()
                new_network[idx], new_network[idx + 1] = \
                    new_network[idx + 1], new_network[idx]
                if is_sorting_network(new_network):
                    new_score = len(new_network)
                    delta = new_score - current_score
                    if delta < 0 or random.random() < np.exp(-delta / temperature):
                        current_network = new_network
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_network = list(current_network)
        else:
            # Try moving a comparator to a random position
            if len(current_network) > 2:
                idx1 = random.randint(0, len(current_network) - 1)
                idx2 = random.randint(0, len(current_network) - 1)
                if idx1 != idx2:
                    new_network = current_network.copy()
                    comp = new_network.pop(idx1)
                    new_network.insert(idx2, comp)
                    if is_sorting_network(new_network):
                        new_score = len(new_network)
                        delta = new_score - current_score
                        if delta < 0 or random.random() < np.exp(-delta / temperature):
                            current_network = new_network
                            current_score = new_score
                            if current_score < best_score:
                                best_score = current_score
                                best_network = list(current_network)
        
        temperature *= cooling_rate
    
    # Final: try to remove more comparators from the best network found
    current_network = list(best_network)
    changed = True
    while changed and time.time() - start_time < TIME_LIMIT - SAFETY_MARGIN:
        changed = False
        for idx in range(len(current_network)):
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            new_network = current_network[:idx] + current_network[idx + 1:]
            if is_sorting_network(new_network):
                current_network = new_network
                changed = True
                break
    
    return current_network
# EVOLVE-BLOCK-END
