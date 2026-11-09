"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 18

# EVOLVE-BLOCK-START

def _verify_network(net):
    """
    Verify that the network sorts all 2^N Boolean inputs using the 0-1 principle.
    Uses vectorized NumPy operations for speed.
    Returns True if valid, False otherwise.
    """
    if not net:
        return False
    
    # Check each comparator for validity
    for i, j in net:
        if not (0 <= i < j < N):
            return False

    # We need to check all 2^N inputs.
    # Since N=18, 2^18 = 262144.
    # We represent the state of all inputs simultaneously.
    # bits[k, :] represents the value on wire k for all inputs.
    # Shape: (N, 2^N)
    
    # Generate all inputs
    # Instead of storing 2^N columns explicitly which might be memory heavy if N was larger,
    # for N=18 it is 18 * 262144 * 1 byte ~ 4.5 MB, which is fine.
    
    # Initialize bits array
    # bit k of input x is (x >> k) & 1
    # We can generate the full matrix of inputs
    # inputs = np.arange(2**N)
    # bits = ((inputs[:, None] >> np.arange(N)[None, :]) & 1).T
    # This creates a (N, 2^N) array.
    
    # Let's be more memory efficient if possible, but 4.5MB is small.
    # Let's use uint8 for bits.
    
    # Efficient generation:
    # We can build the bits incrementally or use the shift trick.
    # The shift trick creates a large intermediate array of shape (2^N, N).
    # Let's try it.
    
    try:
        inputs = np.arange(2**N, dtype=np.uint32)
        # (2^N, N) array
        # bits = ((inputs[:, None] >> np.arange(N)) & 1).astype(np.uint8)
        # Then transpose to (N, 2^N)
        # Or keep as (2^N, N) and operate on columns?
        # Operating on columns in NumPy is slower than rows.
        # So let's make it (N, 2^N).
        
        # Vectorized creation
        # Note: np.arange(N) needs to be broadcastable.
        # inputs[:, None] is (2^N, 1)
        # np.arange(N) is (N,)
        # Result is (2^N, N)
        
        bits = ((inputs[:, None] >> np.arange(N)) & 1).T.copy()
        
    except MemoryError:
        # Fallback to slower method if memory is an issue (unlikely for N=18)
        bits = np.zeros((N, 2**N), dtype=np.uint8)
        for k in range(N):
            bits[k, :] = (np.arange(2**N) >> k) & 1

    # Apply comparators
    for i, j in net:
        # min goes to i, max goes to j
        # For bits: min = AND, max = OR
        # bits[i] = bits[i] & bits[j]
        # bits[j] = bits[i] | bits[j]  -- WAIT, this uses the updated bits[i]?
        # No, we need to use the original values.
        
        # Let's store original values
        bi = bits[i]
        bj = bits[j]
        
        bits[i] = bi & bj
        bits[j] = bi | bj

    # Check if sorted
    # A sequence is sorted (non-decreasing) if for all k < N-1, bits[k] <= bits[k+1]
    # Since bits are 0 or 1, this means bits[k] == 1 implies bits[k+1] == 1.
    # Or simply: no 1 is followed by a 0.
    # i.e., ~bits[k] & bits[k+1] should be 0 for all k.
    
    # We can check this vectorized across all inputs.
    # For each k from 0 to N-2:
    # if np.any(bits[k] > bits[k+1]): return False
    
    # To be even faster, we can sum up violations?
    # Or just check each pair.
    
    for k in range(N - 1):
        # Check if any input has a 1 on wire k and a 0 on wire k+1
        # violation = (bits[k] == 1) & (bits[k+1] == 0)
        # if np.any(violation): return False
        
        # bits[k] > bits[k+1] is equivalent for 0/1
        if np.any(bits[k] > bits[k+1]):
            return False
            
    return True


def _depth(net):
    """Compute the parallel depth of the network."""
    if not net:
        return 0
    wire_layer = [-1] * N
    max_depth = 0
    for i, j in net:
        l = max(wire_layer[i], wire_layer[j]) + 1
        wire_layer[i] = l
        wire_layer[j] = l
        if l > max_depth:
            max_depth = l
    return max_depth


def construct():
    """
    Construct a sorting network on 18 wires.
    Strategy:
    1. Start with Odd-Even Transposition Network (known correct, ~171 comparators).
    2. Use a greedy removal algorithm to remove redundant comparators.
    3. Use a local search (swap/insert/delete) to further optimize, if time permits.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    def odd_even_transposition():
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    best_net = odd_even_transposition()
    best_count = len(best_net)
    
    # Phase 1: Greedy Removal
    # Try to remove each comparator if the network remains valid.
    # We do multiple passes until no more removals are possible.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate backwards so indices remain valid if we remove
        for idx in range(len(best_net) - 1, -1, -1):
            if time.time() > deadline:
                return best_net
            
            # Try removing comparator at idx
            candidate = best_net[:idx] + best_net[idx+1:]
            
            if _verify_network(candidate):
                best_net = candidate
                best_count = len(best_net)
                improved = True
                # Restart the loop to check the new network from the beginning (or end)
                # Since we removed one, the list is shorter.
                # Breaking and restarting the while loop is cleaner.
                break
    
    # Phase 2: Local Search / Simulated Annealing
    # If we have time left, try to swap, insert, or delete comparators to find a better (shorter) network.
    # Note: Greedy removal often gets stuck in local minima.
    # A simple improvement: try to move comparators to different positions?
    # Or try to replace a comparator (i,j) with (i,k) or (j,k)?
    
    # Given the time constraints and the difficulty of sorting network optimization,
    # greedy removal is the most reliable way to get a significant reduction from the seed.
    # The seed is ~171. Greedy removal can typically get this down to ~120-130 for N=18?
    # Actually, known best is 77. We might not reach 77, but we should beat 153.
    
    # Let's add a simple "move" heuristic if time permits.
    # Pick a random comparator, try moving it to a random other position.
    # If valid and shorter? No, moving doesn't change length.
    # We need to find shorter networks.
    
    # A better heuristic: Random Restart with Greedy Removal.
    # 1. Take current best_net.
    # 2. Randomly add a few dummy comparators? No.
    # 3. Randomly swap a few comparators to escape local minima, then run greedy removal again.
    
    # Let's implement a simple random perturbation + greedy removal loop.
    
    import random
    random.seed(42) # For reproducibility in debugging, but maybe random is better?
    
    current_net = best_net[:]
    current_len = len(current_net)
    
    while time.time() < deadline:
        # Create a perturbed version
        perturbed = current_net[:]
        
        # Perform a few random mutations
        # Mutation 1: Swap two comparators
        if len(perturbed) > 1:
            i1 = random.randint(0, len(perturbed) - 1)
            i2 = random.randint(0, len(perturbed) - 1)
            perturbed[i1], perturbed[i2] = perturbed[i2], perturbed[i1]
            
        # Mutation 2: Change a wire in a comparator (risky, might break validity easily)
        # Let's stick to swapping and maybe inserting/removing?
        # Inserting a comparator makes it longer, which we don't want unless it allows more removals.
        
        # Run greedy removal on the perturbed network
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(len(perturbed) - 1, -1, -1):
                if time.time() > deadline:
                    break
                candidate = perturbed[:idx] + perturbed[idx+1:]
                if _verify_network(candidate):
                    perturbed = candidate
                    improved = True
                    break
        
        # If the perturbed+reduced network is shorter than our best, update best
        if len(perturbed) < current_len:
            current_net = perturbed
            current_len = len(perturbed)
            best_net = current_net
            best_count = current_len
            
        # If we are close to the limit, stop
        if time.time() > deadline - 10:
            break

    return best_net

# EVOLVE-BLOCK-END
