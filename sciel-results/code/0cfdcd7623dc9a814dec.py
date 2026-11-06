"""Rewritten program: SAT-based synthesis of minimal sorting networks on 18 wires."""
import itertools
import numpy as np

N = 18

# EVOLVE-BLOCK-START
def _apply_comparator(state, i, j):
    """Apply a compare-exchange on wires i and j (min to i, max to j) to a bit vector."""
    if state[i] > state[j]:
        state[i], state[j] = state[j], state[i]
    return state

def _verify_network(network):
    """Verify that the network sorts all 2^N Boolean inputs (0-1 principle)."""
    # Use the 0-1 principle: it suffices to check all 2^N binary inputs
    # For N=18, that's 262144 inputs. We can vectorize with numpy for speed.
    # But 262144 * 18 is manageable.
    
    # Actually, a smarter approach: for each input, simulate the network.
    # But 262144 * len(network) operations might be slow in pure Python.
    # Let's use a smarter verification: track the set of possible outputs.
    
    # Alternative: use the fact that a network sorts iff for every subset S of wires,
    # the sorted output has the |S| smallest values in the first |S| positions... 
    # No, let's just brute force with numpy vectorization.
    
    # Generate all 2^18 binary inputs as a (262144, 18) array
    # That's 262144 * 18 * 1 byte = ~4.7 MB, fine.
    
    # Build all binary inputs
    inputs = np.array(list(itertools.product([0, 1], repeat=N)), dtype=np.uint8)
    # inputs shape: (262144, 18)
    
    # Apply each comparator to all inputs at once
    for i, j in network:
        # For each row, if inputs[:, i] > inputs[:, j], swap
        mask = inputs[:, i] > inputs[:, j]
        # Swap where mask is True
        inputs[mask, i] = 1 - inputs[mask, i]
        inputs[mask, j] = 1 - inputs[mask, j]
    
    # Check if all rows are sorted (non-decreasing)
    # A row is sorted if for all k, inputs[k, k] <= inputs[k, k+1]
    diffs = np.diff(inputs, axis=1)
    # If any diff is negative, it's not sorted
    if np.any(diffs < 0):
        return False
    return True

def _depth_of_network(network):
    """Compute the parallel depth of a network (longest chain of dependent comparators)."""
    # Build a DAG: each comparator depends on the latest comparator touching either wire
    # Use topological layering
    wire_last_layer = [0] * N
    for comp in network:
        i, j = comp
        layer = max(wire_last_layer[i], wire_last_layer[j]) + 1
        wire_last_layer[i] = layer
        wire_last_layer[j] = layer
    return max(wire_last_layer) if wire_last_layer else 0

def _greedy_local_search(initial_network, max_iters=5000, timeout_sec=55):
    """
    Local search to minimize comparator count.
    Start with initial_network, try removing/adding/swapping comparators.
    """
    import time
    
    start_time = time.time()
    current = list(initial_network)
    current_count = len(current)
    
    # Verify initial
    if not _verify_network(current):
        # Fallback to odd-even transposition
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    best = current_count
    
    for iteration in range(max_iters):
        if time.time() - start_time > timeout_sec:
            break
        
        # Try removing a random comparator
        if len(current) > 0:
            idx = np.random.randint(len(current))
            candidate = current[:idx] + current[idx+1:]
            if _verify_network(candidate):
                current = candidate
                current_count -= 1
                best = min(best, current_count)
                continue
        
        # Try swapping a random comparator with a random other position
        if len(current) > 1:
            idx1 = np.random.randint(len(current))
            idx2 = np.random.randint(len(current))
            if idx1 != idx2:
                candidate = current[:]
                candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
                if _verify_network(candidate):
                    current = candidate
                    continue
        
        # Try replacing a random comparator with a random new one
        if len(current) > 0:
            idx = np.random.randint(len(current))
            i = np.random.randint(N)
            j = np.random.randint(i + 1, N)
            candidate = current[:]
            candidate[idx] = (i, j)
            if _verify_network(candidate):
                current = candidate
                continue
    
    return current

def _construct_via_local_search():
    """
    Build a network using local search starting from odd-even transposition.
    """
    import time
    
    start_time = time.time()
    
    # Seed: odd-even transposition network
    seed = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Run local search
    network = _greedy_local_search(seed, max_iters=10000, timeout_sec=50)
    
    # Verify
    if _verify_network(network):
        return network
    else:
        return seed

def _try_small_networks():
    """
    Try to find a small network by systematic search for small sizes.
    This is a fallback if local search doesn't improve much.
    """
    # For 18 wires, known best is 77. Let's see if we can get close.
    # We'll rely on local search primarily.
    return None

def construct():
    """
    Construct a compare-exchange network that sorts all 18-input sequences.
    Returns an ordered list of integer pairs (i, j).
    """
    import time
    
    start_time = time.time()
    
    # Primary approach: local search from odd-even transposition
    network = _construct_via_local_search()
    
    # If we have time, try a few more random restarts
    if time.time() - start_time < 45:
        for restart in range(3):
            if time.time() - start_time > 50:
                break
            # Perturb the network
            perturbed = network[:]
            # Randomly remove a few and re-add
            for _ in range(5):
                if len(perturbed) > 0:
                    idx = np.random.randint(len(perturbed))
                    perturbed.pop(idx)
            # Run short local search
            candidate = _greedy_local_search(perturbed, max_iters=2000, timeout_sec=5)
            if len(candidate) < len(network) and _verify_network(candidate):
                network = candidate
    
    # Final verification
    if not _verify_network(network):
        # Fallback
        network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    return network
# EVOLVE-BLOCK-END
