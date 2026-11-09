import numpy as np
import time

N = 16
DEADLINE = 85.0  # seconds (90s limit, leave 5s margin)

def _elapsed(start):
    return time.time() - start

def _apply_comparators(state, comps):
    """
    Apply a list of comparators (i, j) to a state vector.
    Comparator (i, j): min goes to i, max goes to j.
    Returns new state.
    """
    state = state.copy()
    for i, j in comps:
        if state[i] > state[j]:
            state[i], state[j] = state[j], state[i]
    return state

def _verify_network_fast(comps):
    """
    Verify if the network sorts all 2^N binary inputs using the zero-one principle.
    Since N=16, 2^16 = 65536 inputs. This is manageable.
    Returns True if valid, False otherwise.
    """
    # Pre-compute the permutation logic or just run it?
    # Running 65536 * len(comps) operations.
    # If len(comps) ~ 60, 65536 * 60 = 3.9M operations.
    # In Python, this might take 1-2 seconds.
    # In NumPy, we can vectorize if we treat inputs as a matrix.
    
    # Let's use NumPy for speed.
    # Inputs: 65536 x 16 matrix of 0s and 1s.
    # We can generate all 2^16 patterns efficiently.
    
    # Generate all 2^N patterns
    # np.arange(2**N).reshape(-1, 1) // np.power(2, N-1 - np.arange(N)) % 2
    patterns = np.arange(2**N).reshape(-1, 1) // np.power(2, N - 1 - np.arange(N)) % 2
    
    # Apply comparators
    # We need to update the matrix in place for each comparator.
    # Comparator (i, j): 
    #   min = np.minimum(patterns[:, i], patterns[:, j])
    #   max = np.maximum(patterns[:, i], patterns[:, j])
    #   patterns[:, i] = min
    #   patterns[:, j] = max
    
    for i, j in comps:
        min_val = np.minimum(patterns[:, i], patterns[:, j])
        max_val = np.maximum(patterns[:, i], patterns[:, j])
        patterns[:, i] = min_val
        patterns[:, j] = max_val
    
    # Check if all rows are sorted
    # A row is sorted if patterns[:, 0] <= patterns[:, 1] <= ... <= patterns[:, N-1]
    # Since binary, this means all 0s come before 1s.
    # Check: for each row, find the first 1. All elements before it must be 0, all after must be 1.
    # Or simply check if np.diff(patterns, axis=1) has any -1.
    # If sorted, diff is either 0 or 1. If unsorted, there is a 1 followed by 0, so diff is -1.
    
    diffs = np.diff(patterns, axis=1)
    # If any diff is -1, it's not sorted.
    if np.any(diffs < 0):
        return False
    return True

def _baseline_network():
    """
    Generate the Odd-Even Transposition network.
    N phases, each phase has N/2 comparators.
    Total N^2/2 = 128 comparators for N=16?
    Let's check:
    Phase 0: (0,1), (2,3), ... (14,15) -> 8 comparators
    Phase 1: (1,2), (3,4), ... (13,14) -> 7 comparators
    ...
    Phase 14: (14,15) -> 1 comparator
    Phase 15: (15,16)? No, indices 0..15.
    Odd-Even Transposition for N wires has N phases.
    Phase p:
      if p even: (0,1), (2,3), ...
      if p odd: (1,2), (3,4), ...
    
    Total comparators = 8 + 7 + 8 + 7 + ...
    For N=16:
    8+7+8+7+8+7+8+7+8+7+8+7+8+7+8+7 = 8*8 + 7*8 = 64 + 56 = 120.
    Yes, 120 comparators.
    """
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _all_pairs():
    """
    Generate all possible comparators (i, j) with i < j.
    """
    return [(i, j) for i in range(N) for j in range(i + 1, N)]

def construct():
    start_time = time.time()
    
    # 1. Start with Baseline (120 comparators)
    current_best = _baseline_network()
    best_score = len(current_best)
    
    # Quick verify
    if not _verify_network_fast(current_best):
        # Should not happen, but just in case
        pass
    
    # 2. Greedy Pruning
    # Try to remove comparators one by one if the network remains valid.
    # Order matters. We can try random orders or specific heuristics.
    
    def greedy_prune(network, time_limit=20.0):
        # Work on a copy
        net = list(network)
        changed = True
        while changed:
            changed = False
            # Shuffle indices to try different orders?
            # For now, just iterate through current list.
            for idx in range(len(net) - 1, -1, -1):
                if time.time() - start_time > time_limit:
                    return net
                # Try removing net[idx]
                candidate = net[:idx] + net[idx+1:]
                # Verify
                if _verify_network_fast(candidate):
                    net = candidate
                    changed = True
        return net
    
    # Run greedy prune with a time limit
    pruned = greedy_prune(current_best, time_limit=15.0)
    if len(pruned) < best_score:
        current_best = pruned
        best_score = len(current_best)
    
    # 3. Random Local Search / Mutation
    # We can try to replace comparators or add/remove them.
    # Given the time, let's try to do a few restarts of greedy prune with shuffled initial order.
    
    for restart in range(5):
        if time.time() - start_time > 60.0:
            break
            
        # Shuffle current best
        shuffled = list(current_best)
        np.random.shuffle(shuffled)
        
        pruned = greedy_prune(shuffled, time_limit=5.0)
        if len(pruned) < best_score:
            current_best = pruned
            best_score = len(current_best)
            
    # 4. Final Verification
    if _verify_network_fast(current_best):
        return current_best
    else:
        return _baseline_network()

if __name__ == "__main__":
    net = construct()
    print(f"Found network with {len(net)} comparators.")
    if _verify_network_fast(net):
        print("Network is valid.")
    else:
        print("Network is INVALID!")
