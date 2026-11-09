"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 16

# EVOLVE-BLOCK-START

def _apply_network(network, state):
    """Apply the comparator network to a boolean state (list of 0/1)."""
    state = list(state)
    for i, j in network:
        if state[i] > state[j]:
            state[i], state[j] = state[j], state[i]
    return state

def _verify_network(network):
    """Verify network sorts all 2^N boolean inputs using zero-one principle."""
    for mask in range(1 << N):
        state = [(mask >> k) & 1 for k in range(N)]
        result = _apply_network(network, state)
        # Check sorted: non-increasing (since min goes to lower index, max to higher)
        # Wait: min to i, max to j where i<j. So wire i gets smaller, wire j gets larger.
        # Sorted order for boolean: 1s then 0s? No.
        # If input is [1, 0, 1, 0], after sorting with min to low index:
        # min(1,0)=0 goes to low index, max=1 goes to high index.
        # So sorted array should be [0, 0, 1, 1] (non-decreasing).
        # Let's check: is result non-decreasing?
        for k in range(N - 1):
            if result[k] > result[k+1]:
                return False
    return True

def _count_comparators(network):
    return len(network)

def _parallel_depth(network):
    """Compute parallel depth of the network."""
    if not network:
        return 0
    # Assign each comparator a level.
    # A comparator (i, j) can be executed in parallel with (k, l) if 
    # {i, j} and {k, l} are disjoint.
    # This is a graph coloring problem on comparators (conflict graph).
    # For simplicity and speed, we use a greedy approach or just return an upper bound.
    # Actually, depth is the minimum number of layers.
    # Let's use a simple heuristic: assign each comparator to the earliest layer 
    # where it doesn't conflict with existing comparators in that layer.
    levels = [0] * len(network)
    for idx in range(len(network)):
        i, j = network[idx]
        # Find earliest level
        level = 0
        while True:
            conflict = False
            for prev_idx in range(idx):
                if levels[prev_idx] == level:
                    k, l = network[prev_idx]
                    if i in (k, l) or j in (k, l):
                        conflict = True
                        break
            if not conflict:
                levels[idx] = level
                break
            level += 1
    return max(levels) + 1

# Known good networks for N=16
# The reference is 60 comparators. We can try to find one via search or use a known construction.
# A common construction is the bitonic sorting network, which for N=16 has 120 comparators (same as odd-even transposition).
# We need something better.
# Let's try a simple local search starting from odd-even transposition to reduce comparators.

def _generate_initial_network():
    # Odd-even transposition
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _local_search(network, time_limit):
    """Try to reduce the number of comparators by removing or swapping."""
    start_time = time.time()
    current_network = list(network)
    current_count = len(current_network)
    
    # Verify initial network
    if not _verify_network(current_network):
        return network, current_count
    
    best_network = list(current_network)
    best_count = current_count
    
    # Strategy: Try removing comparators one by one, or swapping with others.
    # Since 2^16 checks are expensive, we need to be careful.
    # 65536 * 120 ops = ~7.8M ops per verification. 
    # 1000 verifications = 7.8B ops. Too slow in Python.
    # We need a faster verification or fewer checks.
    # Zero-one principle requires all 2^N. 
    # Maybe we can use a smarter check?
    # Or limit the search to a few hundred steps.
    
    # Let's try a different approach: use a known better network.
    # For N=16, the optimal is 60. 
    # We can hardcode a known 60-comparator network if we can find one.
    # Or we can use a heuristic construction.
    
    # Let's try to generate a network using a divide-and-conquer approach (merge sort).
    # Merge sort network for N=16:
    # Split into two 8-wire networks, sort each, then merge.
    # 8-wire merge sort network has 31 comparators? No, 8-wire odd-even is 32.
    # Optimal 8-wire is 19.
    # This is getting complicated.
    
    # Let's stick with a simple hill-climbing that tries to remove comparators.
    # To speed up verification, we can precompute the effect of the network on all inputs?
    # No, 2^16 * 16 = 1M states. We can represent the state as an integer.
    
    def _fast_verify(net):
        # Use numpy for speed?
        # Let's try a simple loop first and see if it's fast enough.
        for mask in range(1 << N):
            state = [0] * N
            for k in range(N):
                state[k] = (mask >> k) & 1
            for i, j in net:
                if state[i] > state[j]:
                    state[i], state[j] = state[j], state[i]
            # Check sorted
            for k in range(N - 1):
                if state[k] > state[k+1]:
                    return False
        return True

    # Try removing each comparator
    for idx in range(len(current_network) - 1, -1, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Try removing comparator at idx
        test_net = current_network[:idx] + current_network[idx+1:]
        if _fast_verify(test_net):
            current_network = test_net
            current_count -= 1
            # Restart search from the beginning? Or continue?
            # Continue, but restart idx to 0 to try removing more.
            idx = len(current_network) - 1
        
    return current_network, current_count

# Let's try to find a better network using a known algorithm.
# The "Batcher's bitonic sort" is not optimal.
# There are known optimal networks.
# For N=16, one optimal network is described in literature.
# We can try to construct it.

def _construct_optimal_16():
    # This is a known 60-comparator network for 16 wires.
    # Source: Various sorting network databases.
    # Let's try to generate it using a simple recursive merge.
    # Actually, let's just return a hardcoded list if we can find one.
    # Since I don't have it memorized, I'll use a heuristic search.
    
    # Let's try a random restart hill climbing.
    # But verification is slow.
    
    # Alternative: Use the fact that the network must sort all 2^16 inputs.
    # We can use a SAT solver? No, standard library only.
    
    # Let's try a simple genetic algorithm.
    # Population of networks, fitness = -comparator count if valid, else large penalty.
    
    # Due to time constraints, I will return the odd-even transposition network
    # but attempt a quick local search to remove redundant comparators.
    
    net = _generate_initial_network()
    # Try to remove comparators
    for _ in range(100): # Limit iterations
        removed = False
        for idx in range(len(net) - 1, -1, -1):
            test_net = net[:idx] + net[idx+1:]
            if _verify_network(test_net):
                net = test_net
                removed = True
                break
        if not removed:
            break
    return net

# EVOLVE-BLOCK-END

def construct():
    # Use the heuristic search to find a better network
    start_time = time.time()
    time_limit = 50  # Leave 10s margin
    
    # Start with odd-even transposition
    net = _generate_initial_network()
    
    # Try to improve
    improved_net, count = _local_search(net, time_limit)
    
    # If we ran out of time, return the best we have
    return improved_net
