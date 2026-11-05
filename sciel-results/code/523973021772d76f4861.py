"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import product

N = 17

def _simulate(network, input_vec):
    """Simulate network on a 0/1 input vector. Returns final state."""
    state = list(input_vec)
    for i, j in network:
        if state[i] > state[j]:
            state[i], state[j] = state[j], state[i]
    return state

def _is_valid(network):
    """Check if network sorts all 0/1 inputs using zero-one principle."""
    n_comps = len(network)
    if n_comps == 0:
        # Only valid if N <= 1
        return N <= 1
    
    # For N=17, we have 2^17 = 131072 possible 0/1 inputs
    # We can be smarter: a network is valid iff it sorts all inputs with exactly k ones
    # for k = 0, 1, ..., N. By symmetry, we only need to check a few cases.
    # Actually, the zero-one principle says we need to check ALL 2^N inputs.
    # But we can optimize: if the network sorts all inputs with weight k, 
    # it sorts all with weight N-k by duality.
    
    # Efficient check: generate all 2^17 inputs and verify
    # This is 131072 inputs * up to ~70 comparators = ~9M operations, should be fast
    
    for bits in range(1 << N):
        input_vec = [(bits >> i) & 1 for i in range(N)]
        result = _simulate(network, input_vec)
        # Check if result is sorted: all 0s then all 1s
        seen_one = False
        for val in result:
            if val == 1:
                seen_one = True
            elif seen_one:
                return False
    return True

def _depth(network):
    """Compute parallel depth of network."""
    wire_depth = [0] * N
    for i, j in network:
        d = max(wire_depth[i], wire_depth[j]) + 1
        wire_depth[i] = d
        wire_depth[j] = d
    return max(wire_depth) if wire_depth else 0

def _construct_odd_even_transposition():
    """Standard odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _try_remove_comparators(network, max_removals=20, deadline=None):
    """Try to remove redundant comparators from the network."""
    current = list(network)
    removed = []
    
    start_time = time.time()
    attempt = 0
    
    while len(removed) < max_removals:
        if deadline and time.time() > deadline:
            break
        attempt += 1
        
        # Try removing each comparator in random order
        indices = list(range(len(current)))
        random.shuffle(indices)
        
        found_removal = False
        for idx in indices:
            if deadline and time.time() > deadline:
                break
            if idx in removed:
                continue
            # Try removing this comparator
            test_network = current[:idx] + current[idx+1:]
            if _is_valid(test_network):
                # Successfully removed
                current = test_network
                removed.append(idx)
                found_removal = True
                break
        
        if not found_removal:
            # No more single removals possible, try pairs
            # This is expensive, so only do a few attempts
            if attempt > 5:
                break
            
            found_pair_removal = False
            for i in range(len(current)):
                if deadline and time.time() > deadline:
                    break
                for j in range(i+1, len(current)):
                    if deadline and time.time() > deadline:
                        break
                    test_network = current[:i] + current[i+1:j] + current[j+1:]
                    if _is_valid(test_network):
                        current = test_network
                        # Update removed indices - this is tricky, so just restart
                        return current
                        found_pair_removal = True
                        break
                if found_pair_removal:
                    break
            if not found_pair_removal:
                break
    
    return current

def construct():
    """Construct a sorting network with as few comparators as possible."""
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    # Start with odd-even transposition
    network = _construct_odd_even_transposition()
    best_network = list(network)
    best_count = len(network)
    
    # Try to remove redundant comparators
    improved = _try_remove_comparators(network, max_removals=30, deadline=deadline)
    if len(improved) < best_count:
        best_network = improved
        best_count = len(improved)
    
    # If we still have time, try more aggressive search
    if time.time() < deadline - 10:
        # Try local search: swap adjacent comparators, add/remove
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            
            # Try random perturbations
            trial = list(best_network)
            
            # Randomly remove a few comparators
            num_removals = random.randint(1, 3)
            for _ in range(num_removals):
                if len(trial) > 0:
                    idx = random.randint(0, len(trial) - 1)
                    trial.pop(idx)
            
            # Optionally add a random comparator
            if random.random() < 0.3 and len(trial) < 200:
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                trial.append((i, j))
            
            if _is_valid(trial) and len(trial) < best_count:
                best_network = trial
                best_count = len(trial)
    
    return best_network
# EVOLVE-BLOCK-END
