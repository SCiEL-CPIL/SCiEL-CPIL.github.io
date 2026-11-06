"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from multiprocessing import Pool, cpu_count

N = 17
MAX_COMPARATORS = 4096
DEADLINE = 55.0  # seconds, leave 5s safety margin for 60s wall-clock limit

# Pre-compute initial wire bit-vectors globally for fast access
_INITIAL_WIRES = None

def _get_initial_wires():
    global _INITIAL_WIRES
    if _INITIAL_WIRES is None:
        _INITIAL_WIRES = []
        for i in range(N):
            half = 1 << i
            block = '0' * half + '1' * half
            reps = 1 << (N - i - 1)
            s = block * reps
            v = int(s, 2)
            _INITIAL_WIRES.append(v)
    return _INITIAL_WIRES

# EVOLVE-BLOCK-START
def _verify_worker(network):
    """Worker function for parallel verification."""
    if not network:
        return N <= 1
    
    wires = _get_initial_wires()[:]
    
    for i, j in network:
        wi = wires[i]
        wj = wires[j]
        mask = wi & ~wj
        wires[i] = wi & ~mask
        wires[j] = wj | mask
    
    for i in range(N - 1):
        if wires[i] & ~wires[i + 1]:
            return False
            
    return True

def _verify_network_sequential(network):
    """Sequential verification - faster for small networks due to no pool overhead."""
    return _verify_worker(network)

def _greedy_remove_sequential(net, deadline):
    """Try to remove comparators sequentially - faster for small batches."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(net)):
            if time.time() > deadline:
                return net
            candidate = net[:idx] + net[idx+1:]
            if _verify_worker(candidate):
                net = candidate
                improved = True
                break
    return net

def construct():
    """
    Returns a sorting network for N=17 wires as a list of (i, j) pairs.
    Uses hybrid sequential/parallel verification and improved search strategy.
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
    if not _verify_worker(current):
        return current
    
    current_count = len(current)
    best_count = current_count
    best_network = current[:]
    
    # Use sequential verification for initial greedy removal (faster for large batches)
    current = _greedy_remove_sequential(current, DEADLINE - 10)
    current_count = len(current)
    if current_count < best_count:
        best_count = current_count
        best_network = current[:]
    
    # Now use parallel verification for smaller batches in hill climbing
    n_workers = min(cpu_count(), 4)
    
    with Pool(n_workers) as pool:
        # Randomized Hill Climbing with parallel verification
        while time.time() - start_time < DEADLINE - 5:
            # Generate a batch of candidate moves
            batch_size = min(6, len(current))
            candidates = []
            
            # Strategy A: Removals (highest priority)
            if len(current) > 0:
                for _ in range(batch_size // 2):
                    idx = random.randint(0, len(current) - 1)
                    candidate = current[:idx] + current[idx+1:]
                    candidates.append(('remove', idx, candidate))
            
            # Strategy B: Swaps
            if len(current) >= 2:
                for _ in range(batch_size // 3):
                    i = random.randint(0, len(current) - 1)
                    j = random.randint(0, len(current) - 1)
                    if i != j:
                        candidate = current[:]
                        candidate[i], candidate[j] = candidate[j], candidate[i]
                        candidates.append(('swap', (i, j), candidate))
            
            # Strategy C: Moves (long-distance preferred)
            if len(current) > 1:
                for _ in range(batch_size // 3):
                    i = random.randint(0, len(current) - 1)
                    # Biased toward long-distance moves
                    if random.random() < 0.7:
                        j = random.randint(0, len(current) - 1)
                        while j == i:
                            j = random.randint(0, len(current) - 1)
                    else:
                        j = random.randint(0, len(current) - 1)
                        while j == i:
                            j = random.randint(0, len(current) - 1)
                    candidate = current[:]
                    comp = candidate.pop(i)
                    candidate.insert(j, comp)
                    candidates.append(('move', (i, j), candidate))
            
            if not candidates:
                break
            
            # Verify all candidates in parallel
            candidate_networks = [c[2] for c in candidates]
            results = pool.map(_verify_worker, candidate_networks)
            
            # Process results - prioritize removals
            move_success = False
            best_candidate = None
            best_type = None
            
            for (move_type, params, candidate), result in zip(candidates, results):
                if result:
                    if move_type == 'remove' or best_candidate is None:
                        best_candidate = candidate
                        best_type = move_type
                    if move_type == 'remove':
                        break
            
            if best_candidate is not None:
                current = best_candidate
                current_count = len(current)
                move_success = True
                
                if current_count < best_count:
                    best_count = current_count
                    best_network = current[:]
            
            # If no move succeeded, apply perturbation
            if not move_success:
                # Perturb: Mix of swaps and long-distance moves
                num_perturbations = random.randint(3, 6)
                for _ in range(num_perturbations):
                    pert_type = random.random()
                    if pert_type < 0.5:
                        # Swap
                        i = random.randint(0, len(current) - 1)
                        j = random.randint(0, len(current) - 1)
                        if i != j:
                            current[i], current[j] = current[j], current[i]
                    else:
                        # Move
                        i = random.randint(0, len(current) - 1)
                        j = random.randint(0, len(current) - 1)
                        if i != j:
                            comp = current.pop(i)
                            current.insert(j, comp)
                
                # Try to greedily remove from the perturbed state (sequential for speed)
                perturbed_current = _greedy_remove_sequential(current, DEADLINE - 5)
                if len(perturbed_current) < current_count:
                    current = perturbed_current
                    current_count = len(current)
                    if current_count < best_count:
                        best_count = current_count
                        best_network = current[:]

    # Return the best network found
    return best_network


def verify_network_fast(network):
    """
    Verify network using bit-vector simulation with Python integers.
    """
    return _verify_worker(network)


def verify_network(network):
    """
    Verify that the network sorts all inputs using the zero-one principle.
    """
    return verify_network_fast(network)
# EVOLVE-BLOCK-END
