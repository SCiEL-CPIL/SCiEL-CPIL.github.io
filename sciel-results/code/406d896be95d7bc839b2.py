"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    def odd_even_transposition():
        comparators = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comparators.append((i, i + 1))
        return comparators
    
    NUM_WORDS = (1 << N) // 64
    total_inputs = 1 << N
    
    # Pre-compute initial wires for exhaustive verification
    indices = np.arange(total_inputs, dtype=np.uint32)
    weights = 1 << np.arange(64, dtype=np.uint64)
    initial_wires = []
    for i in range(N):
        bits_i = (indices >> i) & 1
        bits_i_64 = bits_i.astype(np.uint64)
        packed = (bits_i_64.reshape(NUM_WORDS, 64) * weights).sum(axis=1)
        initial_wires.append(packed)
    
    def verify_exhaustive(comparators):
        """Full bit-parallel verification over all 2^N inputs."""
        wires = [w.copy() for w in initial_wires]
        for (i, j) in comparators:
            wi = wires[i]
            wj = wires[j]
            wires[i] = wi & wj
            wires[j] = wi | wj
        for k in range(N - 1):
            if np.any(wires[k] & ~wires[k + 1]):
                return False
        return True
    
    def verify_sampled(comparators, sample_size=2048, seed=None):
        """Fast Monte Carlo verification using random sample of inputs."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        # Generate random sample of inputs
        sample = rng.randint(0, total_inputs, size=sample_size).astype(np.uint32)
        
        # Initialize wires for sampled inputs
        wires = []
        for i in range(N):
            bits = ((sample >> i) & 1).astype(np.uint8)
            wires.append(bits)
        
        # Apply comparators
        for (i, j) in comparators:
            wi = wires[i]
            wj = wires[j]
            wires[i] = np.minimum(wi, wj)
            wires[j] = np.maximum(wi, wj)
        
        # Check sorted
        for k in range(N - 1):
            if np.any(wires[k] > wires[k + 1]):
                return False
        return True
    
    def verify_two_stage(comparators, sample_size=1024):
        """First check with sampling, then confirm with exhaustive if it passes."""
        if not verify_sampled(comparators, sample_size=sample_size):
            return False
        return verify_exhaustive(comparators)
    
    # Start with odd-even transposition
    best_network = odd_even_transposition()
    best_count = len(best_network)
    current = list(best_network)
    
    # Phase 1: Aggressive greedy removal using sampled verification
    changed = True
    while changed and time.time() < deadline - 10.0:
        changed = False
        # Try removing from random positions, biased toward later positions
        for attempt in range(200):
            if time.time() > deadline - 10.0:
                break
            idx = random.randint(0, len(current) - 1)
            test = current[:idx] + current[idx + 1:]
            if verify_sampled(test, sample_size=1024):
                # Confirm with exhaustive
                if verify_exhaustive(test):
                    current = test
                    best_network = list(current)
                    best_count = len(current)
                    changed = True
                    break
    
    # Phase 2: 2-opt swap local search
    random.seed(42)
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 5.0 and no_improve_count < max_no_improve:
        # Try a swap
        if len(current) < 2:
            break
        idx1 = random.randint(0, len(current) - 1)
        idx2 = random.randint(0, len(current) - 1)
        if idx1 == idx2:
            continue
        idx1, idx2 = min(idx1, idx2), max(idx1, idx2)
        
        test = list(current)
        test[idx1], test[idx2] = test[idx2], test[idx1]
        
        if verify_sampled(test, sample_size=1024):
            if verify_exhaustive(test):
                current = test
                best_network = list(current)
                no_improve_count = 0
                # After a swap, try some removals
                for _ in range(50):
                    if time.time() > deadline - 5.0:
                        break
                    ridx = random.randint(0, len(current) - 1)
                    test2 = current[:ridx] + current[ridx + 1:]
                    if verify_sampled(test2, sample_size=512):
                        if verify_exhaustive(test2):
                            current = test2
                            best_network = list(current)
                            best_count = len(current)
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
    
    # Phase 3: Final intensive greedy removal with exhaustive verification
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > deadline - 2.0:
                break
            test = current[:idx] + current[idx + 1:]
            if verify_exhaustive(test):
                current = test
                best_network = list(current)
                best_count = len(current)
                changed = True
                break
    
    # Final verification
    if not verify_exhaustive(best_network):
        best_network = odd_even_transposition()
    
    return best_network
# EVOLVE-BLOCK-END
