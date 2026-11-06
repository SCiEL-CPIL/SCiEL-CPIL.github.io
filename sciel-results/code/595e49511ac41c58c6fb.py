"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    rng = random.Random(42)

    def odd_even_transposition():
        comparators = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comparators.append((i, i + 1))
        return comparators

    def batcher_odd_even():
        """Batcher's odd-even merge sort network (more compact starting point)."""
        comparators = []
        
        def odd_even_merge(n, offset=0):
            if n <= 1:
                return
            half = n // 2
            odd_even_merge(half, offset)
            odd_even_merge(half, offset + half)
            for i in range(half):
                comparators.append((offset + i, offset + i + half))
        
        def odd_even_sort(n, offset=0):
            if n <= 1:
                return
            for i in range(0, n, 2):
                comparators.append((offset + i, offset + i + 1))
            odd_even_merge(n, offset)
        
        odd_even_sort(N, 0)
        return comparators

    def random_dense_network(size):
        """Generate a random dense network."""
        comparators = []
        for _ in range(size):
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            comparators.append((i, j))
        return comparators

    # Pre-compute bit masks for initialization
    NUM_WORDS = (1 << N) // 64

    # Pre-compute initial wires
    initial_wires = [None] * N
    total_inputs = 1 << N
    indices = np.arange(total_inputs, dtype=np.uint32)
    weights = 1 << np.arange(64, dtype=np.uint64)
    for i in range(N):
        bits_i = (indices >> i) & 1
        bits_i_64 = bits_i.astype(np.uint64)
        packed = (bits_i_64.reshape(NUM_WORDS, 64) * weights).sum(axis=1)
        initial_wires[i] = packed

    def verify_network(comparators):
        """
        Fast verification using bit-parallelism.
        """
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

    def greedy_remove(comparators, time_limit=None):
        """Try to remove comparators one by one, restarting from the end."""
        if time_limit is None:
            time_limit = deadline
        current = list(comparators)
        changed = True
        while changed:
            changed = False
            for idx in range(len(current) - 1, -1, -1):
                if time.time() > time_limit:
                    return current
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    changed = True
                    break
        return current

    def simulated_annealing(comparators, time_budget):
        """Simulated annealing: accept worse solutions to escape local minima."""
        current = list(comparators)
        current_size = len(current)
        best = list(current)
        best_size = current_size
        temp = 5.0
        cooling = 0.995
        
        while time.time() < time_budget and current:
            # Choose a random move
            move_type = rng.random()
            
            if move_type < 0.5 and current:
                # Replace a comparator
                idx = rng.randint(0, len(current) - 1)
                new_i = rng.randint(0, N - 1)
                new_j = rng.randint(0, N - 1)
                if new_i > new_j:
                    new_i, new_j = new_j, new_i
                if new_i == new_j:
                    continue
                test = list(current)
                test[idx] = (new_i, new_j)
            elif move_type < 0.75 and current:
                # Remove a comparator
                idx = rng.randint(0, len(current) - 1)
                test = current[:idx] + current[idx + 1:]
            elif move_type < 0.9 and len(current) < 120:
                # Add a comparator
                new_i = rng.randint(0, N - 1)
                new_j = rng.randint(0, N - 1)
                if new_i > new_j:
                    new_i, new_j = new_j, new_i
                if new_i == new_j:
                    continue
                idx = rng.randint(0, len(current) + 1)
                test = current[:idx] + [(new_i, new_j)] + current[idx:]
            else:
                # Swap two comparators
                if len(current) < 2:
                    continue
                i1, i2 = rng.sample(range(len(current)), 2)
                test = list(current)
                test[i1], test[i2] = test[i2], test[i1]
            
            if not verify_network(test):
                temp *= cooling
                continue
            
            test_size = len(test)
            delta = test_size - current_size
            
            if delta <= 0 or rng.random() < np.exp(-delta / max(temp, 0.1)):
                current = test
                current_size = test_size
                if current_size < best_size:
                    best = list(current)
                    best_size = current_size
                    # Greedy removal after improvement
                    if time.time() < time_budget - 1.0:
                        current = greedy_remove(current, time_budget - 1.0)
                        current_size = len(current)
                        if current_size < best_size:
                            best = list(current)
                            best_size = current_size
            
            temp *= cooling
        
        return best

    # Phase 1: Generate diverse starting points
    start_points = [
        odd_even_transposition(),
        batcher_odd_even(),
    ]
    # Add some random dense networks
    for size in [80, 100, 120]:
        start_points.append(random_dense_network(size))
    
    best_network = None
    best_count = float('inf')
    
    # Phase 2: For each starting point, do greedy removal then simulated annealing
    time_per_start = (deadline - start_time - 2.0) / len(start_points)
    
    for sp in start_points:
        if time.time() > deadline - 3.0:
            break
        
        # Quick greedy removal
        current = greedy_remove(sp, min(deadline - 3.0, start_time + 5.0))
        if len(current) < best_count:
            best_network = current
            best_count = len(current)
        
        # Simulated annealing with time budget
        sa_budget = min(deadline - 3.0, time.time() + time_per_start)
        current = simulated_annealing(current, sa_budget)
        if len(current) < best_count:
            best_network = current
            best_count = len(current)
    
    # Final verification
    if best_network is None or not verify_network(best_network):
        best_network = odd_even_transposition()

    return best_network
# EVOLVE-BLOCK-END
