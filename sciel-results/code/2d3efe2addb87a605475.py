import time
import random
import numpy as np

N = 18

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin (90s limit, leave 5s buffer)

    # Pre-compute bit masks for verification
    total_inputs = 1 << N  # 262144
    NUM_WORDS = total_inputs // 64  # 4096

    indices = np.arange(total_inputs, dtype=np.uint32)
    weights = 1 << np.arange(64, dtype=np.uint64)

    initial_wires = [None] * N
    for i in range(N):
        bits_i = (indices >> i) & 1
        bits_i_64 = bits_i.astype(np.uint64)
        packed = (bits_i_64.reshape(NUM_WORDS, 64) * weights).sum(axis=1)
        initial_wires[i] = packed

    def verify_network(comparators):
        """
        Fast verification using bit-parallelism.
        Returns True if the network sorts all Boolean inputs.
        """
        wires = [w.copy() for w in initial_wires]

        for (i, j) in comparators:
            wi = wires[i]
            wj = wires[j]
            wires[i] = wi & wj
            wires[j] = wi | wj

        # Check sorted: for all k, wires[k] is subset of wires[k+1]
        # i.e., wires[k] & ~wires[k+1] should be all zeros
        for k in range(N - 1):
            if np.any(wires[k] & ~wires[k + 1]):
                return False
        return True

    def greedy_remove(comparators, time_limit=None):
        """
        Iteratively remove redundant comparators.
        """
        current = list(comparators)
        changed = True
        while changed:
            changed = False
            if time_limit and time.time() > time_limit:
                break
            n = len(current)
            if n == 0:
                break
            
            # Try all positions for small networks, sample for large
            if n <= 100:
                positions = list(range(n))
                random.shuffle(positions)
            else:
                positions = random.sample(range(n), min(100, n))

            for idx in positions:
                if idx >= len(current):
                    continue
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    changed = True
                    break

        return current

    def odd_even_transposition():
        comparators = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comparators.append((i, i + 1))
        return comparators

    def bitonic_sort_network():
        """
        Generate a bitonic sort network.
        """
        def bitonic_merge(n, offset=0, direction=1):
            # n is the size of the bitonic sequence to merge
            # direction: 1 for ascending, -1 for descending
            if n == 1:
                return []
            
            comparators = []
            # First, recursively merge the two halves
            # The two halves are of size n/2
            # Note: Bitonic sort usually works on powers of 2.
            # For N=18, we can't do a perfect power-of-2 bitonic sort directly on the whole array.
            # However, we can build a bitonic network for the largest power of 2 <= N, 
            # and then handle the rest, or just use a general bitonic merge logic.
            
            # Actually, standard bitonic sort requires N to be a power of 2.
            # For N=18, we can use a "bitonic-like" network or just use it as a seed 
            # by creating a bitonic network for 16 wires and then adding 2 more wires 
            # in a way that maintains sorting properties? That's complex.
            
            # Alternative: Use a known good network structure.
            # Let's stick to generating a valid sorting network that is dense but structured.
            # A simpler structured network is the Odd-Even Merge Sort network.
            pass
        
        # Since standard bitonic sort is for powers of 2, let's use a different structured seed:
        # The "0-1 Principle" based networks are best, but we need a concrete seed.
        # Let's use a dense random network but with a bias towards adjacent comparators.
        comparators = []
        for _ in range(120):
            i = random.randint(0, N - 2)
            j = i + 1
            # Sometimes jump
            if random.random() < 0.2:
                j = random.randint(i + 1, N - 1)
            comparators.append((i, j))
        random.shuffle(comparators)
        return comparators

    def random_dense_network(num_comparators=150, seed=None):
        """Generate a random dense network and shuffle."""
        if seed is not None:
            random.seed(seed)
        comparators = []
        for _ in range(num_comparators):
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i != j:
                if i > j:
                    i, j = j, i
                comparators.append((i, j))
        random.shuffle(comparators)
        return comparators

    def simulated_annealing(initial_net, t_max=1000, t_min=0.5, alpha=0.997, max_iter=20000, deadline=None, seed=42):
        """
        Simulated Annealing to minimize comparator count.
        """
        current = list(initial_net)
        current_energy = len(current)
        best = list(current)
        best_energy = current_energy
        
        t = t_max
        random.seed(seed)
        
        for i in range(max_iter):
            if deadline and time.time() > deadline:
                break
            
            r = random.random()
            candidate = list(current)
            
            if r < 0.6:
                # Remove (higher probability)
                if len(candidate) == 0:
                    continue
                idx = random.randint(0, len(candidate) - 1)
                candidate.pop(idx)
            elif r < 0.75:
                # Insert
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                pos = random.randint(0, len(candidate))
                candidate.insert(pos, (u, v))
            elif r < 0.9:
                # Swap
                if len(candidate) < 2:
                    continue
                i1, i2 = random.sample(range(len(candidate)), 2)
                candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
            else:
                # Replace
                if len(candidate) == 0:
                    continue
                idx = random.randint(0, len(candidate) - 1)
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                candidate[idx] = (u, v)

            if not verify_network(candidate):
                continue # Invalid, reject

            new_energy = len(candidate)
            
            if new_energy < current_energy:
                # Always accept improvement
                current = candidate
                current_energy = new_energy
                if new_energy < best_energy:
                    best = list(current)
                    best_energy = new_energy
            else:
                # Accept with probability exp(-delta / T)
                delta = new_energy - current_energy
                if delta > 0:
                    if random.random() < np.exp(-delta / t):
                        current = candidate
                        current_energy = new_energy
            
            t *= alpha
            if t < t_min:
                break
        
        return best

    # Generate starting candidates
    candidates = []
    candidates.append(("odd_even", odd_even_transposition()))
    
    # Add several random dense starts with different densities and seeds
    for i in range(8):
        density = 120 + i * 10
        seed_val = 100 + i * 13
        candidates.append((f"random_{i}", random_dense_network(density, seed=seed_val)))

    best_network = None
    best_count = float('inf')

    # Run SA on each candidate
    for idx, (name, net) in enumerate(candidates):
        if time.time() > deadline - 10.0:
            break
        
        # First, try to prune the initial network to get a better starting point for SA
        prune_time = min(start_time + 2.0, deadline - 15.0)
        pruned = greedy_remove(net, time_limit=prune_time)
        
        if len(pruned) < best_count:
            best_count = len(pruned)
            best_network = list(pruned)
        
        # Run SA from the pruned state
        remaining_time = deadline - 10.0 - time.time()
        if remaining_time < 5.0:
            break
            
        # Allocate time to SA based on remaining time
        sa_deadline = deadline - 5.0
        sa_result = simulated_annealing(
            pruned, 
            t_max=500, 
            t_min=0.5, 
            alpha=0.997, 
            max_iter=20000, 
            deadline=sa_deadline,
            seed=42 + idx * 100
        )
        
        if len(sa_result) < best_count:
            best_count = len(sa_result)
            best_network = list(sa_result)
            
        # Final greedy remove on the SA result
        final_pruned = greedy_remove(best_network, time_limit=deadline - 2.0)
        if len(final_pruned) < best_count:
            best_count = len(final_pruned)
            best_network = list(final_pruned)

    if best_network is None:
        # Fallback
        best_network = odd_even_transposition()
        best_network = greedy_remove(best_network, time_limit=deadline - 1.0)

    # Final verification
    if not verify_network(best_network):
        # Should not happen, but safety
        fallback = odd_even_transposition()
        fallback = greedy_remove(fallback, time_limit=deadline - 1.0)
        return fallback

    return best_network
