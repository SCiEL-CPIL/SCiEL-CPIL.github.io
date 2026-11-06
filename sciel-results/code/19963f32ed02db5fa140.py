"""Direct construction of a size-efficient sorting network on 18 wires using Simulated Annealing with multi-seed initialization."""
N = 18
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

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
        for k in range(N - 1):
            if np.any(wires[k] & ~wires[k + 1]):
                return False
        return True

    def greedy_remove(comparators, time_limit=None, full_scan_passes=2):
        """
        Iteratively remove redundant comparators.
        Uses random sampling then full-scan passes for thorough pruning.
        """
        current = list(comparators)
        changed = True
        pass_count = 0
        while changed:
            changed = False
            if time_limit and time.time() > time_limit:
                break
            n = len(current)
            if n == 0:
                break
            pass_count += 1

            # Alternate between random sampling and full scan
            if pass_count <= full_scan_passes and n <= 120:
                # Full scan: try removing each comparator
                for idx in range(n):
                    if time_limit and time.time() > time_limit:
                        break
                    test = current[:idx] + current[idx + 1:]
                    if verify_network(test):
                        current = test
                        changed = True
                        break
            else:
                # Random sampling with bias toward ends
                sample_size = min(25, n)
                positions = list(range(n))
                random.shuffle(positions)
                # Bias: ensure ends are included
                if n > 1:
                    if 0 not in positions[:sample_size]:
                        positions[0] = 0
                    if n - 1 not in positions[:sample_size]:
                        positions[1] = n - 1

                for idx in positions[:sample_size]:
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

    def bubble_sort_network():
        comparators = []
        for i in range(N - 1):
            for j in range(N - 1 - i):
                comparators.append((j, j + 1))
        return comparators

    def bitonic_sort_network():
        """
        Batcher's bitonic sort network for N=18.
        Uses bitonic merge recursively.
        """
        comparators = []

        def bitonic_merge(lo, size, direction):
            """
            Merge a bitonic sequence of 'size' elements starting at 'lo'.
            direction: 1 for ascending, 0 for descending.
            """
            if size <= 1:
                return
            half = size // 2
            # Compare elements distance 'half' apart
            for i in range(lo, lo + half):
                j = i + half
                if direction == 1:
                    comparators.append((i, j))
                else:
                    comparators.append((j, i))
            # Recursively merge halves
            bitonic_merge(lo, half, direction)
            bitonic_merge(lo + half, half, direction)

        def bitonic_sort(lo, size, direction):
            """
            Sort a sequence of 'size' elements starting at 'lo'.
            direction: 1 for ascending, 0 for descending.
            """
            if size <= 1:
                return
            half = size // 2
            # Sort halves in opposite directions
            bitonic_sort(lo, half, 1)
            bitonic_sort(lo + half, half, 0)
            # Merge
            bitonic_merge(lo, size, direction)

        # For N=18, we need to handle the non-power-of-2 case.
        # Pad to 32 (next power of 2) conceptually, but we only have 18 wires.
        # We'll use a recursive approach that handles arbitrary sizes.
        # Actually, let's use a simpler approach: build bitonic sort for 16, then handle remaining 2.
        
        # Build bitonic sort for first 16 wires
        bitonic_sort(0, 16, 1)
        
        # Now merge the sorted 16-element block with the remaining 2 elements
        # The remaining 2 elements (wires 16, 17) need to be inserted
        # First, sort the pair
        comparators.append((16, 17))
        
        # Now we need to merge sorted[0:16] with sorted[16:18]
        # This is a merge of two sorted sequences of length 16 and 2
        # Use odd-even merge or a simple insertion-like approach
        
        # Simple approach: insert elements from the small sorted block into the large one
        # For each element in the small block, compare with appropriate positions
        # This isn't optimal but should work
        
        # Let's use a different approach: build a full bitonic merge
        # Merge [0:16] (ascending) with [16:18] (ascending)
        # First, make [16:18] descending to form a bitonic sequence with [0:16]
        # Actually, let's just use a direct merge network
        
        # Merge two sorted sequences of length 16 and 2
        # Use the fact that we can compare and swap to insert
        # For wire 16: compare with wires that might be larger
        # For wire 17: compare with wires that might be larger
        
        # A simpler valid approach: use odd-even transposition on the last few wires
        # to ensure the whole thing is sorted
        # Actually, let's just append a few comparators to ensure correctness
        
        # The bitonic sort on 16 wires gives us sorted[0:16]
        # We have sorted[16:18]
        # We need to merge these two sorted sequences
        # Use a simple merge: compare cross pairs
        
        # Merge sorted A[0:16] and B[0:2]
        # Result in A[0:18]
        # This is a merge of two sorted lists
        # We can use a network for merging
        
        # For merging two sorted sequences of length m and n, we need at most m*n comparators
        # but we can do better. Let's use a simple approach.
        
        # Actually, let's just use a known-good small merge network
        # Merge [0:16] with [16:18]:
        # Compare (15,16), (14,17), then fix up
        
        # Let's use a more systematic approach
        # The standard way to merge two sorted sequences of lengths a and b
        # is to use a merge network. For 16 and 2, it's small.
        
        # Let me just add comparators to ensure the full 18-wire network sorts
        # by using a final odd-even transpass on the whole thing (expensive but correct)
        # Actually, that would add too many comparators.
        
        # Better: let's build the bitonic sort properly for 18 wires
        # by padding conceptually and only using the first 18 wires
        
        # Restart with a cleaner implementation
        comparators = []
        
        def bitonic_merge_v2(lo, size, direction):
            if size <= 1:
                return
            half = size // 2
            for i in range(lo, lo + half):
                j = i + half
                if i < N and j < N:
                    if direction == 1:
                        comparators.append((i, j))
                    else:
                        comparators.append((j, i))
            bitonic_merge_v2(lo, half, direction)
            bitonic_merge_v2(lo + half, size - half, direction)

        def bitonic_sort_v2(lo, size, direction):
            if size <= 1:
                return
            half = size // 2
            bitonic_sort_v2(lo, half, 1)
            bitonic_sort_v2(lo + half, size - half, 0)
            bitonic_merge_v2(lo, size, direction)

        bitonic_sort_v2(0, 18, 1)
        
        return comparators

    def odd_even_merge_network():
        """
        Construct a network using odd-even merge sort structure.
        """
        comparators = []

        def merge(a_start, a_len, b_start, b_len):
            total_len = a_len + b_len
            if total_len <= 1:
                return
            if total_len == 2:
                comparators.append((a_start, a_start + 1))
                return
            if total_len == 3:
                if a_len == 1:
                    comparators.append((a_start, b_start + 1))
                    comparators.append((b_start, b_start + 1))
                    comparators.append((a_start, b_start))
                else:
                    comparators.append((a_start + 1, b_start))
                    comparators.append((a_start, a_start + 1))
                    comparators.append((a_start + 1, b_start))
                return

            a_half = a_len // 2
            b_half = b_len // 2

            merge(a_start, a_half, b_start, b_half)
            merge(a_start + a_half, a_len - a_half, b_start + b_half, b_len - b_half)

            # Cross comparisons
            cross_count = min(a_len, b_len) // 2 + (1 if (a_len + b_len) % 2 == 0 else 0)
            for i in range(cross_count):
                a_idx = a_start + i
                b_idx = a_start + a_len + i
                if a_idx < N and b_idx < N and a_idx != b_idx:
                    if a_idx < b_idx:
                        comparators.append((a_idx, b_idx))
                    else:
                        comparators.append((b_idx, a_idx))

        # Sort using odd-even merge sort
        for i in range(0, N - 1, 2):
            comparators.append((i, i + 1))

        for i in range(0, N, 4):
            if i + 4 <= N:
                merge(i, 2, i + 2, 2)
            elif i + 2 < N:
                merge(i, 2, i + 2, N - i - 2)

        for i in range(0, N, 8):
            if i + 8 <= N:
                merge(i, 4, i + 4, 4)
            elif i + 4 < N:
                merge(i, 4, i + 4, N - i - 4)

        if N >= 16:
            merge(0, 8, 8, 8)
        if 16 < N <= 18:
            merge(8, 8, 16, N - 16)

        return comparators

    def random_dense_network(num_comparators=150, seed=None):
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

    def simulated_annealing(initial_net, t_max=800, t_min=0.1, alpha=0.997, max_iter=15000, deadline=None):
        """
        Simulated Annealing to minimize comparator count.
        """
        current = list(initial_net)
        current_energy = len(current)
        best = list(current)
        best_energy = current_energy

        t = t_max

        for i in range(max_iter):
            if deadline and time.time() > deadline:
                break

            r = random.random()
            candidate = list(current)

            if r < 0.35:
                # Remove
                if len(candidate) == 0:
                    continue
                idx = random.randint(0, len(candidate) - 1)
                candidate.pop(idx)
                delta_e = -1
            elif r < 0.55:
                # Insert
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                pos = random.randint(0, len(candidate))
                candidate.insert(pos, (u, v))
                delta_e = 1
            elif r < 0.75:
                # Swap
                if len(candidate) < 2:
                    continue
                i1, i2 = random.sample(range(len(candidate)), 2)
                candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
                delta_e = 0
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
                delta_e = 0

            if not verify_network(candidate):
                continue

            new_energy = len(candidate)

            if new_energy < current_energy:
                current = candidate
                current_energy = new_energy
                if new_energy < best_energy:
                    best = list(current)
                    best_energy = new_energy
            else:
                if random.random() < np.exp(-(new_energy - current_energy) / t):
                    current = candidate
                    current_energy = new_energy

            t *= alpha
            if t < t_min:
                break

        return best

    # Generate starting candidates
    random.seed(42)
    candidates = []
    candidates.append(("odd_even", odd_even_transposition()))
    candidates.append(("bubble", bubble_sort_network()))
    candidates.append(("bitonic", bitonic_sort_network()))
    candidates.append(("oo_merge", odd_even_merge_network()))

    # Add random dense starts
    for i in range(3):
        candidates.append((f"random_{i}", random_dense_network(120, seed=200 + i)))

    best_network = None
    best_count = float('inf')

    # Phase 1: Greedy prune each candidate
    for name, net in candidates:
        if time.time() > deadline - 10.0:
            break

        pruned = greedy_remove(net, time_limit=start_time + 3.0, full_scan_passes=1)
        if len(pruned) < best_count:
            best_count = len(pruned)
            best_network = list(pruned)

    # Phase 2: SA from the best pruned candidate
    if best_network is not None:
        sa_result = simulated_annealing(
            best_network,
            t_max=600,
            t_min=0.05,
            alpha=0.998,
            max_iter=12000,
            deadline=deadline - 8.0
        )
        if len(sa_result) < best_count:
            best_count = len(sa_result)
            best_network = list(sa_result)

        # Final greedy polish
        final_pruned = greedy_remove(best_network, time_limit=deadline - 3.0, full_scan_passes=3)
        if len(final_pruned) < best_count:
            best_count = len(final_pruned)
            best_network = list(final_pruned)

    # Phase 3: If time remains, try SA from a second-best seed
    if time.time() < deadline - 12.0:
        # Find the second best pruned candidate
        second_best = None
        second_count = float('inf')
        for name, net in candidates:
            if time.time() > deadline - 12.0:
                break
            if net is best_network:
                continue
            pruned = greedy_remove(net, time_limit=start_time + 2.0, full_scan_passes=1)
            if len(pruned) < second_count and len(pruned) >= best_count:
                second_count = len(pruned)
                second_best = list(pruned)

        if second_best is not None:
            sa_result2 = simulated_annealing(
                second_best,
                t_max=400,
                t_min=0.05,
                alpha=0.998,
                max_iter=8000,
                deadline=deadline - 5.0
            )
            if len(sa_result2) < best_count:
                best_count = len(sa_result2)
                best_network = list(sa_result2)

                final_pruned = greedy_remove(best_network, time_limit=deadline - 2.0, full_scan_passes=3)
                if len(final_pruned) < best_count:
                    best_count = len(final_pruned)
                    best_network = list(final_pruned)

    if best_network is None:
        best_network = odd_even_transposition()
        best_network = greedy_remove(best_network, time_limit=deadline - 1.0)

    # Final verification
    if not verify_network(best_network):
        fallback = odd_even_transposition()
        fallback = greedy_remove(fallback, time_limit=deadline - 1.0)
        return fallback

    return best_network
