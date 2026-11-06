"""Direct construction of a size-efficient sorting network on 18 wires using Simulated Annealing."""
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
        # Use a copy of the initial wires
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
            # Try removing from random positions
            # Bias towards ends? Random is fine for now.
            # To speed up, we can try all positions if n is small, otherwise random sample.
            if n < 20:
                positions = list(range(n))
                random.shuffle(positions)
            else:
                positions = random.sample(range(n), min(20, n))

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

    def bubble_sort_network():
        comparators = []
        for i in range(N - 1):
            for j in range(N - 1 - i):
                comparators.append((j, j + 1))
        return comparators

    def shell_sort_network():
        """
        Build a network based on shell sort gaps, which can be more compact.
        """
        comparators = []
        # Shell sort gaps for N=18
        gaps = [9, 4, 2, 1]
        for gap in gaps:
            for i in range(gap, N):
                comparators.append((i - gap, i))
        return comparators

    def odd_even_merge_network():
        """
        Construct a network using odd-even merge sort structure.
        """
        comparators = []

        def merge(a_start, a_len, b_start, b_len):
            """
            Merge two sorted sequences using odd-even merge.
            a is at wires a_start..a_start+a_len-1
            b is at wires b_start..b_start+b_len-1
            Result is sorted in wires a_start..a_start+a_len+b_len-1
            """
            total_len = a_len + b_len
            if total_len <= 1:
                return
            if total_len == 2:
                comparators.append((a_start, a_start + 1))
                return
            if total_len == 3:
                # Odd-even merge for 3 elements
                # Assume a_len=1, b_len=2 or a_len=2, b_len=1
                if a_len == 1:
                    # Compare b[0] with a[0], then b[1] with min, then...
                    # Actually for odd-even merge of 1+2:
                    # Step 1: compare a[0] with b[1]
                    comparators.append((a_start, b_start + 1))
                    # Step 2: compare b[0] with b[1] 
                    comparators.append((b_start, b_start + 1))
                    # Step 3: compare a[0] with b[0]
                    comparators.append((a_start, b_start))
                else:
                    # a_len=2, b_len=1
                    # Step 1: compare a[1] with b[0]
                    comparators.append((a_start + 1, b_start))
                    # Step 2: compare a[0] with a[1]
                    comparators.append((a_start, a_start + 1))
                    # Step 3: compare a[1] with b[0]
                    comparators.append((a_start + 1, b_start))
                return

            # For larger sizes, use recursive odd-even merge
            a_half = a_len // 2
            b_half = b_len // 2
            a_remainder = a_len % 2
            b_remainder = b_len % 2

            # Recursive merges
            merge(a_start, a_half, b_start, b_half)
            merge(a_start + a_half, a_len - a_half, b_start + b_half, b_len - b_half)

            # Cross comparisons
            # The standard odd-even merge cross step compares elements at even and odd offsets
            # specifically: for k in 0..min(a_len, b_len)-1, compare a[k] with b[k]
            # But wait, the recursive structure of Odd-Even Merge Sort is:
            # 1. Recursively sort the even-indexed elements of the two sequences (interleaved)
            # 2. Recursively sort the odd-indexed elements of the two sequences (interleaved)
            # 3. Compare adjacent pairs in the resulting sequence.
            
            # My previous recursive implementation was slightly off for standard OEMS.
            # Let's stick to a simpler, verified OEMS or just use the standard one.
            # Standard OEMS:
            # To merge two sorted sequences A and B of lengths n and m:
            # 1. Split A into A_even (elements 0, 2, ...) and A_odd (elements 1, 3, ...)
            # 2. Split B into B_even and B_odd
            # 3. Recursively merge A_even and B_even -> result R_even
            # 4. Recursively merge A_odd and B_odd -> result R_odd
            # 5. Interleave R_even and R_odd -> result R
            # 6. Compare adjacent elements in R (i, i+1) for i=0..len(R)-2
            
            # This is hard to do in-place on a fixed wire layout without shuffling wires.
            # The "interleave" step requires moving data between wires, which isn't a comparator.
            # So standard OEMS is not directly a sorting network on fixed wires without extra shuffling.
            # The "Odd-Even Merge Network" usually refers to a specific structure.
            
            # Let's use a simpler, known-good network for merging:
            # Actually, for the purpose of generating a seed, any valid sorting network is fine.
            # The Bubble Sort and Odd-Even Transposition are already valid.
            # Let's just return a valid network, maybe a bit denser.
            
            # Since implementing a correct fixed-wire OEMS is tricky, let's just return
            # a dense random network or a simple merge sort network.
            # A simple merge sort network can be built recursively.
            
            # Let's revert to a simple, correct recursive merge sort network.
            def merge_sort(a_start, length):
                if length <= 1:
                    return
                half = length // 2
                merge_sort(a_start, half)
                merge_sort(a_start + half, length - half)
                # Merge the two sorted halves
                # This is the tricky part. We need a network to merge two sorted lists.
                # A simple way is to use a "butterfly" or just brute force comparators.
                # For small sizes, we can just use a dense set of comparators.
                # Or we can use a known small merge network.
                # For now, let's just add a dense set of comparators to ensure correctness.
                for i in range(half):
                    for j in range(half):
                        comparators.append((a_start + i, a_start + half + j))
                        # Ensure i < j
                        if a_start + i > a_start + half + j:
                            pass # The loop above might add duplicates or invalid order if not careful
                        # Actually, let's just add all pairs between the two halves
                        # (a_start + i, a_start + half + j)
                        # This is O(n^2) per merge, which is fine for a seed.
            
            # Let's just use the simple dense approach for the merge step in this seed generator.
            # It's not efficient, but it's a valid seed for the optimizer.
            pass # The function body above was getting complicated. Let's simplify.
            
            # Actually, let's just use a simple "bubblesort" style merge for the seed.
            # Or better, just return an empty list and let the other seeds (Odd-Even, Bubble) do the work.
            # The Odd-Even Transposition is already a good seed.
            # Let's just make this function return a simple valid network, e.g., Bubble Sort.
            # But we already have bubble_sort_network().
            # Let's make this function return a slightly different structure, e.g., Shell Sort.
            # But we already have shell_sort_network().
            
            # Let's just return a random dense network for this seed.
            pass
            
        # Since the recursive merge is complex to get right for fixed wires without shuffling,
        # let's just return a simple valid network here.
        # We'll use a simple "selection sort" style network.
        for i in range(N):
            for j in range(i + 1, N):
                comparators.append((i, j))
        return comparators

    # Generate starting candidates
    random.seed(123)
    candidates = []
    candidates.append(("odd_even", odd_even_transposition()))
    candidates.append(("bubble", bubble_sort_network()))
    candidates.append(("shell", shell_sort_network()))
    candidates.append(("dense", odd_even_merge_network())) # Now just a dense selection sort network

    # Add some random dense starts
    for i in range(3):
        random.seed(100 + i)
        candidates.append((f"random_{i}", random_dense_network(150)))

    def random_dense_network(num_comparators=200):
        """Generate a random dense network and shuffle."""
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

    def simulated_annealing(initial_net, t_max=1000, t_min=1.0, alpha=0.995, max_iter=20000, deadline=None):
        """
        Simulated Annealing to minimize comparator count.
        """
        current = list(initial_net)
        current_energy = len(current)
        best = list(current)
        best_energy = current_energy
        
        t = t_max
        random.seed(42) # Fixed seed for reproducibility within restart, but we vary initial net
        
        for i in range(max_iter):
            if deadline and time.time() > deadline:
                break
            
            r = random.random()
            candidate = list(current)
            
            if r < 0.4:
                # Remove
                if len(candidate) == 0:
                    continue
                idx = random.randint(0, len(candidate) - 1)
                candidate.pop(idx)
                delta_e = -1
            elif r < 0.6:
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
            elif r < 0.8:
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
                # Accept with probability exp(-delta_e / t)
                # delta_e is positive (worse)
                prob = np.exp(-new_energy / t) # Note: delta is new - old. If new > old, delta > 0.
                # Actually, standard SA uses exp(-delta / T). Here delta = new - old.
                # If new > old, delta > 0. We want to accept with prob exp(-delta/T).
                # If new < old, we already accepted.
                if random.random() < np.exp(-(new_energy - current_energy) / t):
                    current = candidate
                    current_energy = new_energy
            
            t *= alpha
            if t < t_min:
                break
        
        return best

    best_network = None
    best_count = float('inf')

    # Run SA on each candidate
    for name, net in candidates:
        if time.time() > deadline - 5.0:
            break
        
        # First, try to prune the initial network to get a better starting point for SA
        pruned = greedy_remove(net, time_limit=start_time + 5.0) # Quick prune
        if len(pruned) < best_count:
            best_count = len(pruned)
            best_network = list(pruned)
        
        # Run SA from the pruned state
        sa_result = simulated_annealing(
            pruned, 
            t_max=500, 
            t_min=0.5, 
            alpha=0.998, 
            max_iter=10000, 
            deadline=deadline - 5.0
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
