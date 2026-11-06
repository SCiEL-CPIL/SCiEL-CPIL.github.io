"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    def odd_even_transposition():
        comparators = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comparators.append((i, i + 1))
        return comparators

    # Pre-compute bit masks for initialization
    NUM_WORDS = (1 << N) // 64  # 4096
    total_inputs = 1 << N  # 262144
    indices = np.arange(total_inputs, dtype=np.uint32)
    weights = 1 << np.arange(64, dtype=np.uint64)

    # Pre-compute initial wires: wire i has bit k set if input k has bit i set
    initial_wires = [None] * N
    for i in range(N):
        bits_i = (indices >> i) & 1
        bits_i_64 = bits_i.astype(np.uint64)
        packed = (bits_i_64.reshape(NUM_WORDS, 64) * weights).sum(axis=1)
        initial_wires[i] = packed

    def verify_network(comparators):
        """
        Fast verification using bit-parallelism.
        Returns True if the network sorts all 2^N Boolean inputs.
        """
        wires = [w.copy() for w in initial_wires]
        bitwise_and = np.bitwise_and
        bitwise_or = np.bitwise_or
        bitwise_not = np.bitwise_not
        any_ = np.any

        for (i, j) in comparators:
            wi = wires[i]
            wj = wires[j]
            new_i = bitwise_and(wi, wj)
            new_j = bitwise_or(wi, wj)
            wires[i] = new_i
            wires[j] = new_j

        # Check that the output is sorted: wire k should be subset of wire k+1
        for k in range(N - 1):
            if any_(wires[k] & bitwise_not(wires[k + 1])):
                return False
        return True

    def greedy_remove(comparators, time_limit=None):
        """
        Try to remove comparators one by one, scanning from the end.
        Multiple passes until no more removals possible or time limit hit.
        """
        current = list(comparators)
        if time_limit is None:
            time_limit = deadline - 1.0

        while True:
            if time.time() > time_limit:
                break
            removed = False
            for idx in range(len(current) - 1, -1, -1):
                if time.time() > time_limit:
                    return current
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    removed = True
                    break
            if not removed:
                break
        return current

    def local_search(comparators, time_limit=None):
        """
        Systematic comparator elimination with informed insertion.
        First pass: find all removable comparators.
        Second pass: try to replace each non-removable comparator with a 
        different one that preserves sorting, then re-attempt removals.
        """
        if time_limit is None:
            time_limit = deadline - 2.0

        current = list(comparators)
        rng = random.Random(42)
        best = list(current)
        best_count = len(current)

        def find_removable(comps, tl):
            """Find all indices where removal preserves sorting."""
            removable = []
            for idx in range(len(comps) - 1, -1, -1):
                if time.time() > tl:
                    break
                test = comps[:idx] + comps[idx + 1:]
                if verify_network(test):
                    removable.append(idx)
            return removable

        while time.time() < time_limit:
            # Phase 1: Remove all removable comparators
            changed = True
            while changed and time.time() < time_limit:
                changed = False
                removable = find_removable(current, time_limit)
                if not removable:
                    break
                # Remove from end to keep indices valid
                for idx in sorted(removable, reverse=True):
                    current = current[:idx] + current[idx + 1:]
                    changed = True
                    if len(current) < best_count:
                        best = list(current)
                        best_count = len(current)

            if len(current) < best_count:
                best = list(current)
                best_count = len(current)

            # Phase 2: Try to replace non-removable comparators
            # For each comparator, try replacing it with a "better" one
            # that involves closer wires or is more "local"
            improved_this_pass = False
            for idx in range(len(current) - 1, -1, -1):
                if time.time() > time_limit:
                    break
                
                i, j = current[idx]
                # Try replacing with nearby comparators
                candidates = []
                for di in range(-3, 4):
                    for dj in range(-3, 4):
                        ni = i + di
                        nj = j + dj
                        if 0 <= ni < N and 0 <= nj < N and ni != nj:
                            if ni > nj:
                                ni, nj = nj, ni
                            if (ni, nj) != (i, j):
                                candidates.append((ni, nj))
                
                # Also try some random candidates
                for _ in range(2):
                    ni = rng.randint(0, N - 1)
                    nj = rng.randint(0, N - 1)
                    if ni > nj:
                        ni, nj = nj, ni
                    if ni != nj and (ni, nj) != (i, j):
                        candidates.append((ni, nj))

                # Shuffle candidates for diversity
                rng.shuffle(candidates)

                for new_comp in candidates:
                    if time.time() > time_limit:
                        break
                    test = list(current)
                    test[idx] = new_comp
                    if verify_network(test):
                        current = test
                        improved_this_pass = True
                        # Try to remove comparators after improvement
                        current = greedy_remove(current, time_limit)
                        if len(current) < best_count:
                            best = list(current)
                            best_count = len(current)
                        break

            if not improved_this_pass:
                # No more improvements possible with this strategy
                break

        return best

    def hill_climb_remove_and_swap(comparators, time_limit=None):
        """
        Aggressive local search: systematic removal + informed replacement.
        Uses a priority queue of "promising" comparators to replace.
        """
        if time_limit is None:
            time_limit = deadline - 2.0

        current = list(comparators)
        rng = random.Random(123)
        best = list(current)
        best_count = len(current)

        # Precompute all possible comparators
        all_comps = [(i, j) for i in range(N) for j in range(i + 1, N)]

        while time.time() < time_limit:
            if not current:
                break

            # Step 1: Try to remove comparators (from end to start)
            removed_any = False
            for idx in range(len(current) - 1, -1, -1):
                if time.time() > time_limit:
                    break
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    removed_any = True
                    if len(current) < best_count:
                        best = list(current)
                        best_count = len(current)

            if not removed_any:
                # Step 2: Try to replace a comparator with a "better" one
                # Strategy: replace long-range comparators with short-range ones
                # or replace comparators that are "redundant" in position
                
                best_idx = -1
                best_new_comp = None
                best_score = float('inf')
                
                for idx in range(len(current)):
                    if time.time() > time_limit:
                        break
                    
                    i, j = current[idx]
                    # Score: prefer replacing long-range with short-range
                    comp_score = j - i
                    
                    # Try nearby comparators
                    for di in range(-2, 3):
                        for dj in range(-2, 3):
                            ni = i + di
                            nj = j + dj
                            if 0 <= ni < N and 0 <= nj < N and ni != nj:
                                if ni > nj:
                                    ni, nj = nj, ni
                                if (ni, nj) == (i, j):
                                    continue
                                # New score: prefer shorter range
                                new_score = (nj - ni) - comp_score
                                if new_score < best_score:
                                    test = list(current)
                                    test[idx] = (ni, nj)
                                    if verify_network(test):
                                        best_score = new_score
                                        best_idx = idx
                                        best_new_comp = (ni, nj)
                
                if best_idx >= 0:
                    current[best_idx] = best_new_comp
                    if len(current) < best_count:
                        best = list(current)
                        best_count = len(current)
                    # After replacement, try to remove more
                    current = greedy_remove(current, time_limit)
                    if len(current) < best_count:
                        best = list(current)
                        best_count = len(current)
                else:
                    # No improvement found, break
                    break

        return best

    def multi_start_search(time_limit=None):
        """
        Try multiple different starting points and local search strategies.
        """
        if time_limit is None:
            time_limit = deadline - 2.0

        rng = random.Random(999)
        best = None
        best_count = float('inf')

        # Try different random seeds for the local search
        seeds = [42, 123, 456]

        for seed in seeds:
            if time.time() > time_limit:
                break

            # Start from odd-even transposition
            start_network = odd_even_transposition()

            # First do greedy removal
            current = greedy_remove(start_network, time_limit)

            # Then do local search with this seed
            if time.time() > time_limit:
                break

            # Modify local_search to use a specific seed
            def local_search_with_seed(comparators, tl, s):
                current = list(comparators)
                rng = random.Random(s)
                best_local = list(current)
                best_count_local = len(current)

                def find_removable(comps, tlim):
                    removable = []
                    for idx in range(len(comps) - 1, -1, -1):
                        if time.time() > tlim:
                            break
                        test = comps[:idx] + comps[idx + 1:]
                        if verify_network(test):
                            removable.append(idx)
                    return removable

                while time.time() < tl:
                    changed = True
                    while changed and time.time() < tl:
                        changed = False
                        removable = find_removable(current, tl)
                        if not removable:
                            break
                        for idx in sorted(removable, reverse=True):
                            current = current[:idx] + current[idx + 1:]
                            changed = True
                            if len(current) < best_count_local:
                                best_local = list(current)
                                best_count_local = len(current)

                    if len(current) < best_count_local:
                        best_local = list(current)
                        best_count_local = len(current)

                    improved = False
                    for idx in range(len(current) - 1, -1, -1):
                        if time.time() > tl:
                            break
                        i, j = current[idx]
                        candidates = []
                        for di in range(-3, 4):
                            for dj in range(-3, 4):
                                ni = i + di
                                nj = j + dj
                                if 0 <= ni < N and 0 <= nj < N and ni != nj:
                                    if ni > nj:
                                        ni, nj = nj, ni
                                    if (ni, nj) != (i, j):
                                        candidates.append((ni, nj))
                        for _ in range(2):
                            ni = rng.randint(0, N - 1)
                            nj = rng.randint(0, N - 1)
                            if ni > nj:
                                ni, nj = nj, ni
                            if ni != nj and (ni, nj) != (i, j):
                                candidates.append((ni, nj))
                        rng.shuffle(candidates)
                        for new_comp in candidates:
                            if time.time() > tl:
                                break
                            test = list(current)
                            test[idx] = new_comp
                            if verify_network(test):
                                current = test
                                improved = True
                                current = greedy_remove(current, tl)
                                if len(current) < best_count_local:
                                    best_local = list(current)
                                    best_count_local = len(current)
                                break
                    if not improved:
                        break
                return best_local

            result = local_search_with_seed(current, time_limit, seed)
            if len(result) < best_count:
                best = result
                best_count = len(result)

        return best if best is not None else odd_even_transposition()

    # Start with odd-even transposition
    best_network = odd_even_transposition()
    best_count = len(best_network)

    # Phase 1: Aggressive greedy removal
    current = greedy_remove(best_network, deadline - 15.0)
    if len(current) < best_count:
        best_network = current
        best_count = len(best_network)

    # Phase 2: Multi-start local search
    current = multi_start_search(deadline - 5.0)
    if len(current) < best_count:
        best_network = current
        best_count = len(best_network)

    # Phase 3: Final aggressive hill climbing on the best result
    current = hill_climb_remove_and_swap(best_network, deadline - 2.0)
    if len(current) < best_count:
        best_network = current
        best_count = len(best_network)

    # Final verification
    if not verify_network(best_network):
        best_network = odd_even_transposition()

    return best_network
# EVOLVE-BLOCK-END
