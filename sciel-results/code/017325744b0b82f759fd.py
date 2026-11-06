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

    # Pre-compute bit masks for initialization
    NUM_WORDS = (1 << N) // 64  # 4096 words of 64 bits each

    # Pre-compute initial wires using efficient bit packing
    total_inputs = 1 << N
    indices = np.arange(total_inputs, dtype=np.uint32)
    weights = 1 << np.arange(64, dtype=np.uint64)

    initial_wires = [None] * N
    for i in range(N):
        bits_i = (indices >> i) & 1
        bits_i_64 = bits_i.astype(np.uint64)
        packed = (bits_i_64.reshape(NUM_WORDS, 64) * weights).sum(axis=1)
        initial_wires[i] = packed

    # Pre-allocate buffer for fast verification
    wire_buffer = np.empty((N, NUM_WORDS), dtype=np.uint64)

    def verify_network(comparators):
        """
        Fast verification using bit-parallelism.
        Returns True if the network sorts all Boolean inputs.
        """
        # Copy initial wires into pre-allocated buffer
        for i in range(N):
            np.copyto(wire_buffer[i], initial_wires[i])

        for (i, j) in comparators:
            wi = wire_buffer[i]
            wj = wire_buffer[j]
            # min to i, max to j
            new_wi = wi & wj
            new_wj = wi | wj
            wire_buffer[i] = new_wi
            wire_buffer[j] = new_wj

        # Check sorted: for all k, wires[k] is subset of wires[k+1]
        # i.e., wires[k] & ~wires[k+1] == 0
        for k in range(N - 1):
            if np.any(wire_buffer[k] & ~wire_buffer[k + 1]):
                return False
        return True

    def greedy_remove(comparators):
        """
        Iteratively remove redundant comparators.
        Scans from the end first (often more effective for transposition-like nets).
        """
        current = list(comparators)
        changed = True
        while changed:
            changed = False
            if time.time() > deadline - 2.0:
                break
            n = len(current)
            if n == 0:
                break
            # Try removing from the end, then random, then start
            # End is often less critical in sorting networks
            try_indices = list(range(n - 1, -1, -1))
            random.shuffle(try_indices)
            # Ensure we try the last few first
            for idx in try_indices:
                if idx >= len(current):
                    continue
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    changed = True
                    break

        return current

    # Start with odd-even transposition
    current = odd_even_transposition()
    current = greedy_remove(current)

    best_network = list(current)
    best_count = len(current)

    random.seed(42)

    # Local Search with diverse moves
    while time.time() < deadline - 3.0:
        if not current:
            break

        candidate = list(current)
        n_comp = len(candidate)

        if n_comp == 0:
            break

        r = random.random()

        if r < 0.30:
            # Move: Pick a comparator and move it to a new position
            if n_comp > 1:
                i = random.randint(0, n_comp - 1)
                j = random.randint(0, n_comp - 1)
                comp = candidate.pop(i)
                if i < j:
                    j -= 1
                j = max(0, min(j, len(candidate)))
                candidate.insert(j, comp)
        elif r < 0.60:
            # Swap: Swap two comparators
            if n_comp > 1:
                i, j = random.sample(range(n_comp), 2)
                candidate[i], candidate[j] = candidate[j], candidate[i]
        elif r < 0.80:
            # Replace: Replace a comparator with a random one
            i = random.randint(0, n_comp - 1)
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            candidate[i] = (u, v)
        elif r < 0.90:
            # Targeted replace: Replace with a comparator near an existing one
            i = random.randint(0, n_comp - 1)
            old_u, old_v = candidate[i]
            # Try wires near the old comparator
            candidates = []
            for du in [-1, 0, 1]:
                for dv in [-1, 0, 1]:
                    nu = old_u + du
                    nv = old_v + dv
                    if 0 <= nu < N and 0 <= nv < N and nu != nv:
                        if nu > nv:
                            nu, nv = nv, nu
                        if (nu, nv) != (old_u, old_v):
                            candidates.append((nu, nv))
            if candidates:
                candidate[i] = random.choice(candidates)
            else:
                continue
        else:
            # Insert: Add a random comparator (to explore larger nets that might reduce via removal)
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            pos = random.randint(0, n_comp)
            candidate.insert(pos, (u, v))

        # Verify candidate
        if verify_network(candidate):
            # Accept and try to reduce
            candidate = greedy_remove(candidate)

            if len(candidate) < best_count:
                best_count = len(candidate)
                best_network = list(candidate)
                current = candidate
            else:
                # Accept if size is <= current size to allow exploration
                if len(candidate) <= len(current):
                    current = candidate

    # Final verification
    if not verify_network(best_network):
        best_network = odd_even_transposition()

    return best_network
# EVOLVE-BLOCK-END

BET: {mechanism: "Extending the deadline from 55s to 85s allows significantly more local search iterations, increasing the probability of finding better comparator reductions through random restarts and targeted mutations", q: 0.6, delta: [0.05, 0.2]}
