"""Size-efficient sorting network search for 17 wires via local search."""
import time
import random
import numpy as np

N = 17
N_BOOL = 1 << N  # 131072

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    random.seed(42)

    # --- Seed: odd-even transposition network (136 comparators) ---
    def odd_even_transposition():
        comps = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comps.append((i, i + 1))
        return comps

    best = odd_even_transposition()

    # --- Fast zero-one verification using bitmasks ---
    # Each wire holds a bitmask of which inputs currently have a 1 on it.
    def is_valid(comps):
        # For each of the N boolean inputs, simulate the network.
        # Use vectorized approach: track for each wire a bitmask of active inputs.
        # Actually, let's do it per-input but vectorized over inputs.
        # 
        # For input k (0-indexed), wire w has value 1 iff w >= N-1-k (top k wires are 1).
        # We simulate all 2^N inputs at once using bitmasks.
        #
        # state[w] = bitmask over inputs where wire w has value 1
        # For input k: wire w is 1 iff w >= N-1-k, i.e., k >= N-1-w
        # So state[w] = bitmask where bits k are set for k >= N-1-w
        # That's: state[w] = ((1 << N) - 1) ^ ((1 << (N-1-w)) - 1)
        #        = ((1 << N) - 1) - ((1 << (N-1-w)) - 1)
        
        state = np.empty(N, dtype=np.int64)
        for w in range(N):
            # Bits k where k >= N-1-w are set
            # k ranges 0..N-1, set if k >= N-1-w
            # Number of set bits: N - (N-1-w) = w+1
            # They are the top (w+1) bits: bits N-1, N-2, ..., N-1-w
            # Mask: ((1 << N) - 1) ^ ((1 << (N-1-w)) - 1)
            state[w] = ((1 << N) - 1) ^ ((1 << (N - 1 - w)) - 1)
        
        for (i, j) in comps:
            a = state[i]
            b = state[j]
            # min goes to i: a & b (both 1 -> 1)
            # max goes to j: a | b (either 1 -> 1)
            state[i] = a & b
            state[j] = a | b
        
        # After sorting, wire 0 should have all 1s (bitmask = 2^N - 1)
        # and wire N-1 should have all 0s (bitmask = 0)
        # Actually: sorted means wire 0 = max = 1 for all inputs where any input is 1
        # Wait: after sorting, the sorted output has 1s in the top positions.
        # For input k (k ones), sorted output has ones in wires N-k..N-1.
        # So wire w has value 1 for input k iff w >= N-k, i.e., k >= N-w.
        # state[w] should have bits k set for k >= N-w.
        # For w=0: bits k for k >= N, i.e., none... that's wrong.
        
        # Let me reconsider. Input k means k of the inputs are 1.
        # After sorting, the 1s are in the highest k wires: wires N-k, N-k+1, ..., N-1.
        # Wire w has value 1 iff w >= N-k, i.e., k >= N-w.
        # For w=0: k >= N, impossible (k max is N-1). So wire 0 is always 0? No...
        # For w=N-1: k >= 1. So wire N-1 is 1 for all inputs except k=0.
        
        # Correct check: state[w] should equal bitmask with bits k set for k >= N-w.
        for w in range(N):
            expected = ((1 << N) - 1) ^ ((1 << (N - w)) - 1) if w < N else 0
            if w == 0:
                expected = 0  # k >= N is impossible
            else:
                expected = ((1 << N) - 1) ^ ((1 << (N - w)) - 1)
            if state[w] != expected:
                return False
        return True

    # Verify seed
    assert is_valid(best), "Seed network should be valid"

    best_count = len(best)

    # --- Local search: try removing each comparator ---
    def try_remove(comps, idx):
        if idx >= len(comps):
            return None
        new_comps = comps[:idx] + comps[idx+1:]
        if is_valid(new_comps):
            return new_comps
        return None

    # Greedy removal: repeatedly try to remove any comparator
    improved = True
    while improved:
        improved = False
        comps = list(best)
        for idx in range(len(comps) - 1, -1, -1):
            if time.time() > deadline:
                break
            new_comps = try_remove(comps, idx)
            if new_comps is not None:
                comps = new_comps
                improved = True
                break
        if improved:
            best = comps
            best_count = len(best)

    # --- Try swapping adjacent comparators to enable further removals ---
    for _iter in range(200):
        if time.time() > deadline:
            break
        if best_count <= 71:
            break
        comps = list(best)
        # Try a random swap of two comparators
        if len(comps) < 2:
            break
        i = random.randint(0, len(comps) - 2)
        comps[i], comps[i+1] = comps[i+1], comps[i]
        if is_valid(comps):
            # Now try greedy removal again
            improved2 = True
            while improved2:
                improved2 = False
                for idx in range(len(comps) - 1, -1, -1):
                    if time.time() > deadline:
                        break
                    new_comps = try_remove(comps, idx)
                    if new_comps is not None:
                        comps = new_comps
                        improved2 = True
                        break
            if len(comps) < best_count:
                best = comps
                best_count = len(best)

    # --- Try removing pairs of comparators (2-opt style) ---
    for _iter in range(500):
        if time.time() > deadline:
            break
        if best_count <= 71:
            break
        comps = list(best)
        if len(comps) < 2:
            break
        i = random.randint(0, len(comps) - 2)
        j = random.randint(i + 1, len(comps) - 1)
        new_comps = comps[:i] + comps[i+1:j] + comps[j+1:]
        if is_valid(new_comps):
            if len(new_comps) < best_count:
                best = new_comps
                best_count = len(best)
                # Greedy removal after pair removal
                improved3 = True
                while improved3:
                    improved3 = False
                    for idx in range(len(best) - 1, -1, -1):
                        if time.time() > deadline:
                            break
                        new_comps2 = try_remove(best, idx)
                        if new_comps2 is not None:
                            best = new_comps2
                            best_count = len(best)
                            improved3 = True
                            break

    return best
# EVOLVE-BLOCK-END
