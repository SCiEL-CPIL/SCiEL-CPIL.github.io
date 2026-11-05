"""Size-efficient sorting network search on 17 wires."""
import time
import random
import numpy as np

N = 17
M = 1 << N  # 131072

# Precompute a fast verification path:
# We represent the network as a list of (i, j) comparators.
# Verification over all 2^17 boolean inputs is the zero-one principle.

def _verify_fast(network):
    """Verify sorting network over all 2^17 boolean inputs using bit-packing."""
    # Pack all 131072 inputs into a 17 x 131072 boolean matrix (transposed for speed)
    # wires[w] is a numpy uint8 array of length M, where wires[w][k] is the value on wire w for input k
    # Build input matrix: for input k (0..M-1), wire w has bit w of k
    inputs = np.arange(M, dtype=np.uint32)
    wires = np.empty((N, M), dtype=np.uint8)
    for w in range(N):
        wires[w] = ((inputs >> w) & 1).astype(np.uint8)
    
    for (i, j) in network:
        wi = wires[i]
        wj = wires[j]
        # min goes to i, max goes to j
        # min(wi, wj) and max(wi, wj)
        wires[i] = np.minimum(wi, wj)
        wires[j] = np.maximum(wi, wj)
    
    # After sorting, wire 0 should be all 0s and wire 16 should be all 1s
    # Actually, for a boolean input, sorted output has k zeros then (N-k) ones
    # The key check: wire 0 must be all 0, wire 16 must be all 1
    # More strongly: for each input, the number of 1s must be preserved and the output must be sorted
    # Simplest sufficient check: wire 0 all 0, wire 16 all 1, and for each adjacent pair wires[w] <= wires[w+1] for all w
    # But the strongest single check for boolean: the output on each wire w must be:
    # 0 if the input had <= w ones, 1 if input had > w ones... no that's not right.
    # Actually sorted boolean: if input has k ones, output has first (N-k) wires = 0, last k wires = 1.
    # So wire w output = 1 iff k > N-1-w, i.e., k >= N-w.
    # Let's just check: for each input k, the output must be sorted non-decreasing.
    # Check: wires[w][input] <= wires[w+1][input] for all w, and sum is preserved.
    # Fastest: check wire 0 all 0, wire 16 all 1, and wires are non-decreasing across wires for each column.
    
    # Check non-decreasing: for each w, wires[w] <= wires[w+1] element-wise
    for w in range(N - 1):
        if not np.all(wires[w] <= wires[w + 1]):
            return False
    # Check wire 0 all 0 and wire N-1 all 1
    if not np.all(wires[0] == 0):
        return False
    if not np.all(wires[N-1] == 1):
        return False
    return True


def _depth_of_network(network):
    """Compute parallel depth of a compare-exchange network."""
    # depth[w] = earliest time wire w is available
    depth = [0] * N
    for (i, j) in network:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth) if depth else 0


def _seed_odd_even():
    """Odd-even transposition network for N wires."""
    comps = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                comps.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                comps.append((i, i + 1))
    return comps


def _seed_batcher():
    """Batcher's odd-even merge sort network for N wires (not power of 2, use recursive construction)."""
    # For non-power-of-2, we can use a simple approach: pad to next power of 2, build, then trim.
    # But that's complex. Instead, use a known small network or just use odd-even as seed.
    # Let's implement Batcher's for power of 2 and adapt.
    # For N=17, pad to 32, build Batcher, keep only comparators involving wires < 17,
    # then fix up. Actually this is tricky. Let's just use odd-even as the base seed.
    return _seed_odd_even()


def construct():
    start = time.time()
    deadline = start + 85  # 5s margin before 90s kill
    
    best_network = _seed_odd_even()
    best_count = len(best_network)
    
    # Try a few known smaller constructions if we have time
    # Known best for 17 wires is 71 comparators. Let's see if we can find something better.
    
    # Strategy: local search / mutation
    # Start from seed, try removing comparators or swapping, verify.
    
    rng = random.Random(42)
    
    def _time_left():
        return deadline - time.time()
    
    def _try_network(net):
        nonlocal best_network, best_count
        if len(net) < best_count:
            if _verify_fast(net):
                best_network = list(net)
                best_count = len(net)
                return True
        return False
    
    # First, verify our seed
    if not _verify_fast(best_network):
        # Shouldn't happen, but just in case
        pass
    
    # Local search: try removing each comparator and check if still valid
    # This is expensive but let's try a few rounds
    
    improved = True
    round_num = 0
    while improved and _time_left() > 5:
        improved = False
        round_num += 1
        for idx in range(len(best_network)):
            if _time_left() < 5:
                break
            # Try removing comparator idx
            trial = best_network[:idx] + best_network[idx+1:]
            if _verify_fast(trial):
                best_network = trial
                improved = True
                break  # restart scan after a removal
        # If no removal worked, try some mutations
        if not improved:
            # Try swapping adjacent comparators
            for idx in range(len(best_network) - 1):
                if _time_left() < 5:
                    break
                trial = list(best_network)
                trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
                if _verify_fast(trial):
                    best_network = trial
                    improved = True
                    break
            # Try replacing a comparator with a different one
            if not improved:
                for idx in range(len(best_network)):
                    if _time_left() < 5:
                        break
                    for _ in range(3):  # try a few random replacements
                        if _time_left() < 5:
                            break
                        i = rng.randint(0, N - 2)
                        j = rng.randint(i + 1, N - 1)
                        trial = list(best_network)
                        trial[idx] = (i, j)
                        if _verify_fast(trial):
                            best_network = trial
                            improved = True
                            break
    
    # Return the best network found
    return best_network
