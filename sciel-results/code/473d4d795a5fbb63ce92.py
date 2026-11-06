"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16
MAX_COMPARATORS = 4096
SEED_DEADLINE = 45.0  # safety margin before 60s kill

# Pre-compute the input matrix for verification to avoid regeneration overhead
_N_INPUTS = 1 << N
_INDICES = np.arange(_N_INPUTS, dtype=np.uint32)
_INPUT_MATRIX = np.zeros((_N_INPUTS, N), dtype=np.uint8)
for _i in range(N):
    _INPUT_MATRIX[:, _i] = (_INDICES >> _i) & 1

def _apply_network(network, x):
    """Apply the compare-exchange network to input vector x (numpy array)."""
    for i, j in network:
        if x[i] > x[j]:
            x[i], x[j] = x[j], x[i]
    return x

def _verify_network_fastest(network):
    """
    Fast verification using pre-computed input matrix.
    """
    # Copy the input matrix
    x = _INPUT_MATRIX.copy()
    
    # Apply network
    for i, j in network:
        mask = x[:, i] > x[:, j]
        if np.any(mask):
            x[mask, i] = 0
            x[mask, j] = 1
    
    # Verify sorted
    # For boolean sorted: x[:, i] <= x[:, i+1] for all i
    # We can check this by ensuring that for each row, the transition from 0 to 1 happens at most once.
    # Or simply check adjacent pairs.
    for i in range(N - 1):
        if np.any(x[:, i] > x[:, i + 1]):
            return False
    return True

def _count_comparators(network):
    return len(network)

def _compute_depth(network):
    """Compute the parallel depth of the network."""
    if not network:
        return 0
    
    n = len(network)
    wire_to_last_comp = {}  # wire -> last comparator index that touched it
    depths = [0] * n
    
    for idx, (i, j) in enumerate(network):
        max_dep = 0
        for wire in (i, j):
            if wire in wire_to_last_comp:
                prev_idx = wire_to_last_comp[wire]
                if depths[prev_idx] + 1 > max_dep:
                    max_dep = depths[prev_idx] + 1
        depths[idx] = max_dep
        wire_to_last_comp[i] = idx
        wire_to_last_comp[j] = idx
    
    return max(depths) if depths else 0

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network for 16 wires.
    Starts from Batcher's bitonic sort (72 comparators) and performs
    randomized local search (delete, swap, move, insert) to minimize
    comparator count while maintaining correctness.
    """
    def bitonic_sort_network(n):
        """
        Generate Batcher's bitonic sort network for n wires (n must be power of 2).
        """
        def bitonic_merge(n, offset=0, ascending=True):
            """Generate comparators for bitonic merge."""
            if n == 1:
                return []
            half = n // 2
            comparators = []
            # First level: compare i with i+half
            for i in range(half):
                a = i + offset
                b = i + half + offset
                if not ascending:
                    a, b = b, a
                comparators.append((a, b))
            # Recursively merge
            comparators.extend(bitonic_merge(half, offset, ascending))
            comparators.extend(bitonic_merge(half, half + offset, not ascending))
            return comparators
        
        def bitonic_sort(n, offset=0, ascending=True):
            """Generate comparators for bitonic sort."""
            if n == 1:
                return []
            half = n // 2
            comparators = []
            # Sort first half ascending
            comparators.extend(bitonic_sort(half, offset, True))
            # Sort second half descending
            comparators.extend(bitonic_sort(half, half + offset, False))
            # Merge
            comparators.extend(bitonic_merge(n, offset, ascending))
            return comparators
        
        return bitonic_sort(n)
    
    # Generate bitonic sort network
    network = bitonic_sort_network(N)
    
    # Verify
    if not _verify_network_fastest(network):
        # Fallback
        network = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
        return network
    
    best_network = list(network)
    best_len = len(best_network)
    start_time = time.time()
    deadline = start_time + SEED_DEADLINE - 5.0  # safety margin
    
    rng = random.Random(42)
    
    def try_move(net, idx, new_idx):
        """Move comparator at idx to new_idx position."""
        comp = net[idx]
        remaining = [c for k, c in enumerate(net) if k != idx]
        if new_idx > idx:
            new_idx -= 1
        new_idx = max(0, min(new_idx, len(remaining)))
        candidate = remaining[:new_idx] + [comp] + remaining[new_idx:]
        return candidate
    
    def try_insert(net, new_idx):
        """Insert a random comparator at new_idx position."""
        a, b = rng.sample(range(N), 2)
        if a > b:
            a, b = b, a
        comp = (a, b)
        candidate = net[:new_idx] + [comp] + net[new_idx:]
        return candidate
    
    # Local search
    while time.time() < deadline:
        net = list(best_network)
        n_comp = len(net)
        
        # Choose mutation type
        r = rng.random()
        if r < 0.35 and n_comp > 1:
            # Delete a comparator
            idx = rng.randint(0, n_comp - 1)
            candidate = net[:idx] + net[idx+1:]
        elif r < 0.55 and n_comp >= 2:
            # Swap two comparators
            i, j = rng.sample(range(n_comp), 2)
            candidate = list(net)
            candidate[i], candidate[j] = candidate[j], candidate[i]
        elif r < 0.75 and n_comp >= 1:
            # Move a comparator to a new position
            idx = rng.randint(0, n_comp - 1)
            new_idx = rng.randint(0, n_comp - 1)
            candidate = try_move(net, idx, new_idx)
        else:
            # Insert a random comparator
            new_idx = rng.randint(0, n_comp)
            candidate = try_insert(net, new_idx)
        
        # Verify candidate
        if _verify_network_fastest(candidate):
            if len(candidate) < best_len:
                best_network = candidate
                best_len = len(candidate)
                # Greedy removal pass: try to remove each comparator
                for i in range(len(best_network) - 1, -1, -1):
                    if time.time() > deadline:
                        break
                    trial = best_network[:i] + best_network[i+1:]
                    if _verify_network_fastest(trial):
                        best_network = trial
                        best_len = len(best_network)
    
    # Final greedy removal pass
    for i in range(len(best_network) - 1, -1, -1):
        if time.time() > deadline:
            break
        trial = best_network[:i] + best_network[i+1:]
        if _verify_network_fastest(trial):
            best_network = trial
    
    return best_network
# EVOLVE-BLOCK-END
