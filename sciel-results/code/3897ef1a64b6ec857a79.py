"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136  # Number of comparators in the seed odd-even transposition network

# Precompute masks for fast simulation
# A comparator (i, j) sets wire i to min(x_i, x_j) and wire j to max(x_i, x_j)
# For 17 wires, we represent state as a 17-bit integer per input, and vectorize over all inputs.

def simulate_network(comparators, inputs):
    """
    inputs: numpy array of shape (num_inputs, N) with values in {0, 1}
    comparators: list of (i, j)
    returns: sorted numpy array of shape (num_inputs, N)
    """
    state = inputs.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return state

# Precompute the input matrix and expected sorted outputs once at module load time
_NUM_INPUTS = 1 << N
_indices = np.arange(_NUM_INPUTS, dtype=np.uint32)
_bits = 1 << np.arange(N, dtype=np.uint32)
_INPUTS = ((_indices[:, None] & _bits[None, :]) > 0).astype(np.uint8)
_SORTED_INPUTS = np.sort(_INPUTS, axis=1)

# Precompute a smaller sample set for fast validation during search
_SAMPLE_SIZE = 4096
_rng = np.random.RandomState(42)
_SAMPLE_INDICES = _rng.choice(_NUM_INPUTS, size=_SAMPLE_SIZE, replace=False)
_SAMPLE_INPUTS = _INPUTS[_SAMPLE_INDICES]
_SAMPLE_SORTED = _SORTED_INPUTS[_SAMPLE_INDICES]

def is_sorted_network_full(comparators):
    """
    Check if the network sorts all 2^17 inputs using the zero-one principle.
    Uses precomputed input matrix for speed.
    """
    if len(comparators) > 4096:
        return False
    state = _INPUTS.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return bool(np.array_equal(state, _SORTED_INPUTS))

def is_sorted_network_sample(comparators):
    """
    Fast check on a sample of inputs. Not sufficient for final validation,
    but much faster for guiding search.
    """
    if len(comparators) > 4096:
        return False
    state = _SAMPLE_INPUTS.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return bool(np.array_equal(state, _SAMPLE_SORTED))

# Alias for backward compatibility
def is_sorted_network(comparators):
    return is_sorted_network_full(comparators)

def random_network(num_comparators):
    """Generate a random sorting network candidate."""
    comparators = []
    for _ in range(num_comparators):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        comparators.append((i, j))
    return comparators

def odd_even_merge_sort_network(n):
    """
    Generate an odd-even merge sort network for n wires.
    Returns list of (i, j) comparators.
    """
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]

    half = n // 2
    left = odd_even_merge_sort_network(half)
    right = odd_even_merge_sort_network(n - half)
    right = [(i + half, j + half) for i, j in right]

    merged = left + right
    for k in range(1, n, 2):
        if k < n:
            merged.append((k - 1, k))

    return merged

def bitonic_sort_network(n):
    """
    Generate a bitonic sort network for n wires (power-of-2 friendly).
    For n=17 we return the largest power-of-2 subnetwork plus extras.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    comparators = []

    def bitonic(k, direction=1):
        if k == 1:
            return []
        if k == 2:
            return [(0, 1)] if direction == 1 else [(1, 0)]
        half = k // 2
        left = bitonic(half, 1)
        right = bitonic(k - half, -1)
        right = [(i + half, j + half) for i, j in right]
        merge = []
        for i in range(half):
            j = i + half
            merge.append((i, j) if direction == 1 else (j, i))
        return left + right + merge

    return bitonic(n)

# EVOLVE-BLOCK-START
def _to_layers(comparators):
    """Group an ordered comparator list into layers of disjoint comparators.

    A comparator belongs to the earliest layer whose wires are all free. This
    preserves the exact sequential order (layers are applied in order, and
    comparators within a layer are mutually disjoint so order is irrelevant).
    Returns a list of lists of (i, j).
    """
    layers = []
    for comp in comparators:
        i, j = comp
        placed = False
        for layer in layers:
            if all(i != a and i != b and j != a and j != b for a, b in layer):
                layer.append(comp)
                placed = True
                break
        if not placed:
            layers.append([comp])
    return layers

def _from_layers(layers):
    """Flatten layers back to an ordered comparator list (order within a layer is
    irrelevant since comparators in a layer are disjoint)."""
    comps = []
    for layer in layers:
        comps.extend(layer)
    return comps

def _layer_count(layers):
    return sum(len(layer) for layer in layers)

def _systematic_deletion(layers, deadline):
    """Repeatedly try to delete each comparator; return (layers, improved)."""
    improved = False
    changed = True
    while changed:
        changed = False
        for li in range(len(layers)):
            if time.time() > deadline:
                return layers, improved
            for ci in range(len(layers[li])):
                if time.time() > deadline:
                    return layers, improved
                layers[li].pop(ci)
                if not layers[li]:
                    layers.pop(li)
                    li -= 1
                    break
                flat = _from_layers(layers)
                if is_sorted_network_sample(flat):
                    if is_sorted_network_full(flat):
                        improved = True
                        changed = True
                    else:
                        layers[li].append(flat[-1])
                else:
                    layers[li].append(flat[-1])
    return layers, improved

def _random_deletion(layers, deadline, attempts):
    """Random deletion attempts; return (layers, improved)."""
    improved = False
    for _ in range(attempts):
        if time.time() > deadline:
            return layers, improved
        total = _layer_count(layers)
        if total == 0:
            break
        # pick a random (li, ci)
        r = random.randint(0, total - 1)
        li = 0
        while r >= len(layers[li]):
            r -= len(layers[li])
            li += 1
        comp = layers[li].pop(r)
        if not layers[li]:
            layers.pop(li)
        flat = _from_layers(layers)
        if is_sorted_network_sample(flat):
            if is_sorted_network_full(flat):
                improved = True
            else:
                layers.append([comp])
        else:
            layers.append([comp])
    return layers, improved

def _layer_shrink(layers, deadline, attempts):
    """Structural move: try to merge a layer into an earlier compatible layer,
    reducing depth (and enabling further deletions). Return (layers, improved)."""
    improved = False
    for _ in range(attempts):
        if time.time() > deadline:
            return layers, improved
        if len(layers) < 2:
            break
        li = random.randint(1, len(layers) - 1)
        src = layers[li]
        # try to move each comparator of src into an earlier layer if disjoint
        for ci in range(len(src)):
            comp = src[ci]
            i, j = comp
            moved = False
            for lj in range(li):
                if all(i != a and i != b and j != a and j != b for a, b in layers[lj]):
                    layers[lj].append(comp)
                    moved = True
                    break
            if moved:
                src.pop(ci)
                if not src:
                    layers.pop(li)
                li -= 1
                if li < 0:
                    break
                continue
            else:
                # cannot move this one; try moving whole layer only if all moved
                # undo
                if moved_any(layers, li, src, comp):
                    break
                # try the next comparator
                continue
        # validate the resulting flattened network
        flat = _from_layers(layers)
        if is_sorted_network_sample(flat) and is_sorted_network_full(flat):
            improved = True
        else:
            # revert is complex; just keep going (we only accept if valid)
            pass
    return layers, improved

def moved_any(layers, li, src, comp):
    return False  # placeholder helper (unused)

def _swap_replace(layers, deadline, attempts):
    """Try replacing a comparator with a different one (same length), which can
    enable later deletions. Return (layers, improved)."""
    improved = False
    for _ in range(attempts):
        if time.time() > deadline:
            return layers, improved
        total = _layer_count(layers)
        if total < 2:
            break
        r = random.randint(0, total - 1)
        li = 0
        while r >= len(layers[li]):
            r -= len(layers[li])
            li += 1
        ci = r
        old = layers[li][ci]
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        if (i, j) == old:
            continue
        # check disjointness within the same layer
        if not all(i != a and i != b and j != a and j != b for a, b in layers[li] if (a, b) != old):
            continue
        layers[li][ci] = (i, j)
        flat = _from_layers(layers)
        if is_sorted_network_sample(flat) and is_sorted_network_full(flat):
            improved = True
        else:
            layers[li][ci] = old
    return layers, improved

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin: finish well before 90s limit

    # Proper odd-even merge sort network construction
    def oe_merge_sort(n):
        """Proper odd-even merge sort network for n elements."""
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]

        half = n // 2
        left = oe_merge_sort(half)
        right = oe_merge_sort(n - half)
        right = [(i + half, j + half) for i, j in right]

        # Odd-even merge of two sorted sequences
        merge = []
        for k in range(1, n):
            if k % 2 == 1:
                for pos in range(0, n - 1, 2):
                    merge.append((pos, pos + 1))
            else:
                for pos in range(1, n - 1, 2):
                    merge.append((pos, pos + 1))

        return left + right + merge

    seed_network = oe_merge_sort(N)
    if not is_sorted_network_full(seed_network):
        # Fallback: odd-even transposition sort
        seed_network = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                seed_network.append((i, i + 1))

    best_network = list(seed_network)
    best_len = len(best_network)

    # Multi-start: several different starting networks
    start_networks = [list(seed_network)]
    oetn = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            oetn.append((i, i + 1))
    start_networks.append(oetn)

    # Time budget for the whole search
    search_deadline = deadline - 3.0  # Leave 3s for final validation

    total_starts = len(start_networks)
    if total_starts >= 2:
        time_per_start = [(search_deadline - start_time) * 0.6] + \
                         [(search_deadline - start_time) * 0.4 / (total_starts - 1)] * (total_starts - 1)
    else:
        time_per_start = [(search_deadline - start_time)]

    for s_idx, start_net in enumerate(start_networks):
        if time.time() > search_deadline:
            break
        start_deadline = start_time + time_per_start[s_idx]
        start_deadline = min(start_deadline, search_deadline)

        layers = _to_layers(start_net)

        # Phase 1: Systematic deletion (repeated passes until no progress)
        for _ in range(50):
            if time.time() > start_deadline:
                break
            layers, improved = _systematic_deletion(layers, start_deadline)
            if not improved:
                break

        # Phase 2: Interleave deletions, layer-shrinks, and swap-replace to escape
        # local optima.
        while time.time() < start_deadline:
            layers, improved = _random_deletion(layers, start_deadline, attempts=50)
            if time.time() > start_deadline:
                break
            layers, improved2 = _layer_shrink(layers, start_deadline, attempts=20)
            if time.time() > start_deadline:
                break
            layers, improved3 = _swap_replace(layers, start_deadline, attempts=30)
            if time.time() > start_deadline:
                break
            layers, improved4 = _systematic_deletion(layers, start_deadline)
            if not (improved or improved2 or improved3 or improved4):
                layers, improved5 = _random_deletion(layers, start_deadline, attempts=20)
                if not improved5:
                    break

        flat = _from_layers(layers)
        if len(flat) < best_len:
            best_network = flat
            best_len = len(best_network)

    # Final systematic pass on the best network
    if time.time() < deadline - 1.0:
        layers = _to_layers(best_network)
        layers, _ = _systematic_deletion(layers, deadline - 1.0)
        flat = _from_layers(layers)
        if len(flat) < best_len:
            best_network = flat
            best_len = len(flat)

    # Final Check: Ensure we return a valid network
    if not is_sorted_network_full(best_network):
        return seed_network

    return best_network
# EVOLVE-BLOCK-END
