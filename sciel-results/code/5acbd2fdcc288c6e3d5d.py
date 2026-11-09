import time
import random
import numpy as np

N = 16
DEADLINE = 87.0  # Safety margin before 90s kill

def _build_batcher(n):
    """Build Batcher's odd-even merge sort network for n wires."""
    def oe_merge_network(n):
        """Odd-even merge network for 2n inputs (n from top half, n from bottom half)."""
        if n == 1:
            return [(0, 1)]
        # Compare (i, n+i) for i in 0..n-1
        comps = [(i, n + i) for i in range(n)]
        # Recursively merge even-indexed and odd-indexed subsequences
        # Even indices: 0, 2, 4, ... -> map to 0, 1, 2, ...
        # Odd indices: 1, 3, 5, ... -> map to 0, 1, 2, ...
        half = n // 2
        even_merge = oe_merge_network(half)
        odd_merge = oe_merge_network(half)
        # Apply even merge on even positions
        for (a, b) in even_merge:
            comps.append((2 * a, 2 * b))
        # Apply odd merge on odd positions
        for (a, b) in odd_merge:
            comps.append((2 * a + 1, 2 * b + 1))
        return comps

    def sort_network(n):
        """Sorting network using odd-even merge sort."""
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        if n % 2 == 1:
            # Handle odd number: sort n-1, then insert the last element
            base = sort_network(n - 1)
            # Insert element n-1 into sorted network of n-1 elements
            extra = [(i, n - 1) for i in range(n - 1)]
            return base + extra
        # Even: split into two halves
        half = n // 2
        left = sort_network(half)
        right = [(i + half, j + half) for (i, j) in sort_network(half)]
        # Odd-even merge the two sorted halves
        merge = oe_merge_network(half)
        return left + right + merge

    return sort_network(n)

def _find_layers(comps):
    """Group comparators into independent layers (no shared wires within a layer)."""
    layers = []
    current_layer = []
    used_wires = set()
    for comp in comps:
        i, j = comp
        if i not in used_wires and j not in used_wires:
            current_layer.append(comp)
            used_wires.add(i)
            used_wires.add(j)
        else:
            if current_layer:
                layers.append(current_layer)
            current_layer = [comp]
            used_wires = {i, j}
    if current_layer:
        layers.append(current_layer)
    return layers

def _verify_network(comps):
    """Verify sorting network using zero-one principle with layer-parallel application."""
    if not comps:
        return True

    # Generate all 2^16 binary inputs as a (65536, 16) array
    # Efficient generation using bit manipulation
    all_inputs = np.zeros((65536, 16), dtype=np.int8)
    for bit in range(16):
        # Column bit: bit k of integer i
        vals = (np.arange(65536, dtype=np.int32) >> bit) & 1
        all_inputs[:, bit] = vals.astype(np.int8)

    # Apply comparators layer by layer
    layers = _find_layers(comps)
    x = all_inputs
    for layer in layers:
        # For each comparator in the layer, compute min/max
        # Since they're independent, we can process them in parallel
        for (i, j) in layer:
            a = x[:, i]
            b = x[:, j]
            x[:, i] = np.minimum(a, b)
            x[:, j] = np.maximum(a, b)

    # Check if each row is sorted (non-decreasing)
    diffs = np.diff(x, axis=1)
    return bool(np.all(diffs >= 0))

def _verify_network_fast(comps):
    """Faster verification: apply all comparators sequentially but with minimal overhead."""
    if not comps:
        return True

    # Generate all 2^16 binary inputs
    all_inputs = np.zeros((65536, 16), dtype=np.int8)
    for bit in range(16):
        vals = (np.arange(65536, dtype=np.int32) >> bit) & 1
        all_inputs[:, bit] = vals.astype(np.int8)

    x = all_inputs
    for (i, j) in comps:
        a = x[:, i]
        b = x[:, j]
        # Use in-place operations where possible
        x[:, i] = np.minimum(a, b)
        x[:, j] = np.maximum(a, b)

    diffs = np.diff(x, axis=1)
    return bool(np.all(diffs >= 0))

def _greedy_reduce(comps, deadline):
    """Try to remove redundant comparators greedily."""
    current = list(comps)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(current)):
            if time.time() > deadline:
                return current
            trial = current[:i] + current[i+1:]
            if _verify_network_fast(trial):
                current = trial
                changed = True
                break  # Restart loop as indices changed
    return current

def _local_search(comps, deadline):
    """Local search to minimize comparator count."""
    best = list(comps)
    best_count = len(best)
    random.seed(42)

    no_improve = 0
    max_no_improve = 8000

    while time.time() < deadline and no_improve < max_no_improve:
        # Try random removal
        if len(best) > 1:
            idx = random.randint(0, len(best) - 1)
            trial = best[:idx] + best[idx+1:]
            if _verify_network_fast(trial):
                best = trial
                best_count = len(best)
                no_improve = 0
                continue

        # Try swapping two adjacent comparators
        if len(best) > 2:
            idx = random.randint(0, len(best) - 2)
            trial = list(best)
            trial[idx], trial[idx + 1] = trial[idx + 1], trial[idx]
            if _verify_network_fast(trial):
                best = trial
                no_improve = 0
                continue

        # Try moving a comparator to a random position
        if len(best) > 2:
            idx = random.randint(0, len(best) - 1)
            comp = best[idx]
            trial = best[:idx] + best[idx+1:]
            new_idx = random.randint(0, len(trial))
            trial.insert(new_idx, comp)
            if _verify_network_fast(trial):
                best = trial
                no_improve = 0
                continue

        no_improve += 1

    return best

def construct():
    # Start with Batcher's odd-even merge sort (75 comparators for n=16)
    comps = _build_batcher(N)
    initial_count = len(comps)

    if not _verify_network_fast(comps):
        # Fallback to odd-even transposition
        comps = [(i, i + 1) for phase in range(N)
                 for i in range(phase % 2, N - 1, 2)]

    # 1. Greedy reduction (most impactful, do first)
    deadline_greedy = time.time() + 25.0
    comps = _greedy_reduce(comps, deadline_greedy)

    # 2. Local search with moves
    remaining = DEADLINE - time.time() - 5.0
    if remaining > 5.0:
        deadline_local = time.time() + remaining
        comps = _local_search(comps, deadline_local)

    # 3. Final greedy sweep
    remaining2 = DEADLINE - time.time() - 2.0
    if remaining2 > 2.0:
        deadline_final = time.time() + remaining2
        comps = _greedy_reduce(comps, deadline_final)

    return comps
