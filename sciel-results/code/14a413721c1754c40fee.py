"""Size-efficient sorting networks on 18 wires."""
import time
import itertools
import numpy as np

N = 18

# EVOLVE-BLOCK-START

def _build_wires():
    """Pre-compute the 2^N boolean input wires as numpy arrays."""
    wires = []
    for i in range(N):
        wires.append((np.arange(1 << N) >> i) & 1)
    return wires


def _apply_network(network, wires=None):
    """Apply network to all 2^N boolean inputs. Returns True if all sorted."""
    if wires is None:
        wires = _build_wires()
    
    # Work on a copy to avoid mutation
    w = [w.copy() for w in wires]
    
    for (a, b) in network:
        wa = w[a]
        wb = w[b]
        new_a = np.minimum(wa, wb)
        new_b = np.maximum(wa, wb)
        w[a] = new_a
        w[b] = new_b
    
    # Check if sorted: wire[i] <= wire[i+1] for all i
    for i in range(N - 1):
        if np.any(w[i] > w[i + 1]):
            return False
    return True


def _verify_network(network):
    """Verify network sorts all boolean inputs. Returns True if correct."""
    return _apply_network(network)


def _batcher_odd_even_merge_sort(n):
    """Build Batcher's odd-even merge sort network for n wires. Returns list of (i,j)."""
    if n <= 1:
        return []
    
    network = []
    
    def odd_even_merge(k, offset, direction):
        """Odd-even merge of two sorted sequences of length k starting at offset.
        direction: 0 means merge ascending (min to lower indices)"""
        if k == 1:
            return
        half = k // 2
        # Recursively merge odd and even positions
        odd_even_merge(half, offset + 1, direction)
        odd_even_merge(half, offset, direction)
        # Compare-exchange at positions offset + i and offset + half + i for i in 0..half-1
        for i in range(half):
            a = offset + i
            b = offset + half + i
            if a < b:
                network.append((a, b))
            else:
                network.append((b, a))
    
    def odd_even_merge_sort(k, offset):
        """Sort k elements starting at offset using odd-even merge sort."""
        if k <= 1:
            return
        half = k // 2
        # Recursively sort first half and second half
        odd_even_merge_sort(half, offset)
        odd_even_merge_sort(k - half, offset + half)
        # Merge
        odd_even_merge(half, offset, 0)
    
    odd_even_merge_sort(n, 0)
    return network


def _remove_redundant(network, deadline=None):
    """Try to remove comparators that don't affect correctness using vectorized checks."""
    if not network:
        return network
    
    current = list(network)
    base_wires = _build_wires()
    
    # Pre-compute: for each comparator, check if removing it breaks correctness
    # Strategy: iteratively try removal, starting from the end (later comparators
    # are more likely redundant since earlier ones establish structure)
    
    changed = True
    while changed:
        changed = False
        if deadline is not None and time.time() > deadline:
            break
        for i in range(len(current) - 1, -1, -1):
            candidate = current[:i] + current[i+1:]
            if _apply_network(candidate, base_wires):
                current = candidate
                changed = True
                break  # restart from beginning since indices changed
    return current


def _local_search_reduce(network, time_limit):
    """Try to reduce network size via local search with aggressive strategies."""
    current = list(network)
    best = len(current)
    start_time = time.time()
    base_wires = _build_wires()
    
    # Phase 1: Greedy removal from end to start (most effective)
    for i in range(len(current) - 1, -1, -1):
        if time.time() - start_time > time_limit * 0.4:
            break
        candidate = current[:i] + current[i+1:]
        if _apply_network(candidate, base_wires):
            current = candidate
            best = len(current)
    
    # Phase 2: Try swapping adjacent comparators then removing
    for iteration in range(50):
        if time.time() - start_time > time_limit:
            break
        progress = False
        for i in range(len(current) - 1):
            if time.time() - start_time > time_limit:
                break
            # Swap i and i+1
            candidate = current[:]
            candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
            if _apply_network(candidate, base_wires):
                # After valid swap, try removing a comparator
                for j in range(len(candidate) - 1, -1, -1):
                    if time.time() - start_time > time_limit:
                        break
                    cand2 = candidate[:j] + candidate[j+1:]
                    if _apply_network(cand2, base_wires):
                        current = cand2
                        best = len(current)
                        progress = True
                        break
                if progress:
                    break
        if not progress:
            break
    
    # Phase 3: Try removing pairs of comparators (if single removals exhausted)
    if time.time() - start_time < time_limit * 0.8:
        n = len(current)
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                if time.time() - start_time > time_limit:
                    break
                candidate = current[:i] + current[i+1:j] + current[j+1:]
                if _apply_network(candidate, base_wires):
                    current = candidate
                    best = len(current)
    
    return current


def _construct_batcher():
    """Construct Batcher's odd-even merge sort network for N wires."""
    return _batcher_odd_even_merge_sort(N)


def construct():
    deadline = 85.0  # Safety margin: evaluator kills at 90s
    start = time.time()
    
    # Start with Batcher's odd-even merge sort
    network = _construct_batcher()
    base_count = len(network)
    
    # Verify it's correct
    if not _verify_network(network):
        # Fallback: odd-even transposition
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Try to reduce via removal of redundant comparators
    elapsed = time.time() - start
    remaining = deadline - elapsed - 5
    if remaining > 10:
        network = _remove_redundant(network, deadline=deadline - 5)
    
    # Try local search for further reduction
    elapsed = time.time() - start
    remaining = deadline - elapsed - 5  # safety margin
    if remaining > 10:
        network = _local_search_reduce(network, remaining)
    
    # Final verification
    if not _verify_network(network):
        # Fallback to original Batcher
        network = _construct_batcher()
    
    return network

# EVOLVE-BLOCK-END
