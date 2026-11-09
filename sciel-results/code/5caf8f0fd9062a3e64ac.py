"""Size-efficient sorting networks on 18 wires."""
import time
import itertools
import numpy as np

N = 18

# EVOLVE-BLOCK-START

def _apply_network(inputs, network):
    """Apply network to all 2^N boolean inputs. Returns number of failures."""
    # inputs: list of N bit arrays, each of length 2^N
    # For efficiency, use integer representation
    n = len(network)
    # Track each wire as an array of 0/1
    wires = []
    for i in range(N):
        # Wire i: bit i of the input
        wires.append((np.arange(1 << N) >> i) & 1)
    
    for (a, b) in network:
        wa = wires[a]
        wb = wires[b]
        # min goes to a, max goes to b
        new_a = np.minimum(wa, wb)
        new_b = np.maximum(wa, wb)
        wires[a] = new_a
        wires[b] = new_b
    
    # Check if sorted: for each input, wire 0 <= wire 1 <= ... <= wire N-1
    # Equivalent: for each i, wire[i] >= wire[i+1] (since sorted ascending means wire[0] is min)
    # Actually sorted means wire[0] <= wire[1] <= ... so wire[i] <= wire[i+1]
    # Failure if wire[i] > wire[i+1] for any i
    for i in range(N - 1):
        if np.any(wires[i] > wires[i + 1]):
            return 1  # at least one failure
    return 0


def _verify_network(network):
    """Verify network sorts all boolean inputs. Returns True if correct."""
    return _apply_network(None, network) == 0


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


def _remove_redundant(network):
    """Try to remove comparators that don't affect correctness."""
    if not network:
        return network
    
    n = len(network)
    # Try removing each comparator and check if still valid
    # This is expensive, so do it iteratively
    current = list(network)
    changed = True
    while changed:
        changed = False
        for i in range(len(current) - 1, -1, -1):
            candidate = current[:i] + current[i+1:]
            if _verify_network(candidate):
                current = candidate
                changed = True
                break  # restart from beginning since indices changed
    return current


def _local_search_reduce(network, time_limit):
    """Try to reduce network size via local search.
    Strategy: try replacing pairs of comparators, or removing one and adding another."""
    current = list(network)
    best = len(current)
    start_time = time.time()
    
    # First pass: try removing each comparator
    for i in range(len(current) - 1, -1, -1):
        if time.time() - start_time > time_limit * 0.5:
            break
        candidate = current[:i] + current[i+1:]
        if _verify_network(candidate):
            current = candidate
            best = len(current)
    
    # Second pass: try swapping adjacent comparators (may enable further removal)
    for iteration in range(20):
        if time.time() - start_time > time_limit:
            break
        progress = False
        for i in range(len(current) - 1):
            if time.time() - start_time > time_limit:
                break
            # Swap i and i+1
            candidate = current[:]
            candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
            if _verify_network(candidate):
                # Now try removing some comparator
                for j in range(len(candidate) - 1, -1, -1):
                    if time.time() - start_time > time_limit:
                        break
                    cand2 = candidate[:j] + candidate[j+1:]
                    if _verify_network(cand2):
                        current = cand2
                        best = len(current)
                        progress = True
                        break
                if progress:
                    break
        if not progress:
            break
    
    return current


def _construct_batcher():
    """Construct Batcher's odd-even merge sort network for N wires."""
    return _batcher_odd_even_merge_sort(N)


def construct():
    deadline = 90.0
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
    if elapsed < deadline - 30:
        network = _remove_redundant(network)
    
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
