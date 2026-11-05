"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18

# EVOLVE-BLOCK-START
# Precompute bit masks for fast verification
_ALL_MASKS = np.arange(1 << N, dtype=np.int64)
_WIRE_BITS = np.array([(_ALL_MASKS >> i) & 1 for i in range(N)], dtype=np.int8)  # shape (N, 2^N)

def _verify_network_bitparallel(net):
    """Verify that a network sorts all 2^N Boolean inputs using bit-parallel ops."""
    wires = _WIRE_BITS.copy()  # shape (N, 2^N)
    for i, j in net:
        wi = wires[i]
        wj = wires[j]
        wires[i] = np.minimum(wi, wj)
        wires[j] = np.maximum(wi, wj)
    for k in range(N - 1):
        if np.any(wires[k] > wires[k + 1]):
            return False
    return True

def _verify_network(net):
    """Verify that a network sorts all 2^N Boolean inputs."""
    return _verify_network_bitparallel(net)

def _depth(net):
    """Compute parallel depth of a network."""
    depth = [0] * N
    for i, j in net:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth) if depth else 0

def _odd_even_merge(m, n):
    """
    Odd-even merge network to merge two sorted sequences of lengths m and n.
    Returns a list of comparators (i, j) with 0 <= i < j < m+n.
    """
    if m == 0 or n == 0:
        return []
    if m == 1 and n == 1:
        return [(0, 1)]
    
    result = []
    # Step 1: Compare (i, m+i) for i in range(min(m, n))
    # This creates two subsequences: evens and odds
    # Actually, odd-even merge works as follows:
    # 1. Compare (i, m+i) for i in 0..min(m,n)-1
    # 2. The resulting "even" positions (0, 2, 4, ...) form one sorted sequence
    #    The resulting "odd" positions (1, 3, 5, ...) form another sorted sequence
    # 3. Recursively merge these two
    
    # More precisely:
    # After comparing (i, m+i), we have:
    # - Positions 0, 2, 4, ... contain the smaller values (sorted)
    # - Positions 1, 3, 5, ... contain the larger values (sorted)
    # Wait, that's not quite right. Let me think more carefully.
    
    # Standard odd-even merge:
    # Given two sorted sequences A[0..m-1] and B[0..n-1] (conceptually placed at positions 0..m-1 and m..m+n-1)
    # Step 1: Compare (i, m+i) for i in range(min(m,n))
    #   This produces two sequences:
    #   - Even-indexed positions: 0, 2, 4, ... which are sorted
    #   - Odd-indexed positions: 1, 3, 5, ... which are sorted
    # Step 2: Recursively merge the even and odd subsequences
    
    # Actually, the standard formulation:
    # 1. Compare (i, m+i) for i = 0, 1, ..., min(m,n)-1
    # 2. Let E be the sequence of values at even positions (0, 2, 4, ...) after step 1
    #    Let O be the sequence of values at odd positions (1, 3, 5, ...) after step 1
    #    Both E and O are sorted
    # 3. Recursively merge E and O, placing results back at even and odd positions
    
    # For the network generation, we need to track which physical wires correspond to
    # the logical even and odd positions.
    
    # Let me use a different approach: build the network recursively with wire mapping.
    
    def _merge_rec(m, n, offset, stride, wires_map):
        """
        Recursively generate merge comparators.
        m, n: lengths of the two sorted sub-sequences to merge
        offset: starting position in the overall array
        stride: 1 for top-level, 2 for next level, etc.
        wires_map: maps logical positions to physical wire indices
        """
        if m == 0 or n == 0:
            return []
        
        comparators = []
        
        # Step 1: Compare corresponding elements
        # In the logical space, we compare position i with position m+i
        # for i in range(min(m, n))
        for i in range(min(m, n)):
            # Logical positions i and m+i in the current sub-problem
            # Physical positions: offset + i*stride and offset + (m+i)*stride
            phys_i = wires_map[offset + i * stride]
            phys_m_i = wires_map[offset + (m + i) * stride]
            a, b = min(phys_i, phys_m_i), max(phys_i, phys_m_i)
            comparators.append((a, b))
        
        # Step 2: Recursively merge even and odd subsequences
        # Even subsequence: logical positions 0, 2, 4, ... from the combined m+n elements
        # Odd subsequence: logical positions 1, 3, 5, ...
        
        # The even subsequence has length ceil((m+n)/2) but we need to figure out
        # how many elements come from each original sequence.
        
        # After step 1, the even positions contain min(A[i], B[i]) for i where both exist,
        # plus any remaining elements from the longer sequence.
        # Similarly for odd positions.
        
        # Let me count:
        # Even positions in combined array of length m+n: positions 0, 2, 4, ...
        # Number of even positions: (m+n+1)//2
        # Number of odd positions: (m+n)//2
        
        # How many of the even positions came from A (first sequence)?
        # A occupies logical positions 0..m-1
        # Even positions from A: 0, 2, 4, ... up to m-1 -> count = (m+1)//2
        # Even positions from B: m, m+2, m+4, ... -> depends on parity of m
        
        # This is getting complex. Let me use a simpler recursive formulation.
        
        # Alternative: just generate the full odd-even merge network directly.
        return comparators
    
    # Let me use a cleaner recursive approach
    def _gen_merge(m, n, wires):
        """
        Generate odd-even merge network for two sorted sequences.
        wires: list of physical wire indices for the m+n logical positions
        Returns list of comparators.
        """
        if m == 0 or n == 0:
            return []
        if m == 1 and n == 1:
            a, b = wires[0], wires[1]
            return [(min(a,b), max(a,b))]
        
        comparators = []
        total = m + n
        
        # Step 1: Compare (i, m+i) for i in range(min(m,n))
        for i in range(min(m, n)):
            a, b = wires[i], wires[m+i]
            comparators.append((min(a,b), max(a,b)))
        
        # Step 2: Split into even and odd subsequences
        # Even logical positions: 0, 2, 4, ...
        # Odd logical positions: 1, 3, 5, ...
        
        even_wires = [wires[i] for i in range(0, total, 2)]
        odd_wires = [wires[i] for i in range(1, total, 2)]
        
        # Determine how many elements from each original sequence go to even/odd
        # A occupies logical positions 0..m-1
        # B occupies logical positions m..m+n-1
        
        # Even positions from A: logical pos 0, 2, 4, ... < m
        m_even = (m + 1) // 2  # number of even positions in 0..m-1
        m_odd = m // 2         # number of odd positions in 0..m-1
        
        # Even positions from B: logical pos m, m+2, m+4, ... < m+n
        # If m is even, B starts at even position m
        # If m is odd, B starts at odd position m
        if m % 2 == 0:
            n_even = (n + 1) // 2
            n_odd = n // 2
        else:
            n_even = n // 2
            n_odd = (n + 1) // 2
        
        # Recursively merge
        comparators.extend(_gen_merge(m_even, n_even, even_wires))
        comparators.extend(_gen_merge(m_odd, n_odd, odd_wires))
        
        return comparators
    
    wires = list(range(m + n))
    return _gen_merge(m, n, wires)

def _odd_even_merge_sort(n):
    """
    Generate an odd-even merge sort network for n wires.
    Uses the recursive structure:
    - Sort first half and second half independently
    - Merge the two sorted halves using odd-even merge
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    other = n - half
    
    # Sort first half
    left_net = _odd_even_merge_sort(half)
    # Sort second half (shifted)
    right_net_raw = _odd_even_merge_sort(other)
    right_net = [(i + half, j + half) for i, j in right_net_raw]
    
    # Merge the two halves
    merge_net = _odd_even_merge(half, other)
    
    return left_net + right_net + merge_net

def _build_initial_network():
    """Build an initial sorting network using odd-even merge sort."""
    net = _odd_even_merge_sort(N)
    # Normalize pairs
    normalized = []
    for i, j in net:
        if i > j:
            i, j = j, i
        if 0 <= i < j < N:
            normalized.append((i, j))
    return normalized

def _local_search(net, max_time=60.0):
    """
    Local search to improve the network.
    Uses greedy deletion first, then targeted replacement.
    """
    start_time = time.time()
    best_net = list(net)
    
    def _elapsed():
        return time.time() - start_time
    
    # Phase 1: Greedy deletion - try removing each comparator
    while _elapsed() < max_time * 0.4:
        improved = False
        for idx in range(len(best_net)):
            if _elapsed() >= max_time * 0.4:
                break
            candidate = best_net[:idx] + best_net[idx+1:]
            if _verify_network(candidate):
                best_net = candidate
                improved = True
                # Don't break - keep trying to remove more
        if not improved:
            break
    
    # Phase 2: Try replacing comparators to enable further deletions
    while _elapsed() < max_time * 0.7:
        improved = False
        for idx in range(len(best_net)):
            if _elapsed() >= max_time * 0.7:
                break
            i, j = best_net[idx]
            # Try replacing with nearby comparators
            for a in range(max(0, i-2), min(N, i+3)):
                for b in range(a+1, min(N, j+3)):
                    if (a, b) == (i, j):
                        continue
                    candidate = list(best_net)
                    candidate[idx] = (a, b)
                    if _verify_network(candidate):
                        best_net = candidate
                        improved = True
                        break
                if improved:
                    break
            if improved:
                # After a replacement, try deleting again
                for idx2 in range(len(best_net)):
                    if _elapsed() >= max_time * 0.7:
                        break
                    candidate = best_net[:idx2] + best_net[idx2+1:]
                    if _verify_network(candidate):
                        best_net = candidate
                        break
        if not improved:
            break
    
    # Phase 3: More aggressive search - try all single replacements
    while _elapsed() < max_time:
        improved = False
        for idx in range(len(best_net)):
            if _elapsed() >= max_time:
                break
            i, j = best_net[idx]
            for a in range(N):
                if _elapsed() >= max_time:
                    break
                for b in range(a+1, N):
                    if (a, b) == (i, j):
                        continue
                    candidate = list(best_net)
                    candidate[idx] = (a, b)
                    if _verify_network(candidate):
                        best_net = candidate
                        improved = True
                        break
                if improved:
                    break
            if improved:
                # Try deleting again
                for idx2 in range(len(best_net)):
                    if _elapsed() >= max_time:
                        break
                    candidate = best_net[:idx2] + best_net[idx2+1:]
                    if _verify_network(candidate):
                        best_net = candidate
                        break
        if not improved:
            break
    
    return best_net

def construct():
    """Construct a sorting network for N wires."""
    start_time = time.time()
    deadline = 85.0  # Leave 5s margin for safety
    
    # Start with odd-even merge sort network
    net = _build_initial_network()
    
    # Verify it's correct
    if not _verify_network(net):
        # Fallback: use odd-even transposition (guaranteed correct)
        net = []
        for phase in range(N):
            if phase % 2 == 0:
                for i in range(0, N - 1, 2):
                    net.append((i, i + 1))
            else:
                for i in range(1, N - 1, 2):
                    net.append((i, i + 1))
    
    # Local search to improve
    remaining = deadline - time.time()
    if remaining > 3.0:
        net = _local_search(net, max_time=min(remaining - 3.0, 80.0))
    
    # Final safety check
    final_net = []
    for i, j in net:
        if 0 <= i < j < N:
            final_net.append((i, j))
    
    # Final verification
    if not _verify_network(final_net):
        # Absolute fallback
        final_net = []
        for phase in range(N):
            if phase % 2 == 0:
                for i in range(0, N - 1, 2):
                    final_net.append((i, i + 1))
            else:
                for i in range(1, N - 1, 2):
                    final_net.append((i, i + 1))
    
    return final_net
# EVOLVE-BLOCK-END
