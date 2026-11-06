"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 16

# EVOLVE-BLOCK-START
def _verify_network(comparators):
    """Verify sorting network using zero-one principle with numpy vectorization."""
    # Generate all 2^16 boolean inputs as a matrix (65536 x 16)
    n_inputs = 1 << N
    # Create all possible boolean inputs
    # Using bit manipulation for speed
    all_inputs = np.zeros((n_inputs, N), dtype=np.int8)
    for i in range(N):
        # Bit i of each row
        all_inputs[:, i] = (np.arange(n_inputs, dtype=np.uint32) >> i) & 1
    
    # Simulate the network
    state = all_inputs.copy()
    for (i, j) in comparators:
        # min goes to wire i, max goes to wire j
        new_i = np.minimum(state[:, i], state[:, j])
        new_j = np.maximum(state[:, i], state[:, j])
        state[:, i] = new_i
        state[:, j] = new_j
    
    # Check if sorted: each row should be non-decreasing (0s then 1s)
    # For a sorted binary vector, it should be of the form [0...01...1]
    # Check: for each row, all 0s come before all 1s
    # Equivalently: the row is sorted if state[:, k] >= state[:, k+1] is false for all k
    # i.e., state[:, k] <= state[:, k+1] for all k
    diffs = np.diff(state, axis=1)
    # If any diff is -1, it's not sorted
    return not np.any(diffs < 0)

def _batcher_oom():
    """Batcher's odd-even mergesort for 16 wires."""
    comps = []
    
    def merge(a, b):
        """Merge two sorted lists of comparators."""
        comps.extend(a)
        comps.extend(b)
    
    def oom_sort(n):
        """Return comparators for odd-even mergesort on n wires."""
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        # Split into halves
        n1 = n // 2
        n2 = n - n1
        
        # Sort each half
        left = oom_sort(n1)
        right = [(i + n1, j + n1) for (i, j) in oom_sort(n2)]
        
        # Merge
        merged = oom_merge(n1, n2)
        
        return left + right + merged
    
    def oom_merge(n1, n2):
        """Odd-even merge for two sorted halves."""
        if n1 == 0:
            return []
        if n2 == 0:
            return []
        
        # Odd-even merge
        # First: merge odd-indexed and even-indexed elements
        # The odd-even merge pattern
        comps = []
        
        # Step 1: Compare elements at positions (2i, 2i+1) for appropriate i
        # Actually, let me use the standard Batcher OOM construction
        
        # For merging two sorted lists of sizes n1 and n2:
        # The merge network has depth log2(n1+n2)
        
        # Standard Batcher OOM merge:
        # If n1 + n2 == 2: return [(0,1)]
        # If n1 + n2 == 4: merge two pairs
        # General case:
        
        # The OOM merge for sizes n1, n2:
        # First, compare (i, i + n1) for i in range(0, min(n1,n2), 2) -- odd phase
        # Then compare (i, i + n1) for i in range(1, min(n1,n2), 2) -- even phase
        # Recursively merge the two halves
        
        # Let me use a cleaner recursive definition
        
        # For OOM merge of two sorted arrays of sizes n1 and n2 (total n = n1+n2):
        if n1 + n2 <= 1:
            return []
        
        # The OOM merge pattern:
        # Compare pairs (i, i + n1) for i = 0, 2, 4, ... (odd-indexed in 0-based)
        # and (i, i + n1) for i = 1, 3, 5, ... (even-indexed)
        # Then recursively merge the two resulting sorted halves
        
        # Actually, the standard Batcher OOM merge:
        # Given two sorted sequences A (size n1) and B (size n2) concatenated:
        # 1. Compare (i, i+n1) for i = 0, 2, 4, ... up to min(n1,n2)
        # 2. Compare (i, i+n1) for i = 1, 3, 5, ... up to min(n1,n2)  
        # 3. Recursively merge the first half and second half
        
        # Let me just implement it properly
        comps = []
        
        # The OOM merge for total size n = n1 + n2
        # Phase 1: odd-indexed comparisons
        # Compare (i, i + n1) for i in 0, 2, 4, ... where both i < n and i+n1 < n
        for i in range(0, min(n1, n2), 2):
            comps.append((i, i + n1))
        
        # Phase 2: even-indexed comparisons  
        for i in range(1, min(n1, n2), 2):
            comps.append((i, i + n1))
        
        # Now recursively merge the two halves
        # First half: wires 0 to n1-1 (but wait, this isn't quite right)
        
        # Actually, the OOM merge is more subtle. Let me use a known correct implementation.
        
        # Standard Batcher OOM merge for n1 + n2 elements:
        # The merge network M(n1, n2) where n1 <= n2:
        # If n1 == 0: empty
        # If n1 == 1: compare (0, n1), (1, n1), ..., (n2-1, n1) -- no, that's not right either
        
        # Let me just hardcode a known good OOM for 16 wires
        # Or use a different approach entirely
        
        return comps
    
    # This recursive approach is getting complicated. Let me use a known good construction.
    # For 16 wires, I'll use a known 60-comparator network or construct one systematically.
    
    # Let me try a different approach: use the known optimal 16-wire sorting network
    # The best known 16-wire sorting network has 60 comparators.
    
    # I'll construct it using a combination of smaller sorting networks and merges.
    
    # Actually, let me just try to get a good starting point and then optimize.
    # Start with Batcher's bitonic sort which is correct but uses more comparators.
    
    def bitonic_sort_comps(n):
        """Generate comparators for Batcher's bitonic sort on n wires."""
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        comps = []
        
        # Split into halves
        n1 = n // 2
        n2 = n - n1
        
        # Sort first half ascending, second half descending
        left = bitonic_sort_comps(n1)
        right = [(i + n1, j + n1) for (i, j) in bitonic_sort_comps(n2)]
        # Reverse the second half's sort direction by swapping i,j
        right = [(j + n1, i + n1) for (i, j) in bitonic_sort_comps(n2)]
        
        comps.extend(left)
        comps.extend(right)
        
        # Merge the two bitonic halves
        merged = bitonic_merge_comps(n1, n2)
        comps.extend(merged)
        
        return comps
    
    def bitonic_merge_comps(n1, n2):
        """Bitonic merge for two halves."""
        if n1 + n2 <= 1:
            return []
        
        comps = []
        # Compare (i, i + n1) for i in range(n1+n2) where both are valid
        # Actually, bitonic merge: compare (i, n1+i) for i = 0..min(n1,n2)-1
        # Then recursively merge
        
        # Standard bitonic merge of two sorted halves (one ascending, one descending):
        # Compare (i, i + n1) for i = 0, 1, ..., min(n1,n2)-1
        # Then recursively merge the two resulting halves
        
        m = min(n1, n2)
        for i in range(m):
            comps.append((i, i + n1))
        
        # Recursively merge
        # First half: wires 0 to n1-1
        # Second half: wires n1 to n1+n2-1
        # But after the comparisons, we need to recursively merge each half
        
        # The bitonic merge is:
        # 1. Compare (i, i+n1) for i in 0..m-1
        # 2. Recursively merge wires 0..n1-1 (size n1)
        # 3. Recursively merge wires n1..n1+n2-1 (size n2)
        
        # Wait, that's not quite right. The bitonic merge:
        # Given a bitonic sequence of length n = n1 + n2 where first n1 is ascending 
        # and last n2 is descending (or vice versa), we merge by:
        # 1. Compare (i, i + n1) for i = 0, ..., min(n1,n2)-1
        # 2. The result splits into two bitonic sequences of length n1 and n2
        # 3. Recursively merge each
        
        # Actually for bitonic sort, after sorting halves (one up, one down),
        # the merge step compares (i, i+n1) for all valid i, then recursively merges.
        
        # Let me simplify: just use the standard construction
        # For bitonic merge of size n1 + n2:
        # Compare (i, i+n1) for i in 0..min(n1,n2)-1
        # Then recursively bitonic_merge(n1, n2) on each half
        
        # But this is getting circular. Let me just use a flat approach.
        
        return comps
    
    # This is getting too complicated with recursion. Let me just hardcode a known good network.
    # 
    # Known: The optimal 16-wire sorting network has 60 comparators.
    # Let me try to construct one using a systematic approach.
    #
    # Alternative: Use the "AKS" style or just try to find a good network through search.
    #
    # For now, let me start with a simpler correct network and optimize from there.
    
    # Simple approach: use odd-even transposition but try to prune it.
    # Start with the full OET network (120 comparators) and try to remove comparators.
    
    # Actually, let me just return a known good construction.
    # The 60-comparator network for 16 wires is known. Let me try to recall it.
    
    # One approach: use the "Shear sort" or other known constructions.
    
    # For practical purposes, let me implement a local search starting from OET.
    return None  # Placeholder

def construct():
    """Construct a sorting network for 16 wires."""
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline (10s margin from 90s limit)
    
    # Start with odd-even transposition network (120 comparators)
    network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Verify initial network
    if not _verify_network(network):
        raise ValueError("Initial network is invalid")
    
    # Try to reduce the network by removing comparators
    # Local search: try removing each comparator and check if still valid
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(network)):
            if time.time() > deadline:
                break
            # Try removing comparator at idx
            candidate = network[:idx] + network[idx+1:]
            if _verify_network(candidate):
                network = candidate
                improved = True
                break  # Restart from beginning after a removal
    
    # Try swapping comparators to enable further reductions
    # This is more complex, so let's also try adding back and re-removing
    
    # Final verification
    if not _verify_network(network):
        # Fallback to original
        network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    return network
# EVOLVE-BLOCK-END
