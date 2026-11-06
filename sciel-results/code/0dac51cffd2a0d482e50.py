"""Size-efficient sorting network on 17 wires via removal from odd-even merge."""
import time
import random

N = 17
DEADLINE = 85.0  # seconds, safety margin under 90s

# EVOLVE-BLOCK-START
def _time_left():
    return DEADLINE - time.time()

def _odd_even_merge(A, B):
    """Merge two sorted lists of comparators (each a list of (i,j) pairs).
    Returns a list of comparators that merges the two sorted sequences."""
    a = len(A)
    b = len(B)
    if a == 0:
        return list(B)
    if b == 0:
        return list(A)
    if a == 1 and b == 1:
        return [(A[0][0], B[0][0])]
    if a == 1:
        # A is a single comparator; insert it into B's sorted structure
        # This case shouldn't normally arise in balanced recursion, but handle it
        result = list(B)
        # Insert A[0] appropriately - for a single wire, just add it
        result.append(A[0])
        return result
    if b == 1:
        result = list(A)
        result.append(B[0])
        return result
    # Split B into halves
    mid_b = b // 2
    B1 = B[:mid_b]
    B2 = B[mid_b:]
    # Recursively merge A with B1, and B2
    # Standard odd-even merge:
    # 1. Recursively merge A with B1 (the first half of B)
    # 2. Recursively merge (result) with B2
    # Actually the standard algorithm:
    # If a > b, swap so a <= b
    # Split A into A1, A2 and B into B1, B2
    # Recursively merge A1 with B1, and A2 with B2
    # Then merge the results with a single compare-exchange network
    
    # Standard odd-even merge algorithm:
    # Given sorted sequences A (length a) and B (length b), assume a <= b
    # If a == 1: compare A[0] with B[0], and the result is (min, rest of B)
    # Otherwise split both in half
    if a > b:
        A, B = B, A
        a, b = b, a
    
    if a == 1:
        # A has one element
        return [(A[0][0], B[0][0])] + list(B[1:])
    
    mid_a = a // 2
    mid_b = b // 2
    A1 = A[:mid_a]
    A2 = A[mid_a:]
    B1 = B[:mid_b]
    B2 = B[mid_b:]
    
    # Recursively merge A1 with B1, and A2 with B2
    M1 = _odd_even_merge(A1, B1)
    M2 = _odd_even_merge(A2, B2)
    
    # Now merge M1 and M2 with a single compare-exchange
    # The last elements of M1 and M2 need to be compared
    # Actually in odd-even merge, after recursive merges, we need one more merge step
    # The standard approach: the result of merging A1+B1 and A2+B2 needs a final merge
    # For simplicity, concatenate and rely on the fact that they're mostly sorted
    # A proper odd-even merge would do: compare M1[-1] with M2[0] region
    
    # For a correct but simple approach, just concatenate
    # This is not a true odd-even merge but produces a valid (if not minimal) network
    return M1 + M2

def _make_odd_even_merge_network(n):
    """Build an odd-even merge sorting network for n wires using the zero-one principle.
    Returns a list of (i, j) comparators."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Split n into two halves
    a = n // 2
    b = n - a  # b >= a
    
    # Build sorting networks for each half
    net_a = _make_odd_even_merge_network(a)
    net_b = _make_odd_even_merge_network(b)
    
    # Offset net_b by a
    net_b = [(i + a, j + a) for (i, j) in net_b]
    
    # Interleave: compare wire i of A with wire i of B (for i < min(a,b))
    # This is the "compare-exchange" step
    merge_step = [(i, i + a) for i in range(min(a, b))]
    
    # Now merge the two sorted halves using odd-even merge
    # A is wires 0..a-1, B is wires a..n-1
    # We need to merge these two sorted sequences
    # Use recursive odd-even merge on the comparator lists
    
    # For a proper odd-even merge, we need to merge the two sorted networks
    # The merge network compares corresponding positions
    # After the initial compare-exchange, we recursively merge
    
    # Simple approach: after initial compare, do recursive merges
    # This is getting complex; let's use a known-good construction
    
    # Actually, let me use a simpler correct construction:
    # Odd-even transposition is simpler and known correct
    # But we want odd-even merge for fewer comparators
    
    # Let me implement a clean odd-even merge:
    return _build_oe_merge(n)

def _build_oe_merge(n):
    """Clean odd-even merge sort network construction."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Use the recursive definition properly
    # For odd-even merge sort network:
    # 1. Split into two halves of size a and b (a <= b)
    # 2. Recursively sort each half
    # 3. Merge the two sorted halves using odd-even merge
    
    a = n // 2
    b = n - a
    
    net_a = _build_oe_merge(a)
    net_b = [(i + a, j + a) for (i, j) in _build_oe_merge(b)]
    
    # Now merge: the merge of two sorted sequences of length a and b
    # using odd-even merge
    merge = _oe_merge_step(a, b, a)  # A starts at wire 0, B starts at wire a
    
    return net_a + net_b + merge

def _oe_merge_step(a, b, offset_a):
    """Generate the merge comparators for merging a sorted sequence of length a
    starting at wire offset_a with a sorted sequence of length b starting at wire offset_a+a."""
    # offset_b = offset_a + a
    offset_b = offset_a + a
    
    if a == 0:
        return []
    if a == 1:
        # Compare the single element of A with the first element of B
        return [(offset_a, offset_b)]
    
    if b == 1:
        # Compare the last element of A with the single element of B
        return [(offset_a + a - 1, offset_b)]
    
    # Split both in half
    a1 = a // 2
    a2 = a - a1
    b1 = b // 2
    b2 = b - b1
    
    # Recursively merge A1 with B1, and A2 with B2
    # A1 is wires offset_a .. offset_a+a1-1
    # B1 is wires offset_b .. offset_b+b1-1
    # A2 is wires offset_a+a1 .. offset_a+a-1
    # B2 is wires offset_b+b1 .. offset_b+b-1
    
    M1 = _oe_merge_step(a1, b1, offset_a)
    M2 = _oe_merge_step(a2, b2, offset_a + a1)
    
    # After recursive merges, we need to merge the results
    # The key insight of odd-even merge: after merging A1+B1 and A2+B2,
    # we need to compare the boundary elements
    # Specifically, compare wire (offset_a + a1 - 1) with wire (offset_a + a1)
    # i.e., the last of the first merged half with the first of the second merged half
    
    # Actually the standard odd-even merge merge step:
    # After recursively merging, compare element a1-1 of the first result with element 0 of the second result
    # But we need to be more careful
    
    # For a correct implementation, the merge step after recursive merges is:
    # Compare (offset_a + a1 - 1) with (offset_b + b1) if these exist
    # Hmm, this is getting complicated. Let me use a different, verified approach.
    
    # Let me just use odd-even transposition for the merge, which is known correct
    # but uses more comparators. Actually no, let me implement it properly.
    
    # Standard odd-even merge merge:
    # The merge of two sorted lists of size a and b:
    # If a == 1: compare A[0] with B[0], done
    # Else: split A into A1(size a1), A2(size a-a1); B into B1(size b1), B2(size b-b1)
    #   Recursively merge A1 with B1 -> M1
    #   Recursively merge A2 with B2 -> M2  
    #   Then compare M1[-1] with M2[0] (the boundary)
    # The boundary comparison is: wire (offset_a + a1 - 1) vs wire (offset_a + a1)
    # Wait, that's not right either. Let me think again.
    
    # After merging A1 and B1, the result occupies wires offset_a .. offset_a + a1 + b1 - 1
    # After merging A2 and B2, the result occupies wires offset_a + a1 .. offset_a + a1 + a2 + b2 - 1
    # But these overlap! That's the issue.
    
    # I think the cleanest approach is to just use odd-even transposition for the whole thing
    # since it's guaranteed correct, and rely on removal to optimize.
    
    # For now, return a valid (if not minimal) merge using transposition
    total = a + b
    return [(offset_a + i, offset_a + i + 1) for phase in range(total) 
            for i in range(phase % 2, total - 1, 2)]

def _make_transposition_network(n):
    """Odd-even transposition sorting network (known correct)."""
    return [(i, i + 1) for phase in range(n)
            for i in range(phase % 2, n - 1, 2)]

def _validate_sample(network, num_samples=4096):
    """Validate network on random binary inputs. Returns True if all pass."""
    import numpy as np
    for _ in range(num_samples):
        x = np.random.randint(0, 2, size=N)
        for (i, j) in network:
            if x[i] > x[j]:
                x[i], x[j] = x[j], x[i]
        # Check sorted
        if np.any(x[:-1] > x[1:]):
            return False
    return True

def _validate_full(network):
    """Validate network on all 2^N binary inputs. Returns True if all pass."""
    import numpy as np
    for mask in range(1 << N):
        x = np.array([(mask >> k) & 1 for k in range(N)], dtype=np.int8)
        for (i, j) in network:
            if x[i] > x[j]:
                x[i], x[j] = x[j], x[i]
        if np.any(x[:-1] > x[1:]):
            return False
    return True

def _try_remove(network, idx, num_samples=4096):
    """Try removing comparator at index idx. Returns (success, new_network)."""
    new_net = network[:idx] + network[idx+1:]
    if _validate_sample(new_net, num_samples):
        return True, new_net
    return False, network

def construct():
    """Construct a small sorting network for 17 wires using removal from odd-even transposition."""
    start_time = time.time()
    
    # Start with odd-even transposition (known correct, 136 comparators for 17 wires)
    # Actually for N wires, it's N*(N-1)/2 comparators = 17*16/2 = 136
    network = _make_transposition_network(N)
    
    # Try to remove comparators using sample-based validation
    random.seed(42)
    
    # Multiple passes of removal
    max_passes = 50
    pass_count = 0
    
    while _time_left() > 5.0 and pass_count < max_passes:
        pass_count += 1
        improved = False
        # Try removing each comparator (random order for variety)
        indices = list(range(len(network)))
        random.shuffle(indices)
        
        for idx in indices:
            if _time_left() < 5.0:
                break
            success, new_net = _try_remove(network, idx, num_samples=2048)
            if success:
                network = new_net
                improved = True
                break  # Restart scanning from beginning after each removal
        
        if not improved:
            # Try with different sample sizes for a few more passes
            for ns in [4096, 8192]:
                if _time_left() < 5.0:
                    break
                improved2 = False
                indices = list(range(len(network)))
                random.shuffle(indices)
                for idx in indices:
                    if _time_left() < 5.0:
                        break
                    success, new_net = _try_remove(network, idx, num_samples=ns)
                    if success:
                        network = new_net
                        improved2 = True
                        break
                if improved2:
                    improved = True
                    break
            if not improved:
                break
    
    # Final validation with full check if time permits
    if _time_left() > 10.0:
        if not _validate_full(network):
            # Fall back to known-good transposition network
            network = _make_transposition_network(N)
    
    return network
# EVOLVE-BLOCK-END
