import numpy as np

N = 18

# EVOLVE-BLOCK-START
def _batcher_odd_merge(n):
    """
    Generate a Batcher Odd-Merge sorting network for n wires.
    Returns a list of (i, j) comparators.
    """
    if n == 1:
        return []
    
    # Recursive helper
    def merge(a, b, i, j):
        # a and b are lists of comparators for the two halves
        # i and j are the starting indices for the two halves in the final network
        # This implementation is a bit tricky to get indices right without building a tree.
        # Let's use a standard recursive generator that returns (comparators, offset) or similar.
        # Actually, it's easier to build the network structure directly.
        
        pass

    # Standard Batcher Odd-Merge implementation
    def odd_merge(n):
        if n == 1:
            return []
        
        # Split into odd and even indices
        # Odd indices: 0, 2, 4... (0-indexed in the sub-network)
        # Even indices: 1, 3, 5...
        
        # Recursively sort the odd-indexed wires (0, 2, 4...)
        # These are effectively a network of size ceil(n/2)
        # Recursively sort the even-indexed wires (1, 3, 5...)
        # These are effectively a network of size floor(n/2)
        
        # Let's define the indices carefully.
        # Wires 0..n-1
        # Odd wires: 0, 2, 4, ...
        # Even wires: 1, 3, 5, ...
        
        # Sort odd wires
        # Map wire k in sub-network to wire 2*k in main network
        # Sort even wires
        # Map wire k in sub-network to wire 2*k+1 in main network
        
        comps = []
        
        # Recursive calls
        # The odd sub-network operates on indices 0, 2, 4...
        # The even sub-network operates on indices 1, 3, 5...
        
        # We need a helper that returns comparators relative to a start offset and stride?
        # No, Batcher's algorithm is usually defined on contiguous blocks.
        # Let's use a standard recursive function that takes (start, length, stride)
        
        def generate(start, length, stride):
            if length == 1:
                return []
            
            # Split into two halves
            # First half: start, start+stride, ..., start + (length//2 - 1)*stride
            # Second half: start + length//2*stride, ...
            
            half_len = length // 2
            # Odd merge:
            # 1. Sort the "odd" positions (indices 0, 2, 4... in the local block)
            # 2. Sort the "even" positions (indices 1, 3, 5... in the local block)
            # 3. Merge the results
            
            # The "odd" positions in the local block of length L are at local indices 0, 2, 4...
            # The "even" positions are at local indices 1, 3, 5...
            
            # Recursively sort the sub-network formed by the odd positions
            # This sub-network has length ceil(L/2)
            # Its wires correspond to global indices: start + 0*stride, start + 2*stride, ...
            
            # This stride approach is getting complicated. Let's use a simpler recursive structure.
            # Batcher's Odd-Merge for N wires:
            # 1. Recursively sort the odd-indexed wires (0, 2, 4...)
            # 2. Recursively sort the even-indexed wires (1, 3, 5...)
            # 3. Merge the two sorted sequences.
            
            # Let's implement a function that returns the list of comparators for a network
            # of size n, assuming the wires are 0..n-1.
            
            pass

    # Let's use a known correct implementation from scratch.
    def sort_network(n):
        if n == 1:
            return []
        
        # Split into odd and even
        # Odd wires: 0, 2, 4...
        # Even wires: 1, 3, 5...
        
        # Recursively sort odd wires
        # The odd wires are at indices 0, 2, 4... in the current network.
        # They form a network of size n_odd = (n + 1) // 2
        # The even wires are at indices 1, 3, 5...
        # They form a network of size n_even = n // 2
        
        # We need to generate comparators for these sub-networks and then offset them.
        
        # Helper to generate comparators for a sub-network of size m
        # and map them to the main network indices.
        
        # Actually, it's easier to just write the recursive step directly.
        
        # 1. Sort odd indices
        # The odd indices are 0, 2, 4...
        # Let's call a recursive function that returns comparators for a network
        # of size m, where the k-th wire of the sub-network maps to a specific global index.
        
        def gen(m, start, stride):
            """
            Generate comparators for a network of size m.
            The wires are at global indices: start, start+stride, start+2*stride, ...
            """
            if m == 1:
                return []
            
            # Split m into two halves
            m_odd = (m + 1) // 2
            m_even = m // 2
            
            # 1. Sort the odd-indexed wires of the sub-network
            # These are sub-indices 0, 2, 4...
            # Global indices: start + 0*stride, start + 2*stride, ...
            # This is a network of size m_odd, with stride 2*stride, starting at start.
            comps = gen(m_odd, start, 2 * stride)
            
            # 2. Sort the even-indexed wires of the sub-network
            # These are sub-indices 1, 3, 5...
            # Global indices: start + 1*stride, start + 3*stride, ...
            # This is a network of size m_even, with stride 2*stride, starting at start + stride.
            comps += gen(m_even, start + stride, 2 * stride)
            
            # 3. Merge
            # Now we have two sorted sequences:
            # A: wires at start, start+2*stride, ... (size m_odd)
            # B: wires at start+stride, start+3*stride, ... (size m_even)
            # We need to merge them.
            # The merge step involves comparing specific pairs.
            # For i in range(min(m_odd, m_even)):
            #   Compare A[i] with B[i]
            #   If A[i] > B[i], swap (but they are already sorted internally, so we just need to interleave)
            # Actually, the standard merge step for odd-merge is:
            # For i in range(0, min(m_odd, m_even)):
            #   Compare wire (start + 2*i*stride) with wire (start + (2*i+1)*stride)
            #   Wait, that's not right.
            
            # Let's look up the exact merge step for Batcher's Odd-Merge.
            # After sorting the odds and evens, we have:
            # O_0 <= O_1 <= ... <= O_{m_odd-1}
            # E_0 <= E_1 <= ... <= E_{m_even-1}
            # We want to merge O and E into a single sorted sequence.
            # The standard merge compares O_i with E_i? No.
            
            # The merge step for Batcher's Odd-Merge is:
            # For i in range(0, min(m_odd, m_even)):
            #   Compare O_i with E_i? 
            #   Actually, the merge is done by comparing specific pairs.
            #   Let's use the standard algorithm:
            #   For i in range(0, min(m_odd, m_even)):
            #       Compare (start + 2*i*stride) with (start + (2*i+1)*stride)
            #       This compares O_i with E_i.
            #   But this is not sufficient.
            
            # Correct Merge Step:
            # The merge of two sorted lists A (length L1) and B (length L2) in Batcher's network
            # involves comparing A_i with B_i for i=0..min(L1,L2)-1?
            # No, that's not a full merge.
            
            # Let's rely on a known correct implementation of Batcher's Odd-Merge.
            
            pass
            
        # This manual derivation is error-prone. Let's use a simpler, known correct network generation.
        # Batcher's Bitonic Sort is easier to implement correctly and has comparable size.
        # Bitonic Sort for N=18 (pad to 32? No, bitonic requires power of 2 or can be done with variable sizes).
        # Bitonic Sort size is O(N^2). For N=18, N^2 = 324. Batcher Odd-Merge is O(N^2) but with a smaller constant.
        # Actually, Batcher's Odd-Merge is generally smaller than Bitonic.
        
        # Let's try a different approach: Use the `network` library if available? No, standard library only.
        
        # Let's implement Bitonic Sort. It's easier to get right.
        # Bitonic Sort for N wires.
        # If N is not a power of 2, we can still use it, but the standard implementation assumes power of 2.
        # We can pad to the next power of 2 (32) and then trim? No, that changes the problem.
        # We need a network for exactly 18 wires.
        
        # Alternative: Use a pre-computed optimal or near-optimal network for N=18?
        # The task says "best known network has 77 comparators".
        # If I can hardcode a 77-comparator network, that would be great.
        # But I don't have one memorized.
        
        # Let's go back to Batcher's Odd-Merge. I will implement it carefully.
        
        def odd_merge(n):
            if n == 1:
                return []
            
            # Split into odd and even
            # Odd indices: 0, 2, 4...
            # Even indices: 1, 3, 5...
            
            # Recursively sort odd and even
            # The odd sub-network has size (n+1)//2
            # The even sub-network has size n//2
            
            # We need to map the sub-network indices to the main network indices.
            
            # Let's define a function that generates comparators for a sub-network
            # of size m, where the k-th wire of the sub-network is mapped to
            # a global index given by a function.
            
            # To simplify, let's just generate the comparators for the sub-networks
            # and then offset them.
            
            # Sub-network 1 (Odd): wires 0, 2, 4...
            # Sub-network 2 (Even): wires 1, 3, 5...
            
            # Recursively get comparators for size n_odd and n_even.
            # But the recursive call needs to know the mapping.
            
            # Let's use a helper that takes (size, start, stride)
            
            def gen(size, start, stride):
                if size == 1:
                    return []
                
                # Split size into two halves
                size_odd = (size + 1) // 2
                size_even = size // 2
                
                # 1. Sort the odd-indexed wires of this sub-network
                # These are local indices 0, 2, 4...
                # Global indices: start + 0*stride, start + 2*stride, ...
                # This is a sub-network of size size_odd, start=start, stride=2*stride
                comps = gen(size_odd, start, 2 * stride)
                
                # 2. Sort the even-indexed wires of this sub-network
                # These are local indices 1, 3, 5...
                # Global indices: start + 1*stride, start + 3*stride, ...
                # This is a sub-network of size size_even, start=start+stride, stride=2*stride
                comps += gen(size_even, start + stride, 2 * stride)
                
                # 3. Merge
                # Now we have two sorted sequences:
                # A: global indices start, start+2*stride, ... (size size_odd)
                # B: global indices start+stride, start+3*stride, ... (size size_even)
                
                # The merge step in Batcher's Odd-Merge:
                # For i in range(0, min(size_odd, size_even)):
                #   Compare A[i] with B[i]
                #   A[i] is at start + 2*i*stride
                #   B[i] is at start + stride + 2*i*stride = start + (2*i+1)*stride
                #   So compare (start + 2*i*stride) with (start + (2*i+1)*stride)
                
                # Wait, is this a full merge?
                # If A = [1, 3, 5] and B = [2, 4, 6]
                # Comparing A0-B0 (1,2), A1-B1 (3,4), A2-B2 (5,6)
                # Results: [1,2], [3,4], [5,6] -> 1,2,3,4,5,6. Sorted.
                
                # If A = [1, 4, 7] and B = [2, 3, 5]
                # A0-B0 (1,2) -> 1,2
                # A1-B1 (4,3) -> 3,4
                # A2-B2 (7,5) -> 5,7
                # Result: 1,2,3,4,5,7. Sorted.
                
                # It seems comparing corresponding elements works because both lists are sorted.
                
                merge_count = min(size_odd, size_even)
                for i in range(merge_count):
                    i_a = start + 2 * i * stride
                    i_b = start + (2 * i + 1) * stride
                    comps.append((i_a, i_b))
                
                return comps
            
            return gen(n, 0, 1)

    return odd_merge(n)

def construct():
    # Generate Batcher's Odd-Merge network
    network = _batcher_odd_merge(N)
    
    # Pre-generate all Boolean inputs
    # Use NumPy for fast verification
    num_inputs = 1 << N
    # Create a matrix of shape (num_inputs, N)
    # inputs[i] is the binary representation of i, with N bits
    # We can do this efficiently:
    # np.arange(num_inputs)[:, None] // (1 << np.arange(N)[::-1]) % 2
    inputs = (np.arange(num_inputs)[:, None] // (1 << np.arange(N-1, -1, -1))) % 2
    
    def is_valid(net):
        # Simulate the network on all inputs
        state = inputs.copy()
        for i, j in net:
            # Compare and exchange
            # min to i, max to j
            # Since values are 0 or 1:
            # min = 0 if state[:,i] == 0 or state[:,j] == 0
            # max = 1 if state[:,i] == 1 or state[:,j] == 1
            # We can use np.minimum and np.maximum
            min_val = np.minimum(state[:, i], state[:, j])
            max_val = np.maximum(state[:, i], state[:, j])
            state[:, i] = min_val
            state[:, j] = max_val
        
        # Check if all rows are sorted
        # state[:, 1:] >= state[:, :-1]
        return np.all(state[:, 1:] >= state[:, :-1])
    
    # Prune the network
    # Try removing each comparator in reverse order
    # If the network remains valid, keep it removed
    # Repeat until no more improvements
    improved = True
    while improved:
        improved = False
        for k in range(len(network) - 1, -1, -1):
            # Tentatively remove network[k]
            test_net = network[:k] + network[k+1:]
            if is_valid(test_net):
                network = test_net
                improved = True
    
    return network
# EVOLVE-BLOCK-END
