import numpy as np
import time

N = 16

def _batcher_odd_merge(n):
    """
    Generate Batcher's Odd-Merge sorting network for n inputs.
    Returns a list of (i, j) comparators.
    """
    if n == 1:
        return []
    
    # Recursively sort the first half and second half
    # The Odd-Merge algorithm splits the array into odd and even indexed elements
    # Actually, the standard recursive definition is:
    # 1. Recursively sort the first n/2 and second n/2
    # 2. Merge them using Odd-Merge
    
    def merge_odd_even(n):
        # Merge two sorted lists of length n/2
        # The structure of Odd-Merge:
        # 1. Compare (i, i + n/2) for i in 0..n/2-1 ? No.
        # Let's stick to the recursive construction of the network.
        
        # The comparator network for merging two sorted halves of size n/2:
        # 1. Compare (0, n/2), (1, n/2+1), ..., (n/2-1, n-1)
        # 2. Recursively merge the odd-indexed and even-indexed results?
        
        # Actually, the standard Batcher Odd-Merge network construction:
        # To merge two sorted sequences of length m:
        # 1. Compare (i, i+m) for i=0..m-1
        # 2. Recursively merge the sequence of min(a_i, b_i) and max(a_i, b_i)?
        # No, that's not quite right.
        
        # Let's use the known recursive structure:
        # C(n) for n=2m:
        # 1. Sort first m and second m (done by caller)
        # 2. Merge:
        #    - Compare (i, i+m) for i in 0..m-1
        #    - Then merge the resulting odd and even sub-sequences?
        
        # Actually, it's easier to just hardcode the structure for N=16 or use a library.
        # But we can generate it recursively.
        
        # Let's implement the recursive comparator generation.
        def gen_merge(m):
            # m is the size of each half being merged
            if m == 1:
                return [(0, 1)] # Wait, if m=1, we merge two elements?
                # If we are merging two sorted lists of size 1, we just compare them.
                # But the indices are relative.
            
            # This approach is tricky with relative indices.
            # Let's use the explicit recurrence for the list of comparators.
            
            # For merging two sorted halves of size m (total 2m):
            # 1. Compare (i, i+m) for i=0..m-1
            # 2. Recursively merge the odd-indexed outputs and even-indexed outputs?
            
            # Actually, the Odd-Merge merge step is:
            # 1. Compare (i, i+m) for i=0..m-1
            # 2. The outputs are split into two sequences:
            #    - The "odd" positions (1, 3, 5, ...) and "even" positions (0, 2, 4, ...)?
            #    No, it's:
            #    - The elements at indices 0, 2, 4... form one sorted subsequence?
            #    - The elements at indices 1, 3, 5... form another?
            
            # Let's look at the standard definition:
            # Merge(2m):
            # 1. Compare (i, i+m) for i=0..m-1
            # 2. Merge the odd-indexed elements (1, 3, ..., 2m-1) recursively?
            # 3. Merge the even-indexed elements (0, 2, ..., 2m-2) recursively?
            
            # Let's try to implement this.
            
            coms = []
            # Step 1: Cross compares
            for i in range(m):
                coms.append((i, i + m))
            
            # Step 2: Recursive merges on odd and even indices
            # The odd indices in the 0..2m-1 range are 1, 3, 5, ..., 2m-1
            # The even indices are 0, 2, 4, ..., 2m-2
            
            # We need to map the recursive merge of size m-1 (for odd) and m (for even)?
            # No, the sizes are m and m.
            
            # Actually, the recursive calls are on the sub-sequences.
            # The odd sub-sequence has length m. The even sub-sequence has length m.
            
            # We need to generate comparators for the odd sub-sequence and map them to global indices.
            # Odd indices: 1, 3, 5, ..., 2m-1.
            # If we call gen_merge(m) for the odd part, the local indices 0..m-1 map to 1, 3, ..., 2m-1.
            # Local i -> Global 2*i + 1
            
            # Even indices: 0, 2, 4, ..., 2m-2.
            # Local i -> Global 2*i
            
            # Wait, is the size m for both?
            # Yes, the merge of two sorted lists of size m results in 2m elements.
            # The odd-indexed elements form a sorted list of size m?
            # Let's check for m=2 (total 4).
            # Compare (0,2), (1,3).
            # Odd indices: 1, 3. Even indices: 0, 2.
            # Merge odd (size 2? No, size 1 pair? No, 2 elements).
            # Merge even (2 elements).
            # If we recursively merge size 2 lists?
            # gen_merge(1) for odd?
            # If m=2, we compare (0,2) and (1,3).
            # Then we need to merge the results at 1,3 and 0,2.
            # Merging two elements (size 1 list? No, the sub-sequences have length m=2? No, length m/2? No.)
            
            # Let's look at the recurrence again.
            # C(2m) = 2C(m) + C(m-1) + 2m - 1?
            # No, C(2m) = 2C(m) + 2C(m/2) + 2m - 1 was my previous guess.
            
            # Let's just use the known network for N=16.
            # Or, simpler: Use the Batcher network generator from a known source or derive it carefully.
            
            # Alternative: Use the "AKS" or "Batcher" network from a library if available?
            # No external libs for sorting networks.
            
            # Let's implement the recursive generator properly.
            
            def generate_odd_merge(n):
                if n == 1:
                    return []
                if n == 2:
                    return [(0, 1)]
                
                # Split into two halves of size n/2
                half = n // 2
                coms = []
                
                # 1. Recursively sort the two halves
                # First half: indices 0..half-1
                coms.extend(generate_odd_merge(half))
                
                # Second half: indices half..n-1
                # We need to shift the comparators from the recursive call
                second_half_coms = generate_odd_merge(half)
                for i, j in second_half_coms:
                    coms.append((i + half, j + half))
                
                # 2. Merge the two sorted halves
                # The merge step for Odd-Merge:
                # Compare (i, i + half) for i in 0..half-1
                for i in range(half):
                    coms.append((i, i + half))
                
                # Then recursively merge the odd and even indexed elements
                # The odd indexed elements are at 1, 3, 5, ..., n-1
                # The even indexed elements are at 0, 2, 4, ..., n-2
                
                # The number of odd elements is half. The number of even elements is half.
                
                # We need to generate a merge network for these sub-sequences.
                # But they are not contiguous.
                
                # Let's generate the comparators for the odd sub-sequence (size half)
                # and map them to the global odd indices.
                # Local index k in the odd sub-sequence corresponds to global index 2*k + 1.
                # We call generate_odd_merge(half) ? No, we call a merge function?
                # The sub-sequences are already sorted?
                # No, after the cross compares, the odd and even sub-sequences are not necessarily sorted.
                # But they are "partially" sorted.
                
                # Actually, the recursive step is to merge the odd-indexed outputs and even-indexed outputs.
                # The odd-indexed outputs form a sequence of length half.
                # The even-indexed outputs form a sequence of length half.
                
                # We apply the Odd-Merge network to these sub-sequences?
                # No, we apply the *merge* part of the Odd-Merge algorithm to these sub-sequences?
                # The sub-sequences are not two sorted lists, they are just lists.
                # Wait, the property of Odd-Merge is that after the cross compares,
                # the odd-indexed elements are sorted among themselves? No.
                
                # Let's check the definition again.
                # "The Odd-Merge sort algorithm... splits the array into two arrays of odd and even indexed elements, sorts them recursively, and then merges them."
                # That's the *sort* algorithm.
                # The *merge* network for two sorted lists of length m:
                # 1. Compare (i, i+m) for i=0..m-1.
                # 2. Recursively merge the odd-indexed elements (1, 3, ..., 2m-1) and even-indexed elements (0, 2, ..., 2m-2).
                
                # So, in step 2, we are merging two sequences of length m.
                # Are these sequences sorted?
                # Yes, it can be proven that after step 1, the odd-indexed elements are sorted and the even-indexed elements are sorted.
                # So we can recursively apply the *merge* network to them.
                
                # So we need a function `generate_merge(m)` that generates the merge network for two sorted lists of length m.
                
                def generate_merge(m):
                    if m == 1:
                        return [(0, 1)] # Merge two single elements
                    if m == 2:
                        # Compare (0, 2), (1, 3)
                        # Then merge odd (1,3) and even (0,2)
                        # Merge odd: (1,3) -> local (0,1)
                        # Merge even: (0,2) -> local (0,1)
                        # So: (0,2), (1,3), (1,3)?? No.
                        # Let's trace m=2.
                        # Input: a0,a1,b0,b1.
                        # Compare (0,2), (1,3).
                        # Odd indices: 1, 3. Even indices: 0, 2.
                        # Merge odd: Compare (1,3).
                        # Merge even: Compare (0,2).
                        # Total: (0,2), (1,3), (1,3), (0,2).
                        # Wait, (1,3) is the same as the cross compare?
                        # No, the cross compare was (0,2) and (1,3).
                        # The recursive merge on odd (1,3) is just comparing (1,3).
                        # The recursive merge on even (0,2) is just comparing (0,2).
                        # So we add (1,3) and (0,2) again?
                        # That would be redundant.
                        # Let's check C(4) = 5.
                        # My trace: (0,2), (1,3) [cross], (1,3) [odd merge], (0,2) [even merge].
                        # That's 4 comparators? But C(4)=5.
                        # Ah, the cross compare is (0,2) and (1,3).
                        # The odd merge is on indices 1 and 3.
                        # The even merge is on indices 0 and 2.
                        # So we add (1,3) and (0,2).
                        # Total 4?
                        # But C(4) is 5.
                        # Where is the 5th?
                        
                        # Let's re-read Batcher's Odd-Merge for 4.
                        # 1. Sort first 2: (0,1)
                        # 2. Sort second 2: (2,3)
                        # 3. Merge:
                        #    - Compare (0,2), (1,3)
                        #    - Compare (1,2)
                        # Total: 2 + 2 + 1 = 5.
                        
                        # So the recursive merge step is NOT just comparing the pair.
                        # For m=2, the merge of two sorted pairs is:
                        # (0,2), (1,3), (1,2).
                        
                        # So my recursive definition of `generate_merge` is incomplete.
                        # The recursive step should call `generate_merge` on the sub-sequences.
                        # For m=2, sub-sequences are size 1.
                        # `generate_merge(1)` returns `[]`? No, merging two size-1 lists is 1 comparator.
                        # But the sub-sequences are size m/2? No, size m.
                        # Wait, the odd-indexed elements are 1 and 3. That's 2 elements.
                        # So we are merging two lists of size 1? No, we are sorting the odd-indexed elements?
                        # No, the property is that they are *already sorted*?
                        # If they are already sorted, we don't need to merge them?
                        # But for m=2, after (0,2) and (1,3), are 1 and 3 sorted?
                        # a0<=a1, b0<=b1.
                        # c0=min(a0,b0), c1=max(a0,b0)
                        # c2=min(a1,b1), c3=max(a1,b1)
                        # Are c1 and c3 sorted? c1<=c3? Yes.
                        # Are c0 and c2 sorted? c0<=c2? Yes.
                        # So why do we need (1,2)?
                        # Because the final output must be sorted.
                        # The odd and even sub-sequences are sorted, but interleaving them requires a final merge?
                        # No, the Odd-Merge algorithm says:
                        # 1. Cross compare.
                        # 2. Recursively merge odd and even.
                        
                        # If the sub-sequences are already sorted, the recursive merge should be trivial?
                        # But for m=2, we need (1,2).
                        # (1,2) connects an even index (2) and an odd index (1).
                        # So it's not a merge of the odd sub-sequence with itself, nor even with itself.
                        
                        # I think I am misinterpreting the recursion.
                        # Let's just hardcode the 63-comparator network for N=16 or find a reliable generator.
                        # Given the complexity and time, I will use a known valid 60-comparator network if I can recall it, 
                        # or fall back to the 63-comparator Batcher network and try to prune.
                        
                        # Actually, I can just use the 120-comparator network and prune it.
                        # But 63 is much better.
                        
                        # Let's try to construct the Batcher network for N=16 explicitly.
                        # I will use the recursive structure but fix the base case.
                        
                        pass # Placeholder

            # Since deriving the perfect recursive generator is error-prone, 
            # I will use a simpler approach: 
            # 1. Start with the 120-comparator Odd-Even Transposition network.
            # 2. Use a greedy pruning algorithm with NumPy verification.
            # This is safer and likely to find a network < 120, possibly close to 60.
            
            return []

    # Fallback: Use the 120-comparator network and prune.
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _verify_network_bit_parallel(network):
    """
    Verify if the network sorts all 2^N inputs.
    Uses bit-parallelism: represent the state of all 2^N inputs as a matrix.
    """
    # Create a matrix of shape (2^N, N)
    # Each row is an input vector.
    # We can generate this using np.arange(2**N) and unpacking bits.
    
    # For N=16, 2^16 = 65536.
    # Matrix size: 65536 x 16.
    # This is 1M elements. Fits in memory.
    
    inputs = np.arange(2**N, dtype=np.uint8)
    # Unpack bits
    # We want row i to have bits of i.
    # bit j of row i is (i >> j) & 1.
    
    # Efficient unpacking:
    # Create a matrix of powers of 2?
    # Or use bitwise operations on the range.
    
    # Let's create the matrix directly.
    # M[i, j] = (i >> j) & 1
    # We can do this by broadcasting.
    
    # indices = np.arange(2**N)[:, np.newaxis]
    # bits = np.arange(N)[np.newaxis, :]
    # M = (indices >> bits) & 1
    
    # This creates a 65536x16 matrix.
    
    state = np.zeros((2**N, N), dtype=np.uint8)
    for j in range(N):
        state[:, j] = (np.arange(2**N) >> j) & 1
        
    # Apply comparators
    for i, j in network:
        # Compare wire i and wire j
        # min = state[:, i] & state[:, j]
        # max = state[:, i] | state[:, j]
        
        min_val = state[:, i] & state[:, j]
        max_val = state[:, i] | state[:, j]
        
        state[:, i] = min_val
        state[:, j] = max_val
        
    # Check if sorted
    # Sorted means state[:, 0] >= state[:, 1] >= ... >= state[:, N-1]
    # Or state[:, 0] == 1 if any 1, then 0s.
    # Actually, sorted binary vector is of the form 111...1000...0.
    # So state[:, 0] should be >= state[:, 1], etc.
    
    # Check differences
    # diff = state[:, :-1] - state[:, 1:]
    # If sorted, diff should be 0 or 1.
    # If any diff is -1, it's not sorted.
    
    # Alternatively, check if state[:, 0] == 1 implies all previous are 1?
    # Simpler: check if state[:, :-1] >= state[:, 1:]
    
    if np.any(state[:, :-1] < state[:, 1:]):
        return False
    return True

def construct():
    # Start with a good initial network.
    # Batcher's Odd-Merge for 16 is 63 comparators.
    # I will hardcode a known 60-comparator network if possible, 
    # or generate the 63-comparator one and prune.
    
    # Let's try to generate the Batcher network.
    # If I can't, I'll use the 120-comparator one and prune.
    
    # Given the time, I'll use the 120-comparator one and prune.
    # It's guaranteed to be correct.
    
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Prune using local search
    # We will try to remove comparators.
    # To speed up, we can check validity less frequently?
    # No, we must ensure correctness.
    
    # We'll do a greedy removal:
    # For each comparator in reverse order (later comparators are often redundant),
    # try removing it and check validity.
    
    # To fit in 90s, we might not be able to check all 65536 inputs for every single removal
    # if we do it many times.
    # 65536 * 16 operations per check.
    # If we do 1000 checks, that's 1000 * 1M ops = 1B ops.
    # NumPy can handle this in a few seconds.
    
    # Let's try to remove as many as possible.
    
    # We'll iterate through the network in reverse.
    # If removing a comparator keeps the network valid, remove it.
    
    # To avoid O(N^2) checks, we can just do one pass.
    # If we remove one, the network changes, so we might need to re-check.
    # But greedy removal is usually effective.
    
    # Let's do a few passes.
    
    start_time = time.time()
    timeout = 80.0 # Leave 10s margin
    
    for pass_num in range(10):
        if time.time() - start_time > timeout:
            break
            
        changed = False
        # Iterate in reverse
        for idx in range(len(network) - 1, -1, -1):
            if time.time() - start_time > timeout:
                break
                
            # Try removing network[idx]
            candidate = network[:idx] + network[idx+1:]
            
            if _verify_network_bit_parallel(candidate):
                network = candidate
                changed = True
                # Since we removed one, the indices shift, but we are iterating backwards,
                # so it's safe to continue to idx-1.
                
    return network

# Test the construction
if __name__ == "__main__":
    net = construct()
    print(f"Comparators: {len(net)}")
    # Verify
    print("Valid:", _verify_network_bit_parallel(net))
