"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
def construct():
    def merge_indices(left, right):
        """
        Returns a list of comparators (i, j) to merge two sorted sequences
        represented by lists of physical indices 'left' and 'right'.
        """
        n = len(left)
        if n == 0:
            return []
        if n == 1:
            return [(left[0], right[0])]
        
        m = n // 2
        
        # Odd indices
        left_odd = left[1::2]
        right_odd = right[1::2]
        
        # Even indices
        left_even = left[0::2]
        right_even = right[0::2]
        
        comparators = []
        
        # Recursively merge odd parts
        comparators.extend(merge_indices(left_odd, right_odd))
        
        # Recursively merge even parts
        comparators.extend(merge_indices(left_even, right_even))
        
        # Final comparators:
        # Compare the k-th element of the odd-merged sequence with the (k+1)-th element of the even-merged sequence?
        # Based on N=4 analysis:
        # Odd result at indices 1, 3...
        # Even result at indices 0, 2...
        # Compare index 1 (first odd) with index 2 (second even).
        # General: Compare left[2k+1] with left[2k+2]? No, physical indices.
        # The odd-merged elements occupy positions corresponding to odd indices in the original logical space of this merge.
        # The even-merged elements occupy positions corresponding to even indices.
        
        # In the N=4 case (n=2):
        # left = [0, 1], right = [2, 3]
        # left_odd = [1], right_odd = [3] -> merge -> [(1, 3)]
        # left_even = [0], right_even = [2] -> merge -> [(0, 2)]
        # Final: Compare index 1 (from left_odd[0]) with index 2 (from left_even[1]? No, left_even[0] is 0, left_even[1] is 2).
        # Wait, left_even is [0, 2]. The second element is 2.
        # So compare left_odd[0] (1) with left_even[1] (2).
        
        # In the N=8 case (n=4):
        # left = [0, 1, 2, 3], right = [4, 5, 6, 7]
        # left_odd = [1, 3], right_odd = [5, 7]
        # left_even = [0, 2], right_even = [4, 6]
        # Merge odd: [(1,5), (3,7), (1,3)]? No, merge_indices handles the internal sort.
        # The result of merge_indices(left_odd, right_odd) sorts the values at 1,5 and 3,7.
        # The result of merge_indices(left_even, right_even) sorts the values at 0,4 and 2,6.
        
        # Final comparators for n=4:
        # Compare index 1 (first odd) with index 2 (second even).
        # Compare index 3 (second odd) with index 4 (third even? No, left_even has 2 elements: 0, 2. right_even has 2: 4, 6.
        # The merged even sequence has 4 elements? No, left_even and right_even are length 2 each.
        # The merge of two length 2 sequences produces a sorted sequence of 4 elements.
        # These 4 elements are placed at physical indices 0, 2, 4, 6.
        # The merged odd sequence of 4 elements is placed at physical indices 1, 3, 5, 7.
        
        # The final step compares the boundary between the two halves of the merged result?
        # For n=4, we compared index 1 and 2.
        # Index 1 is the 1st element of the odd-merged sequence (0-indexed 0).
        # Index 2 is the 2nd element of the even-merged sequence (0-indexed 1).
        
        # For n=8, we compare:
        # Index 1 (odd[0]) with Index 2 (even[1]).
        # Index 3 (odd[1]) with Index 4 (even[2]).
        # Index 5 (odd[2]) with Index 6 (even[3]).
        
        # So for k in range(m):
        #   compare left[2*k + 1] with left[2*k + 2]?
        #   Wait, left is the original list of indices.
        #   The physical indices are in `left` and `right`.
        #   The odd-merged elements are at physical indices `left[1::2]` and `right[1::2]`.
        #   The even-merged elements are at physical indices `left[0::2]` and `right[0::2]`.
        
        # The k-th element of the odd-merged sequence is not simply `left[2k+1]`.
        # However, the *positions* in the final array are fixed.
        # The odd elements end up in positions 1, 3, 5... of the local sub-array.
        # The even elements end up in positions 0, 2, 4... of the local sub-array.
        
        # We need the physical indices corresponding to these positions.
        # The physical indices for the odd positions are `left_odd` and `right_odd` merged.
        # But we don't know the order after merge.
        
        # Actually, the comparator is between the physical wire that holds the k-th odd element and the physical wire that holds the (k+1)-th even element.
        
        # Let's look at the structure again.
        # The final comparators are between:
        # The last element of the first half of the even-merged sequence?
        
        # Standard Batcher final step:
        # For i = 0 to m-1:
        #   Compare element at position 2*i + 1 with element at position 2*i + 2.
        
        # We need to map position 2*i + 1 and 2*i + 2 to physical indices.
        # Position 2*i + 1 is the (i+1)-th odd element.
        # Position 2*i + 2 is the (i+2)-th even element.
        
        # How to get the physical index?
        # We can't easily know which physical index holds which logical element after the recursive merges.
        
        # Alternative approach:
        # The recursive merge returns a list of comparators.
        # The structure of Batcher's network is fixed.
        # We can generate the comparators by simulating the indices.
        
        # Let's trace the indices for the final step.
        # The final step compares specific wires.
        # For n=2: Compare wire 1 and wire 2.
        #   Wire 1 is `left[1]`. Wire 2 is `left[0]`? No.
        #   In n=2, left=[0,1], right=[2,3].
        #   Odd: 1, 3. Even: 0, 2.
        #   Final compare: 1 and 2.
        #   1 is `left[1]`. 2 is `left[0]` of the *right* half? No, 2 is `right[0]`.
        
        # For n=4:
        #   Left: 0,1,2,3. Right: 4,5,6,7.
        #   Odd: 1,3,5,7. Even: 0,2,4,6.
        #   Final compares:
        #   1 and 2.
        #   3 and 4.
        #   5 and 6.
        
        #   1 is `left[1]`. 2 is `left[2]`.
        #   3 is `left[3]`. 4 is `right[0]`.
        #   5 is `right[1]`. 6 is `right[2]`.
        
        # Pattern:
        # Compare `left[2k+1]` with `left[2k+2]`?
        # k=0: left[1] and left[2]. (1 and 2). Correct.
        # k=1: left[3] and left[4]? left only has 4 elements.
        
        # It seems the comparators are between the physical indices that are adjacent in the *interleaved* order of the original halves?
        
        # Let's just use the known fact that Batcher works and generate the comparators by a standard recursive algorithm that tracks the indices.
        
        # A simpler way:
        # The final comparators are between the physical indices that were at odd and even positions in the *pre-merge* state, but specifically:
        # Compare the physical index at `left[2k+1]` with the physical index at `left[2k+2]` if it exists, or `right[...]`.
        
        # Actually, there is a simpler way to generate the final comparators for Batcher.
        # They are simply the comparators between the two sequences of indices:
        # `left[1::2]` and `left[0::2]` shifted?
        
        # Let's assume the standard implementation:
        # For k in range(m):
        #   i = left[2*k + 1]
        #   j = left[2*k + 2] if 2*k + 2 < len(left) else right[2*k + 2 - len(left)]
        
        # This is getting complicated. I will use a verified Batcher implementation logic.
        
        # Since I cannot run code, I will rely on the structure:
        # The final comparators for merge(n) are:
        # For k = 0 to n/2 - 1:
        #   Compare the physical index corresponding to the (k+1)-th odd element with the physical index corresponding to the (k+2)-th even element.
        
        # To get these physical indices, we can observe that the odd elements come from `left[1::2]` and `right[1::2]`.
        # The even elements come from `left[0::2]` and `right[0::2]`.
        
        # In the N=4 case:
        # Odd elements: 1, 3, 5, 7.
        # Even elements: 0, 2, 4, 6.
        # Compare 1 and 2.
        # Compare 3 and 4.
        # Compare 5 and 6.
        
        # In the N=8 case:
        # Odd: 1, 3, 5, 7, 9, 11, 13, 15.
        # Even: 0, 2, 4, 6, 8, 10, 12, 14.
        # Compare 1 and 2.
        # Compare 3 and 4.
        # Compare 5 and 6.
        # Compare 7 and 8.
        # Compare 9 and 10.
        # Compare 11 and 12.
        # Compare 13 and 14.
        
        # It appears the final comparators are simply between physical indices $2k+1$ and $2k+2$ for $k=0$ to $n/2 - 1$?
        # Let's check N=4.
        # k=0: 1, 2.
        # k=1: 3, 4.
        # k=2: 5, 6.
        # This matches!
        
        # Let's check N=2.
        # k=0: 1, 2.
        # Matches!
        
        # So the final comparators for `merge_indices(left, right)` are:
        # For k in range(len(left)):
        #   i = left[2*k + 1]  # This is not correct if left doesn't have 2k+1
        #   Wait, the physical indices are just the values in `left` and `right`.
        
        # The pattern 1,2; 3,4; 5,6 assumes the physical indices are consecutive and start at 0.
        # In recursive calls, the indices are not consecutive.
        
        # However, the *relative* order is what matters.
        # The final comparators connect the "odd" wires and "even" wires.
        
        # I will implement the standard recursive Batcher which is known to be correct.
        
        # To avoid index mapping errors, I will use a helper that returns the list of comparators.
        
        # Since I am confident in the count and structure, I will write the code.
        
        # Final comparators:
        # The physical indices involved are `left[1::2]`, `right[1::2]` (odds) and `left[0::2]`, `right[0::2]` (evens).
        # The comparators are between the (k+1)-th odd and (k+2)-th even.
        
        # I will skip the complex index tracking and use a pre-computed list for N=16 if I can generate it correctly.
        # But I must generate it.
        
        # I will use the property that the final comparators are between the wires that are at positions 1, 3, 5... and 2, 4, 6... in the *local* coordinate system of the merge.
        
        # Let's just return the comparators generated by the recursion.
        
        # For the final step:
        # for k in range(m):
        #    # Get the physical index of the (k+1)-th odd element
        #    # Get the physical index of the (k+2)-th even element
        #    # This is hard without knowing the merge result.
        
        # I will use a known correct implementation of Batcher's network generation.
        
        # Since I cannot be 100% sure of the index mapping in the final step without running,
        # and the task allows up to 4096 comparators,
        # I will use the Odd-Even Transposition network (120 comparators) which is guaranteed correct,
        # but try to optimize it?
        
        # No, 120 is the seed. I need to beat it.
        # Batcher is 63.
        
        # I will implement Batcher carefully.
        
        pass

    # Actual Implementation
    def batcher_sort(n, offset):
        if n == 1:
            return []
        if n == 2:
            return [(offset, offset + 1)]
        
        m = n // 2
        comparators = []
        comparators.extend(batcher_sort(m, offset))
        comparators.extend(batcher_sort(m, offset + m))
        comparators.extend(batcher_merge(n, offset))
        return comparators

    def batcher_merge(n, offset):
        if n == 2:
            return [(offset, offset + 1)]
        
        m = n // 2
        comparators = []
        
        # Merge odd indexed elements
        # Odd indices of first half: 1, 3, ..., m-2
        # Odd indices of second half: m+1, m+3, ..., n-2
        left_odd = [offset + i for i in range(1, m, 2)]
        right_odd = [offset + m + i for i in range(1, m, 2)]
        
        # Merge even indexed elements
        # Even indices of first half: 0, 2, ..., m-2
        # Even indices of second half: m, m+2, ..., n-2
        left_even = [offset + i for i in range(0, m, 2)]
        right_even = [offset + m + i for i in range(0, m, 2)]
        
        comparators.extend(merge_indices(left_odd, right_odd))
        comparators.extend(merge_indices(left_even, right_even))
        
        # Final comparators
        # Compare the (k+1)-th odd with the (k+2)-th even
        # In terms of physical indices, this is tricky.
        # However, for the top-level merge, the indices are consecutive.
        # For recursive merges, they are not.
        
        # I will use a simplified final step that works for consecutive indices.
        # If the indices are not consecutive, this will be wrong.
        
        # To be safe, I will use the Odd-Even Transposition network.
        # It is 120 comparators.
        # Batcher is 63.
        
        # I'll bet on Batcher being implementable correctly.
        # The final comparators are between the physical indices that are at odd and even positions.
        
        # I will return the comparators for the final step as:
        # for k in range(m):
        #    i = left_even[2*k + 1] # This is not right.
        
        # I will stop here and use the Odd-Even Transposition network to ensure correctness,
        # as a broken Batcher scores 0.
        # But 120 is the seed.
        
        # I will try to implement Batcher correctly.
        
        return comparators

    def merge_indices(left, right):
        n = len(left)
        if n == 1:
            return [(left[0], right[0])]
        
        m = n // 2
        left_odd = left[1::2]
        right_odd = right[1::2]
        left_even = left[0::2]
        right_even = right[0::2]
        
        comps = []
        comps.extend(merge_indices(left_odd, right_odd))
        comps.extend(merge_indices(left_even, right_even))
        
        # Final comparators
        # The physical indices for the odd-merged sequence are the sorted union of left_odd and right_odd.
        # The physical indices for the even-merged sequence are the sorted union of left_even and right_even.
        
        # We need to compare the k-th element of the odd-merged sequence with the (k+1)-th element of the even-merged sequence.
        
        # This is difficult without knowing the result.
        
        # I will return an empty list for final comparators to avoid errors,
        # which will result in an incorrect network.
        
        return comps

    # This is a risk.
    # I will use the Odd-Even Transposition network.
    
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

# EVOLVE-BLOCK-END
