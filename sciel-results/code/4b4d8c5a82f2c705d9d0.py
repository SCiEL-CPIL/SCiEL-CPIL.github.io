N = 18

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a sorting network for 18 wires using a bitonic sort approach.
    Uses the structure of Batcher's bitonic sort, adapted for N=18.
    """
    comparators = []
    
    def bitonic_merge(n, offset, direction=1):
        """
        Merge two adjacent sorted sequences of length n/2 into one sorted sequence.
        direction: 1 for ascending, -1 for descending.
        Since our comparator always puts min in lower index, we handle direction
        by the structure of which pairs we compare.
        """
        if n <= 1:
            return
        k = n // 2
        for i in range(k):
            a = offset + i
            b = offset + i + k
            if direction == 1:
                # Standard: min to a, max to b (a < b)
                if a < b:
                    comparators.append((a, b))
                else:
                    comparators.append((b, a))
            else:
                # For descending: we want max to a, min to b
                # But comparator puts min to lower index
                # So if a < b, min goes to a (wrong for descending)
                # We need to swap the roles: compare (b, a) so min goes to b, max to a
                if a < b:
                    comparators.append((b, a))
                else:
                    comparators.append((a, b))
        bitonic_merge(k, offset, direction)
        bitonic_merge(k, offset + k, direction)
    
    def bitonic_sort(n, offset, direction=1):
        """
        Sort n wires starting at offset in the given direction.
        """
        if n <= 1:
            return
        k = n // 2
        bitonic_sort(k, offset, 1)       # First half ascending
        bitonic_sort(k, offset + k, -1)   # Second half descending
        bitonic_merge(n, offset, direction)
    
    # For N=18, we can't directly use power-of-2 bitonic sort.
    # Strategy: Use bitonic sort on 16 wires, then merge in the remaining 2.
    # 
    # Alternative: Use a known efficient network.
    # 
    # Let's use the approach: sort 16 with bitonic (80 comps), sort 2 (1 comp),
    # then merge 16 and 2.
    #
    # For merging a sorted sequence of 16 with a sorted sequence of 2:
    # We can use a "merge" network. The simplest correct approach:
    # Use odd-even merge or a simple comparison-based merge.
    #
    # Actually, let's use a cleaner approach: implement a general bitonic sort
    # that works for N=18 by using the recursive structure with non-power-of-2.
    #
    # Better: Use the fact that we can pad to 32, run bitonic sort, and only
    # keep the first 18. But that wastes comparators on dummy wires.
    #
    # Most practical: Use bitonic sort for 16, then use a small network to
    # insert 2 elements.
    
    # Step 1: Bitonic sort on wires 0-15
    bitonic_sort(16, 0, 1)
    
    # Step 2: Sort wires 16-17
    comparators.append((16, 17))
    
    # Step 3: Merge the sorted 16-element sequence with the sorted 2-element sequence.
    # We need a merge network for 16 and 2.
    # 
    # A simple correct merge: use a "merge" network that takes two sorted inputs
    # and produces one sorted output.
    #
    # For merging A[0..15] (sorted) with B[16,17] (sorted):
    # We can use the following approach:
    # - Compare B[16] with A[7] (median of A)
    # - Based on result, B[16] goes to left or right half
    # - Similarly for B[17]
    #
    # But this requires data-dependent routing, which isn't allowed in a fixed network.
    #
    # Correct approach: Use a known merge network structure.
    # The merge network for m and n has size O((m+n) log(m+n)).
    #
    # Simpler: Just run a full odd-even transposition on all 18 wires after
    # the initial sort. This is wasteful but correct.
    #
    # Better: Use the fact that if we have two sorted sequences, we can merge
    # them using a specific network. Let's use a "compare-exchange merge":
    #
    # For each element in B, insert it into A using a binary search network.
    #
    # Actually, the cleanest correct approach within time constraints:
    # Use bitonic sort for 16 (80), sort 2 (1), then use a simple merge:
    # Compare each B element with all A elements in a specific pattern.
    #
    # Let me use a different strategy: implement a general sorting network
    # using a known efficient construction.
    #
    # Known result: The optimal network for 18 wires is 77 comparators.
    # I can't easily generate that, but I can do better than 153.
    #
    # Let's use: Bitonic sort 16 (80) + Sort 2 (1) + Merge via 16+2 merge network.
    #
    # A correct 16+2 merge network:
    # The two B elements need to be inserted into the 16-element sorted array.
    # 
    # Network for inserting 1 element into sorted array of size n:
    # It's a binary tree of comparisons, size n.
    #
    # For inserting 2 elements into sorted array of size 16:
    # We can do it in 16 + 16 = 32 comparators (insert each separately).
    # Total: 80 + 1 + 32 = 113.
    #
    # But we need to be careful: after inserting the first element, the array
    # is still sorted but shifted. The second insertion must account for this.
    #
    # A simpler correct approach: use a full merge network.
    #
    # Let me just use a known correct construction: 
    # Odd-even merge for 18 wires.
    #
    # Odd-even merge network for N=18:
    # This is complex to generate correctly.
    #
    # Given the time constraints, let me use a practical approach:
    # Use bitonic sort for 16, then use a small number of additional comparators
    # to correctly merge in the last 2 elements.
    
    # For now, let's use a safe but slightly suboptimal approach:
    # Run bitonic sort on all 18 by padding to 32 and only using first 18.
    # But that uses comparators on dummy wires which don't exist.
    #
    # Alternative: Use the current bubble sort (153) as fallback if we can't
    # do better, but try to find a valid improvement.
    #
    # Let me implement a correct merge for 16+2:
    # 
    # The key insight: if A is sorted and B is sorted (|B|=2), we can merge
    # them using the following network:
    # 1. Compare B[0] with A[0], A[1], ..., A[15] in a specific pattern
    # 2. Similarly for B[1]
    #
    # A correct merge network for 16 and 2:
    # Use the "merge" network from Batcher's work.
    #
    # For simplicity and correctness, let's use:
    # - Bitonic sort 16: 80 comparators
    # - Sort 2: 1 comparator  
    # - Then run a "merge" by doing a full odd-even transposition on the
    #   boundary region only.
    #
    # Actually, the simplest correct approach that's better than 153:
    # Use bitonic sort for 16 (80), then use 8 more comparators to merge
    # the 2 elements into the 16-element sorted array.
    #
    # Here's a correct 16+2 merge network with ~16 comparators:
    # 
    # Since B has only 2 elements, and A has 16, we can:
    # 1. Compare B[0] with A[8] (middle of A)
    # 2. The smaller goes to left half, larger to right half
    # 3. Recursively insert into the appropriate half
    #
    # But this is data-dependent. For a fixed network, we need to handle
    # all cases.
    #
    # Let me use a different strategy: implement a simple but correct
    # sorting network that's better than 153.
    #
    # Known: The "AKS" network is too complex. Let's use a practical
    # construction.
    #
    # Final decision: Use bitonic sort for 16 wires (80 comparators),
    # sort the remaining 2 wires (1 comparator), then use a merge network
    # that compares each of the 2 elements with all 16 in a structured way.
    #
    # A correct merge: for each element in B, compare with all elements in A
    # in a specific order that ensures correctness.
    #
    # This is getting complex. Let me just return a known correct network
    # that's better than 153.
    #
    # The odd-even transposition is 153. Let me try to prune it.
    #
    # Actually, let me just implement a correct bitonic sort for 18 by
    # using the recursive structure properly.
    
    # Clear previous comparators and start fresh
    comparators.clear()
    
    # Use a simple but correct approach: 
    # Generate a sorting network using a known efficient method.
    #
    # For N=18, a practical approach is to use the "shell sort" network
    # or a variant of insertion sort network.
    #
    # Let me use a different strategy: use the fact that we can combine
    # bitonic sort for 16 with a small merge network.
    #
    # Correct merge network for 16+2:
    # After sorting A[0..15] and B[16,17], we need to merge them.
    # 
    # The following network is correct for merging two sorted sequences:
    # For i in range(16):
    #   Compare A[i] with B[16]
    #   Compare A[i] with B[17]
    # This is 32 comparators, total 80+1+32=113.
    #
    # But is this correct? No, because after comparing A[i] with B[16],
    # the values move, so subsequent comparisons are affected.
    #
    # A correct merge requires a specific network structure.
    #
    # Given the complexity, let me use a safe fallback: return the bubble
    # sort network (153) to ensure correctness, but try to find a valid
    # improvement within the time budget.
    
    # For now, return the seed program's network to ensure validity.
    # The inspirations didn't provide a valid improvement.
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
# EVOLVE-BLOCK-END
