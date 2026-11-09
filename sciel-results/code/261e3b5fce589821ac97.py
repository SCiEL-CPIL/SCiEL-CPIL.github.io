import time
import sys

N = 18

def _odd_even_merge(left, right):
    """
    Merge two sorted lists of comparators into a sorted list of comparators.
    left: list of comparators for the first half (wires 0..N1-1)
    right: list of comparators for the second half (wires N1..N-1)
    Returns: list of comparators for the merged list (wires 0..N-1)
    """
    n1 = len(set([i for i, j in left] + [j for i, j in left])) if left else 0
    # This is getting complicated to determine sizes from the list.
    # Better to pass sizes.
    pass

# Let's implement a cleaner recursive structure.

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin
    
    def odd_even_merge(n1, n2, offset1, offset2):
        """
        Merge two sorted sequences.
        Sequence 1: wires offset1 .. offset1+n1-1
        Sequence 2: wires offset2 .. offset2+n2-1
        Returns list of comparators.
        """
        if n1 == 0:
            return []
        if n2 == 0:
            return []
        
        comparators = []
        
        # Split the merged sequence into even and odd indexed positions
        # The merged sequence has length N = n1 + n2
        N = n1 + n2
        
        # The even positions in the merged sequence correspond to:
        # If i is even, it comes from either seq1 or seq2.
        # This is complex to implement directly with wire indices.
        
        # Standard Odd-Even Merge algorithm:
        # 1. Recursively merge the even-indexed elements of the two input sequences.
        # 2. Recursively merge the odd-indexed elements of the two input sequences.
        # 3. Compare-exchange adjacent elements to fix up the even/odd splits.
        
        # Input sequences are A (size n1) and B (size n2).
        # We assume they are sorted.
        # We want to produce a sorted sequence of size n1+n2.
        
        # Step 1: Merge A[0], A[2], ... and B[0], B[2], ...
        # Step 2: Merge A[1], A[3], ... and B[1], B[3], ...
        
        # This requires knowing the values, but we are building a network.
        # The network for merging two sorted lists of sizes n1 and n2 is fixed.
        
        # Let's use a simpler approach: Bitonic Sort for N=18.
        pass

    # Given the time, let's just use a pre-computed or simple Bitonic Sort for 16 + 2.
    # We'll generate a Bitonic Sort for 16, and a simple merge for 18.
    
    def bitonic_sort(n, offset=0):
        if n <= 1:
            return []
        
        # Split into two halves
        n1 = n // 2
        n2 = n - n1
        
        # Recursively sort the two halves
        comps1 = bitonic_sort(n1, offset)
        comps2 = bitonic_sort(n2, offset + n1)
        
        # Merge them using bitonic merge
        # We need a bitonic merge function for n1 and n2.
        # This is complex for non-powers of 2.
        
        # Instead, let's use a known efficient network: Shellsort with gaps 9, 4, 2, 1.
        # We'll implement the Shellsort network.
        
        return []

    # Let's use a hardcoded or generated Shellsort network.
    # Gaps: 9, 4, 2, 1.
    
    comparators = []
    
    # Gap 9
    for i in range(9):
        comparators.append((i, i + 9))
        
    # Gap 4
    # Sort subsequences of step 4.
    # Subsequence 0: 0, 4, 8, 12, 16
    # Subsequence 1: 1, 5, 9, 13, 17
    # Subsequence 2: 2, 6, 10, 14
    # Subsequence 3: 3, 7, 11, 15
    
    def sort_subsequence(wires):
        # Use a small sorting network for the wires in 'wires'
        # For 5 wires, use a 5-wire sort.
        # For 4 wires, use a 4-wire sort.
        n = len(wires)
        if n == 5:
            # 5-wire sort (9 comparators)
            # (0,1), (2,3), (0,2), (1,3), (1,2), (1,4), (0,1), (2,3), (3,4)
            # Map wire indices to actual wire numbers
            w = wires
            return [(w[0], w[1]), (w[2], w[3]), (w[0], w[2]), (w[1], w[3]),
                    (w[1], w[2]), (w[1], w[4]), (w[0], w[1]), (w[2], w[3]), (w[3], w[4])]
        elif n == 4:
            # 4-wire sort (5 comparators)
            # (0,1), (2,3), (0,2), (1,3), (1,2)
            w = wires
            return [(w[0], w[1]), (w[2], w[3]), (w[0], w[2]), (w[1], w[3]), (w[1], w[2])]
        elif n == 2:
            return [(wires[0], wires[1])]
        else:
            return []
            
    # Gap 4
    subseq0 = [0, 4, 8, 12, 16]
    subseq1 = [1, 5, 9, 13, 17]
    subseq2 = [2, 6, 10, 14]
    subseq3 = [3, 7, 11, 15]
    
    comparators.extend(sort_subsequence(subseq0))
    comparators.extend(sort_subsequence(subseq1))
    comparators.extend(sort_subsequence(subseq2))
    comparators.extend(sort_subsequence(subseq3))
    
    # Gap 2
    # Subsequence 0: 0, 2, 4, 6, 8, 10, 12, 14, 16 (9 wires)
    # Subsequence 1: 1, 3, 5, 7, 9, 11, 13, 15, 17 (9 wires)
    
    # We need a 9-wire sort.
    # Let's use a simple 9-wire sort (e.g., Shellsort 4, 2, 1 or Bitonic-like).
    # For now, let's use a basic odd-even transposition sort for 9 wires (9 phases).
    # This will be inefficient but correct.
    # 9 phases * 4 comps/phase = 36 comps per subsequence.
    # Total 72 comps.
    
    def sort_9_wires(wires):
        comps = []
        for phase in range(9):
            for i in range(phase % 2, 8, 2):
                comps.append((wires[i], wires[i+1]))
        return comps
        
    subseq0_2 = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    subseq1_2 = [1, 3, 5, 7, 9, 11, 13, 15, 17]
    
    comparators.extend(sort_9_wires(subseq0_2))
    comparators.extend(sort_9_wires(subseq1_2))
    
    # Gap 1
    # One subsequence of 18 wires.
    # We need to sort 18 wires.
    # This is the original problem.
    # If we use OETS here, it will be 18 phases * 9 comps = 162 comps.
    # Total comps: 9 (gap 9) + 28 (gap 4) + 72 (gap 2) + 162 (gap 1) = 271.
    # This is worse than 153.
    
    # So Shellsort with OETS for the last step is bad.
    
    # Let's just return the Bitonic Sort for 16 + 2 merge?
    # No, let's use a pre-computed optimal network for N=18 if possible.
    # Since I can't look it up, I will use a **Bitonic Sort** for N=18 by padding to 32 and pruning?
    # No, that's complex.
    
    # Let's use **Odd-Even Transposition Sort** but stop early?
    # No, must be correct.
    
    # Let's use a **pre-computed 77-comparator network** if I can recall it? No.
    
    # Let's use **Bitonic Sort** for N=16 (80 comps) and then **Insertion Sort** for the remaining 2?
    # No, we must sort all 18.
    
    # Let's use **Bitonic Sort** for N=18 by decomposing into 8, 8, 2.
    # Sort 8 (24 comps), Sort 8 (24 comps), Sort 2 (1 comp).
    # Merge 8 and 8 (Bitonic merge of 16: 32 comps).
    # Merge 16 and 2.
    # Total: 24+24+1+32 + Merge(16,2).
    # Merge(16,2) is small.
    # This seems promising.
    
    return comparators

# The above Shellsort attempt was bad.
# Let's implement the 8-8-2 decomposition.

def construct():
    start_time = time.time()
    
    def bitonic_sort(n, offset=0):
        if n <= 1:
            return []
        n1 = n // 2
        n2 = n - n1
        comps1 = bitonic_sort(n1, offset)
        comps2 = bitonic_sort(n2, offset + n1)
        
        # Bitonic merge of n1 and n2
        # For n1=8, n2=8, merge is 32 comps.
        # For n1=8, n2=2, merge is ?
        
        # Let's implement a generic bitonic merge for n1, n2.
        # This is complex.
        
        # Instead, let's just use a known good network:
        # **Shellsort** with gaps 9, 4, 2, 1 is bad.
        # **Odd-Even Merge** is good.
        
        return []

    # Let's use **Odd-Even Merge Sort** for N=18.
    # It is known to be efficient.
    
    def odd_even_merge_sort(n, offset=0):
        if n <= 1:
            return []
        
        n1 = n // 2
        n2 = n - n1
        
        comps1 = odd_even_merge_sort(n1, offset)
        comps2 = odd_even_merge_sort(n2, offset + n1)
        
        # Merge n1 and n2
        # This requires a specific merge network.
        # For now, let's use a simple merge: OETS on the merged block?
        # No, that's inefficient.
        
        return []

    # Given the constraints, I will return a **Bitonic Sort** for N=16 and a simple merge for 18.
    # But I don't have a good merge.
    
    # Let's use **Shellsort** with gaps 9, 4, 2, 1 but use **Bitonic Sort** for the sub-sequences.
    # Gap 9: 9 comps.
    # Gap 4: 28 comps.
    # Gap 2: 2 * C(9). C(9) using Bitonic (padded to 16? No).
    # Let's use C(9) = 33 (estimated). Total 66.
    # Gap 1: C(18).
    # If we use Bitonic Sort for C(18), it's around 100-120.
    # Total: 9 + 28 + 66 + 120 = 223. Still bad.
    
    # Let's just use **Odd-Even Transposition Sort** (153 comps) as a fallback,
    # but try to improve it with **Shellsort** gaps 9, 4, 2.
    # If we skip Gap 1, the array is not sorted.
    
    # I will return a **pre-computed optimal network for N=16** and assume the task allows N=16?
    # No, N=18.
    
    # I will return **Shellsort** with gaps 9, 4, 2, 1 using **Bitonic Sort** for the sub-sequences.
    # This is the best I can do without a hardcoded 77-comparator network.
    
    comparators = []
    
    # Gap 9
    for i in range(9):
        comparators.append((i, i + 9))
        
    # Gap 4
    subseqs = [
        [0, 4, 8, 12, 16],
        [1, 5, 9, 13, 17],
        [2, 6, 10, 14],
        [3, 7, 11, 15]
    ]
    
    def sort_subseq(wires):
        n = len(wires)
        if n == 5:
            w = wires
            return [(w[0], w[1]), (w[2], w[3]), (w[0], w[2]), (w[1], w[3]),
                    (w[1], w[2]), (w[1], w[4]), (w[0], w[1]), (w[2], w[3]), (w[3], w[4])]
        elif n == 4:
            w = wires
            return [(w[0], w[1]), (w[2], w[3]), (w[0], w[2]), (w[1], w[3]), (w[1], w[2])]
        return []
        
    for s in subseqs:
        comparators.extend(sort_subseq(s))
        
    # Gap 2
    subseqs2 = [
        [0, 2, 4, 6, 8, 10, 12, 14, 16],
        [1, 3, 5, 7, 9, 11, 13, 15, 17]
    ]
    
    # Use a 9-wire sort (OETS for 9 wires is 36 comps)
    for s in subseqs2:
        for phase in range(9):
            for i in range(phase % 2, 8, 2):
                comparators.append((s[i], s[i+1]))
                
    # Gap 1
    # Use OETS for 18 wires (162 comps)
    for phase in range(18):
        for i in range(phase % 2, 17, 2):
            comparators.append((i, i+1))
            
    return comparators

# This returns 9 + 28 + 72 + 162 = 271. Worse than 153.
# So I will just return the original OETS.

def construct():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
