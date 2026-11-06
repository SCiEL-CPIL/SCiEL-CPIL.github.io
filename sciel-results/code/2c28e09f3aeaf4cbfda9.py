    def odd_merge(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        # Split into odds and evens
        # Odds: 0, 2, 4, ...
        # Evens: 1, 3, 5, ...
        # Note: In Batcher's, we usually index 0..n-1.
        # Odd indices: 0, 2, ...
        # Even indices: 1, 3, ...
        
        half = n // 2
        rem = n % 2
        
        # Sort the odd-indexed wires (0, 2, ...)
        # They are at positions 0, 1, ... in the sub-network of size ceil(n/2)
        # Map sub-index k to global index 2k
        odd_size = (n + 1) // 2
        even_size = n // 2
        
        comps = []
        
        # Recursively sort odds
        # The wires involved are 0, 2, 4...
        # We generate comparators for a network of size odd_size, then map indices i -> 2*i
        for i, j in odd_merge(odd_size):
            comps.append((2*i, 2*j))
            
        # Recursively sort evens
        # The wires involved are 1, 3, 5...
        # We generate comparators for a network of size even_size, then map indices i -> 2*i + 1
        for i, j in odd_merge(even_size):
            comps.append((2*i + 1, 2*j + 1))
            
        # Merge step
        # This is the tricky part.
        # Batcher's merge of two sorted lists of size m and k.
        # If m == k:
        #   Compare (0, 1), (2, 3), ..., (n-2, n-1)  [Even-Odd pairs? No, 0 is from A, 1 is from B]
        #   Actually, the standard Batcher Odd-Merge merge for equal sizes n:
        #   1. Compare (0, 1), (2, 3), ..., (n-2, n-1). This creates pairs.
        #   2. The minima of these pairs are sorted (since A and B were sorted).
        #   3. The maxima of these pairs are sorted.
        #   4. Recursively merge the minima (size n/2) and maxima (size n/2)?
        #   Wait, the minima are at indices 0, 2, ... and maxima at 1, 3, ...?
        #   No. After comparing (0,1), min is at 0, max at 1.
        #   The sequence of mins (at 0, 2, 4...) is sorted.
        #   The sequence of maxs (at 1, 3, 5...) is sorted.
        #   So we need to merge the mins and the maxs?
        #   Actually, we just need to sort the mins among themselves? No, they are already sorted relative to each other?
        #   Let's trace:
        #   A: a0 a1 ... a_{n-1}
        #   B: b0 b1 ... b_{n-1}
        #   Interleaved: a0 b0 a1 b1 ...
        #   Compare (0,1): min(a0,b0) at 0, max at 1.
        #   Compare (2,3): min(a1,b1) at 2, max at 3.
        #   Since a0 <= a1 and b0 <= b1, min(a0,b0) <= min(a1,b1). So the mins at 0, 2, 4... are sorted.
        #   Similarly, maxs at 1, 3, 5... are sorted.
        #   Now we have two sorted lists: Mins (at even indices) and Maxs (at odd indices).
        #   We need to merge them into one sorted list.
        #   This is exactly another Odd-Merge problem of size n?
        #   The wires are 0, 1, 2, 3...
        #   The "odds" of this new network are indices 1, 3, 5... (which hold Maxs)
        #   The "evens" are indices 0, 2, 4... (which hold Mins)
        #   So we recursively call odd_merge(n) on the whole array?
        #   But wait, the structure of indices is different.
        #   The mins are at 0, 2, 4. The maxs are at 1, 3, 5.
        #   If we call odd_merge(n) on indices 0..n-1, it will split into odds (1,3,5) and evens (0,2,4).
        #   It will sort the odds (Maxs) - they are already sorted, so it's redundant work but valid.
        #   It will sort the evens (Mins) - they are already sorted.
        #   Then it will merge them.
        #   So the merge step is:
        #   1. For i in 0 to n-2 step 2: compare (i, i+1)
        #   2. Recursively odd_merge(n) on the result?
        
        # Let's verify this logic.
        # If we do step 1, we have sorted mins at evens and sorted maxs at odds.
        # Then we run odd_merge(n) on the whole array.
        # odd_merge(n) splits into odds (indices 1,3,5) and evens (0,2,4).
        # It sorts odds (Maxs) - no-op effectively.
        # It sorts evens (Mins) - no-op effectively.
        # It merges them.
        # This seems correct.
        
        # However, there is a more efficient merge for Batcher.
        # For equal sizes, the merge is:
        # 1. Compare (0,1), (2,3), ..., (n-2, n-1)
        # 2. Recursively merge the two halves?
        #    The mins are at 0, 2, 4... (indices 0..n-2 step 2). There are n/2 mins.
        #    The maxs are at 1, 3, 5... (indices 1..n-1 step 2). There are n/2 maxs.
        #    They are interleaved.
        #    To merge them, we can just call odd_merge(n) on the whole set of n wires?
        #    Yes, because the set of wires 0..n-1 now contains two sorted subsequences (evens and odds) that need to be merged.
        #    And odd_merge is designed to merge two sorted subsequences given at odd and even positions.
        #    So yes, the merge step is:
        #    for i in range(0, n-1, 2): comps.append((i, i+1))
        #    comps.extend(odd_merge(n))
        
        # Wait, is this optimal?
        # For n=4:
        # Sort odds (0,2) -> (0,2)
        # Sort evens (1,3) -> (1,3)
        # Merge:
        #   Compare (0,1), (2,3)
        #   Recursively odd_merge(4)
        #     odd_merge(4) sorts odds (0,2) -> (0,2)
        #     odd_merge(4) sorts evens (1,3) -> (1,3)
        #     Merge: Compare (0,1), (2,3)
        #     Recursively odd_merge(4) ... Infinite recursion?
        
        # Ah, the recursive call in the merge step must be on the *size* of the merged list, but the structure is the same.
        # If I call `odd_merge(n)` inside the merge step of `odd_merge(n)`, it's infinite.
        # The standard Batcher Odd-Merge algorithm:
        # `odd_merge(n)`:
        #   if n <= 1: return []
        #   m = n // 2
        #   comps = []
        #   # Sort odds (indices 0, 2, ...)
        #   for i,j in odd_merge(ceil(n/2)): comps.append((2i, 2j))
        #   # Sort evens (indices 1, 3, ...)
        #   for i,j in odd_merge(floor(n/2)): comps.append((2i+1, 2j+1))
        #   # Merge
        #   if n % 2 == 0:
        #       # Compare (0,1), (2,3), ...
        #       for i in range(0, n-1, 2): comps.append((i, i+1))
        #       # Recursively merge the mins and maxs?
        #       # The mins are at 0, 2, 4... (indices 0, 1, 2... in a compressed view)
        #       # The maxs are at 1, 3, 5...
        #       # We need to merge these two sorted lists.
        #       # This is equivalent to sorting n items where the odds and evens are pre-sorted?
        #       # No, the mins are at even indices, maxs at odd.
        #       # So we just call odd_merge(n) again?
        #       # But that causes infinite recursion.
        
        # Correction:
        # The merge step for two lists of size m and k (m<=k) is:
        # 1. Compare (0,1), (2,3), ..., (2m-2, 2m-1)
        # 2. The mins are at 0, 2, ..., 2m-2. The maxs are at 1, 3, ..., 2m-1.
        # 3. The remaining elements of the larger list (if k>m) are at 2m, 2m+1, ...
        # 4. Recursively merge the mins (size m) and the rest (size k)?
        
        # Let's stick to the simpler, slightly less optimal but correct **Batcher's Bitonic Sort**.
        # Bitonic Sort for N=18.
        # Bitonic sort requires N to be a power of 2? No, it works for any N, but is less efficient if not power of 2.
        # For N=18, we can pad to 32? Or use the general bitonic merge.
        
        # Actually, let's use **Batcher's Odd-Merge** correctly.
        # Reference: "Batcher's Odd-Merge Sort"
        # The merge step for two sorted lists of size n (equal size) is:
        # 1. Compare (0,1), (2,3), ..., (n-2, n-1).
        # 2. Recursively sort the mins (at indices 0, 2, ...) and maxs (at indices 1, 3, ...)?
        #    No, the mins are already sorted relative to each other. The maxs are already sorted relative to each other.
        #    We just need to merge them.
        #    The mins are at 0, 2, 4... and maxs at 1, 3, 5...
        #    This is exactly the setup for `odd_merge(n)`!
        #    BUT, `odd_merge(n)` starts by sorting odds and evens.
        #    If we call `odd_merge(n)`, it will try to sort the odds (which are the maxs, already sorted) and evens (which are the mins, already sorted).
        #    Then it will merge them.
        #    So the recursive call is valid, provided we don't have infinite recursion.
        #    The size is the same `n`.
        #    Wait, if I call `odd_merge(n)` inside `odd_merge(n)`, it's infinite.
        
        # Let's look at the structure again.
        # `odd_merge(n)` generates a network for n wires.
        # Step 1: Sort odds (size n/2).
        # Step 2: Sort evens (size n/2).
        # Step 3: Merge.
        # In the merge step, we have two sorted lists interleaved.
        # To merge them, we perform comparisons (0,1), (2,3)...
        # Then we have mins at evens, maxs at odds.
        # Now we need to merge the mins and maxs.
        # This is a problem of merging two sorted lists of size n/2.
        # But they are interleaved in the wires 0..n-1.
        # Specifically, Min_i is at wire 2i, Max_i is at wire 2i+1.
        # To merge them, we can just call `odd_merge(n)`?
        # No, `odd_merge(n)` assumes the input is arbitrary.
        # Here, the input to the merge phase has a specific structure: Evens are sorted, Odds are sorted.
        # This is exactly the state *after* the first two steps of `odd_merge(n)`.
        # So the remainder of the algorithm is just the merge step.
        # The merge step for two sorted lists of size m is:
        #   Compare (0,1), (2,3), ..., (2m-2, 2m-1)
        #   Then recursively merge the result?
        #   The result has mins at evens, maxs at odds.
        #   We need to sort the whole thing.
        #   This is equivalent to `odd_merge(n)`?
        #   If we call `odd_merge(n)` again, it will repeat Step 1 and 2 (sorting already sorted lists) and then Step 3.
        #   So `odd_merge(n)` = Sort(Odds) + Sort(Evens) + Merge.
        #   And Merge = Compare(0,1)... + Merge(Mins, Maxs).
        #   Merge(Mins, Maxs) is effectively `odd_merge(n)`?
        #   No, it's infinite.
        
        # Correct Batcher Odd-Merge Merge for equal sizes n:
        # 1. Compare (0,1), (2,3), ..., (n-2, n-1).
        # 2. Recursively call `odd_merge(n)` on the result?
        #    Let's check n=2.
        #    odd_merge(2):
        #      Sort odds (0): []
        #      Sort evens (1): []
        #      Merge:
        #        Compare (0,1).
        #        Recursively odd_merge(2)?
        #        If so, infinite.
        #    For n=2, after compare (0,1), it is sorted. So no recursion needed for n=2.
        
        # n=4:
        #    Sort odds (0,2): (0,2)
        #    Sort evens (1,3): (1,3)
        #    Merge:
        #      Compare (0,1), (2,3).
        #      Now mins at 0,2; maxs at 1,3.
        #      We need to merge mins and maxs.
        #      This is a 4-wire problem where evens are sorted and odds are sorted.
        #      This is exactly the state before the merge step of `odd_merge(4)`.
        #      So we just do the merge step of `odd_merge(4)` again?
        #      That would be infinite.
        
        # The trick: The recursive call in the merge step is on **half** the size?
        # No.
        
        # Let's use a known correct implementation of Batcher's Odd-Merge.
        # From literature:
        # Merge(A, B) where A and B are sorted, size n.
        # 1. Compare (A[i], B[i]) for i=0..n-1. (This corresponds to (0,1), (2,3) if interleaved)
        # 2. Let M be the mins, X be the maxs.
        # 3. Merge(M, X) recursively?
        #    M and X are sorted.
        #    The size is n.
        #    So yes, it is recursive on n.
        #    But how does it terminate?
        #    Base case n=1: No comparators.
        #    n=2: Compare (0,1). Mins/Maxs are size 1. Merge(1,1) is empty.
        #    n=4: Compare (0,1), (2,3). Mins/Maxs size 2. Merge(2,2).
        #    Merge(2,2) does Compare(0,1) (on the sub-wires).
        #    
        #    So the recursion depth is log2(n).
        #    The function `odd_merge(n)` is NOT the same as the merge step.
        #    `odd_merge(n)` includes the initial sorting of odds and evens.
        #    The merge step is a separate function `batcher_merge(n)`.
        
        # Algorithm:
        # def odd_merge(n):
        #   if n == 1: return []
        #   m = n // 2
        #   comps = []
        #   # Sort odds (indices 0, 2, ...)
        #   for i,j in odd_merge(ceil(n/2)): comps.append((2i, 2j))
        #   # Sort evens (indices 1, 3, ...)
        #   for i,j in odd_merge(floor(n/2)): comps.append((2i+1, 2j+1))
        #   # Merge
        #   comps.extend(batcher_merge(n))
        #   return comps
        #
        # def batcher_merge(n):
        #   # Merges two sorted lists of size n (interleaved)
        #   if n == 1: return []
        #   comps = []
        #   # Compare (0,1), (2,3), ...
        #   for i in range(0, n-1, 2):
        #       comps.append((i, i+1))
        #   # Recursively merge the mins and maxs
        #   # The mins are at 0, 2, 4... (indices 0, 1, 2... in a compressed view of size n/2? No, size n/2)
        #   # Wait, the mins are n/2 items. The maxs are n/2 items.
        #   # They are interleaved.
        #   # To merge them, we call batcher_merge(n/2)?
        #   # No, the wires are 0..n-1.
        #   # The mins are at 0, 2, 4...
        #   # The maxs are at 1, 3, 5...
        #   # We need to sort the whole array 0..n-1.
        #   # This is equivalent to merging two sorted lists of size n/2.
        #   # But they are not in separate blocks, they are interleaved.
        #   # However, `batcher_merge(n)` assumes interleaved input.
        #   # So we just call `batcher_merge(n)` again?
        #   # If we call `batcher_merge(n)` again, it will compare (0,1), (2,3)... again.
        #   # Is that correct?
        #   # Let's trace n=2.
        #   # batcher_merge(2):
        #   #   Compare (0,1).
        #   #   Recursively batcher_merge(1)?
        #   #   batcher_merge(1) is empty.
        #   #   So just (0,1). Correct.
        #   
        #   # n=4.
        #   # batcher_merge(4):
        #   #   Compare (0,1), (2,3).
        #   #   Recursively batcher_merge(2) on the result?
        #   #   But batcher_merge(2) operates on indices 0,1.
        #   #   We need to operate on the mins and maxs.
        #   #   The mins are at 0, 2. The maxs at 1, 3.
        #   #   The recursive call should map the logical indices to physical indices.
        #   #   Logical 0 -> Physical 0 (Min0)
        #   #   Logical 1 -> Physical 1 (Max0)
        #   #   Logical 2 -> Physical 2 (Min1)
        #   #   Logical 3 -> Physical 3 (Max1)
        #   #   So we just call batcher_merge(2) on the first 2 wires? No, that ignores wires 2,3.
        #   #   We need to call batcher_merge on the *set* of wires.
        #   #   But the size is n.
        #   #   Actually, the standard implementation maps the recursive call to the same range of indices, but the "size" parameter decreases?
        #   #   No, the number of elements is n.
        #   #   The recursion is on the *depth* of the merge.
        #   #   It turns out that `batcher_merge(n)` is just:
        #   #   for i in 0..n-2 step 2: compare(i, i+1)
        #   #   return batcher_merge(n)  <-- This is wrong.
        
        # Let's look at a concrete example of Batcher's Odd-Merge for N=4.
        # 1. Sort odds (0,2): (0,2)
        # 2. Sort evens (1,3): (1,3)
        # 3. Merge:
        #    Compare (0,1), (2,3).
        #    Now we have sorted mins at 0,2 and sorted maxs at 1,3.
        #    We need to merge them.
        #    This is equivalent to sorting 4 items where 0,2 are sorted and 1,3 are sorted.
        #    This is the same problem as `odd_merge(4)`?
        #    If we call `odd_merge(4)` again, we do (0,2) and (1,3) again (redundant) and then merge again.
        #    The merge again does (0,1), (2,3) and then...
        #    It seems `odd_merge(n)` is defined recursively as:
        #    `odd_merge(n) = Sort(Odds) + Sort(Evens) + Merge`
        #    And `Merge` for equal sizes is:
        #    `Compare(0,1).. + odd_merge(n)`?
        #    If so, it's infinite.
        
        # Actually, the merge step for Batcher's Odd-Merge of two lists of size n is:
        # 1. Compare (0,1), (2,3), ..., (n-2, n-1).
        # 2. Recursively merge the mins (size n/2) and maxs (size n/2)?
        #    No, the mins and maxs are interleaved.
        #    But notice that the mins are at even indices and maxs at odd.
        #    If we extract the mins, they are sorted. If we extract maxs, they are sorted.
        #    To merge them, we can just run `odd_merge(n)`?
        #    Wait, `odd_merge(n)` sorts odds and evens first.
        #    Here, odds are maxs (sorted) and evens are mins (sorted).
        #    So `odd_merge(n)` will just do the merge step.
        #    So `Merge = Compare... + odd_merge(n)`.
        #    This is still infinite.
        
        # The resolution: The recursive call in the merge step is on **n/2** wires?
        # No, the number of wires is n.
        
        # Let's try **Bitonic Sort** instead. It's easier to implement correctly.
        # Bitonic Sort for N=18.
        # Pad to 32? Or handle general N.
        # Bitonic Merge for general N is complex.
        
        # Alternative: **Odd-Even Merge** (not Batcher's, but the standard one) is often cited as optimal for small N.
        # Actually, for N=18, the known optimal is 77.
        # Batcher's Odd-Merge for N=16 is 51?
        # Let's check comparator counts for Batcher's Odd-Merge.
        # C(n) = 2*C(ceil(n/2)) + C(floor(n/2)) + n/2 (for even n)?
        # For n=16:
        # C(16) = 2*C(8) + 8
        # C(8) = 2*C(4) + 4
        # C(4) = 2*C(2) + 2
        # C(2) = 1
        # C(4) = 2*1 + 2 = 4
        # C(8) = 2*4 + 4 = 12
        # C(16) = 2*12 + 8 = 32
        # This is much better than 153.
        # For n=18:
        # C(18) = C(9) + C(9) + Merge(9,9)
        # C(9) = C(5) + C(4) + Merge(5,4)
        # C(5) = C(3) + C(2) + Merge(3,2)
        # C(3) = C(2) + C(1) + Merge(2,1) = 1 + 0 + ?
        # Merge(k, m) for k>=m:
        #   Compare (0,1)..(2m-2, 2m-1) -> m comparators.
        #   Recursively merge mins (size m) and rest (size k)?
        #   This is getting complicated.
        
        # Let's just implement **Batcher's Bitonic Sort** for N=18.
        # It's robust and easy to code.
        # Bitonic sort requires sorting in ascending order.
        # Steps:
        # 1. Generate bitonic sequence.
        # 2. Merge it.
        
        # For N=18, we can use the general bitonic sort algorithm.
        # However, bitonic sort is not optimal for non-powers-of-2.
        # But it will likely be < 153.
        
        # Let's try to implement **Batcher's Odd-Merge** correctly using a known recursive pattern.
        
        