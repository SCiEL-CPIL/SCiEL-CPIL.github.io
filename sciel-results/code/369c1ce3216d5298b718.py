    def batcher_odd_merge(N):
        if N == 2:
            return [(0, 1)]
        
        M = N // 2
        # Recursively sort the two halves
        left = batcher_odd_merge(M)
        right = [(i + M, j + M) for i, j in left]
        
        # Merge phase
        merge = batcher_odd_merge_merge(M)
        
        return left + right + merge

    def batcher_odd_merge_merge(M):
        # Merges two sorted halves of size M (total 2M wires)
        if M == 1:
            return [(0, 1)]
        
        M_half = M // 2
        # Phase 1: Compare (i, i+M) for i in 0..M-1
        # This is not quite right. The standard odd merge merge step is:
        # 1. Compare (0, M), (1, M+1), ...
        # 2. Recursively merge the even-indexed wires and odd-indexed wires?
        
        # Actually, the recursive step for the merge is:
        # The merge of 2M elements consists of:
        # 1. M comparators: (0, M), (1, M+1), ..., (M-1, 2M-1)
        # 2. Two recursive merges of size M? No.
        
        # Let's look at N=4 (M=2).
        # Left: (0,1), Right: (2,3).
        # Merge M=2:
        # Compare (0,2) and (1,3).
        # Then compare (1,2)?
        # If we do (0,2), (1,3), then (1,2), total comparators = 2 (sort) + 2 (sort) + 2 (cross) + 1 (middle) = 7?
        # But C(4) should be 4.
        
        # Ah, Batcher's Odd-Merge for N=4 is just:
        # (0,2), (1,3), (0,1), (2,3)? No.
        
        # Let's re-read the definition of Batcher's Odd-Merge Sort.
        # It is a divide and conquer algorithm.
        # Sort(N):
        #   Sort(N/2) on left
        #   Sort(N/2) on right
        #   Merge(N/2)
        
        # Merge(M) merges two sorted lists of length M.
        # If M=1: Compare (0,1).
        # If M>1:
        #   Compare (0, M), (1, M+1), ..., (M-1, 2M-1).
        #   Recursively Merge(M/2) on the even indices?
        #   Recursively Merge(M/2) on the odd indices?
        
        # Let's test this hypothesis for M=2 (N=4).
        # Merge(2):
        # 1. Compare (0, 2), (1, 3).
        # 2. Recursively Merge(1) on even indices (0, 2)?
        #    The wires involved in the "even" sub-problem are 0 and 2.
        #    But they are not adjacent in the original array?
        #    The recursive call `Merge(M/2)` operates on the sub-wires.
        #    For M=2, M/2=1.
        #    Merge(1) on wires (0, 2) means compare (0, 2)? But we just did that.
        #    Merge(1) on wires (1, 3) means compare (1, 3)? But we just did that.
        
        # This implies the comparator count for Merge(M) is:
        # M (first phase) + 2 * Merge(M/2).
        # Let K(M) be comparators in Merge(M).
        # K(1) = 1.
        # K(2) = 2 + 2*K(1) = 2 + 2 = 4.
        # K(4) = 4 + 2*K(2) = 4 + 8 = 12.
        # K(8) = 8 + 2*K(4) = 8 + 24 = 32.
        
        # Total C(N) = 2*C(N/2) + K(N/2).
        # C(2) = 1.
        # C(4) = 2*C(2) + K(2) = 2*1 + 4 = 6?
        # But we know C(4) is 4.
        
        # There is a discrepancy. Let's check the structure of Batcher's Odd-Merge Sort again.
        # Maybe the "Merge" step doesn't include the recursive sorts?
        # Yes, `Sort(N)` calls `Sort(N/2)` twice and `Merge(N/2)` once.
        # If C(4) = 4, and 2*C(2) = 2, then K(2) must be 2.
        # If K(2) = 2, then the formula K(M) = M + 2*K(M/2) fails for M=2 (2 + 2*1 = 4 != 2).
        
        # Let's look at the actual network for N=4 Batcher Odd-Merge.
        # Wires: 0, 1, 2, 3.
        # Sort Left (0,1): (0,1).
        # Sort Right (2,3): (2,3).
        # Merge:
        #   Compare (0,2) and (1,3).
        #   Compare (1,2).
        # Total: 1+1+2+1 = 5 comparators?
        # But optimal is 4.
        
        # Is Batcher's Odd-Merge Sort for N=4 actually 5 comparators?
        # Let's check a source.
        # "Batcher's odd-even merge sort"
        # N=4:
        # Phase 1: (0,2), (1,3)
        # Phase 2: (0,1), (2,3) ?? No, that's just sorting again.
        
        # Actually, the standard Batcher Odd-Merge Sort for N=4 is:
        # (0,2), (1,3), (0,1), (2,3) is 4 comparators.
        # Does this sort?
        # Input: 3, 2, 1, 0
        # (0,2): 1, 2, 3, 0
        # (1,3): 1, 0, 3, 2
        # (0,1): 0, 1, 3, 2
        # (2,3): 0, 1, 2, 3. Sorted.
        
        # So the Merge step for M=2 (N=4) is:
        # Compare (0, M) i.e. (0,2)
        # Compare (1, M+1) i.e. (1,3)
        # Then what?
        # The recursive merge part?
        # If the total is 4, and sorts took 2, merge took 2.
        # So Merge(2) has 2 comparators.
        # The comparators are (0,2) and (1,3).
        # This implies that for M=2, the recursive merge part is empty or zero cost?
        # Or maybe the recursion is on the "sub-sequences" formed by the odd/even splits.
        
        # Let's look at N=8.
        # C(8) = 16?
        # 2*C(4) + K(4).
        # If C(4)=4, then 2*4 = 8.
        # If K(4) = 8, then C(8) = 16.
        
        # Let's assume the standard Batcher Odd-Merge Sort implementation:
        # `build(N)`:
        #   if N==2: return [(0,1)]
        #   left = build(N/2)
        #   right = build(N/2) shifted
        #   merge = merge_odd(N/2)
        #   return left + right + merge
        
        # `merge_odd(M)`:
        #   if M==1: return [(0,1)]
        #   # Compare (i, i+M) for i in 0..M-1
        #   comps = [(i, i+M) for i in range(M)]
        #   # Recursively merge the even and odd parts
        #   # The even part involves wires 0, 2, 4...
        #   # The odd part involves wires 1, 3, 5...
        #   # But the recursive merge operates on the "logical" positions?
        #   # No, it operates on the physical wires.
        #   # For M=2 (N=4):
        #   # comps = [(0,2), (1,3)]
        #   # Recursively merge M/2 = 1.
        #   # Merge(1) on even wires (0, 2)?
        #   # Merge(1) on odd wires (1, 3)?
        #   # Merge(1) returns [(0,1)] relative to its inputs.
        #   # So for even wires (0,2), it adds (0,2).
        #   # For odd wires (1,3), it adds (1,3).
        #   # Total merge comparators: (0,2), (1,3) + (0,2), (1,3) = 4.
        #   # Total C(4) = 2 (sorts) + 4 (merge) = 6.
        #   # But we know C(4) can be 4.
        
        # It seems my understanding of the recursive merge indices is flawed or Batcher's Odd-Merge is not the most compact for N=4, but it is a valid sorting network.
        # If C(4)=6, then C(8) = 2*6 + K(4).
        # K(4) = 4 (first phase) + 2*K(2).
        # K(2) = 2 (first phase) + 2*K(1).
        # K(1) = 1.
        # K(2) = 2 + 2 = 4.
        # K(4) = 4 + 8 = 12.
        # C(8) = 12 + 12 = 24.
        # C(16) = 2*24 + K(8).
        # K(8) = 8 + 2*K(4) = 8 + 24 = 32.
        # C(16) = 48 + 32 = 80.
        
        # 80 is better than 120, but not 32.
        
        # Is there a better network?
        # The prompt mentions a "verified 60-comparator construction".
        # Maybe I can just hardcode a known efficient network?
        # Or use a simpler heuristic:
        # The "AKS" network is too complex to generate quickly.
        # 
        # Let's try **Bitonic Sort**.
        # Bitonic Sort for N=16:
        # C(N) = 2 C(N/2) + N/2 log2(N)? No.
        # Bitonic Sort comparator count:
        # C(2) = 1
        # C(4) = 2*C(2) + 2 = 4? No, Bitonic merge for 4 is 4 comparators?
        # Bitonic Sort N=4:
        # Sort 0,1 and 2,3 (2 comps).
        # Merge:
        #   (0,2), (1,3) (2 comps).
        #   (0,1), (2,3) (2 comps)?
        #   Total 6?
        #   Actually, Bitonic Sort for N=4 is 6 comparators?
        #   Let's check.
        #   Bitonic merge of 2+2:
        #   (0,2), (1,3).
        #   Then sort the resulting pairs?
        #   The standard Bitonic Sort network for N=4 has 6 comparators.
        #   C(8) = 2*C(4) + 8?
        #   C(16) = 2*C(8) + 16?
        #   If C(4)=6, C(8) = 12+8=20, C(16) = 40+16=56.
        #   
        #   56 is better than 80 and close to 60.
        
        # Let's implement **Bitonic Sort**. It's simpler to get right than Odd-Merge if I'm unsure about the indices.
        
        **Bitonic Sort Construction:**
        `build(N)`:
        -   If $N=2$: `[(0, 1)]`
        -   Else:
            -   `left = build(N/2)`
            -   `right = build(N/2)` shifted
            -   `merge = bitonic_merge(N/2)`
            -   Return `left + right + merge`
            
        `bitonic_merge(M)`: Merges two bitonic sequences of size M?
        No, Bitonic Sort builds a bitonic sequence then sorts it.
        
        Standard recursive definition for Bitonic Sort Network $B(N)$:
        1.  Sort the first half $B(N/2)$.
        2.  Sort the second half $B(N/2)$ in reverse order (or just sort, then flip?).
            Actually, to create a bitonic sequence of size $N$:
            -   Sort first $N/2$ ascending.
            -   Sort second $N/2$ descending.
            -   Merge them.
            
        To get a sorted network, we just need to sort.
        $S(N)$:
        -   If $N=2$: `[(0,1)]`
        -   Else:
            -   $S(N/2)$ on left.
            -   $S(N/2)$ on right.
            -   $M(N/2)$ to merge the two sorted halves into a bitonic sequence? No, into a sorted sequence.
            
        Wait, Bitonic Sort works by:
        1.  Divide into two halves.
        2.  Sort first half ascending.
        3.  Sort second half descending.
        4.  Merge the two bitonic halves.
        
        But if we just want to sort, we can use the same recursive structure.
        
        Let's use the known comparator count for Bitonic Sort:
        $C(N) = 2 C(N/2) + \frac{N}{2} \log_2 N$? No.
        The merge step for Bitonic Sort of size $N$ (merging two bitonic sequences of size $N/2$) uses $\frac{N}{2} \log_2 N$?
        
        Let's just implement it and count.
        
        `def construct():`
        `    return bitonic_sort(16)`
        
        `def bitonic_sort(N):`
        `    if N == 2:`
        `        return [(0, 1)]`
        `    M = N // 2`
        `    left = bitonic_sort(M)`
        `    right = [(i + M, j + M) for i, j in bitonic_sort(M)]`
        `    # To sort, we need to merge two sorted sequences.`
        `    # Bitonic merge assumes input is bitonic.`
        `    # If we have two sorted sequences, they form a bitonic sequence if one is reversed?`
        `    # No, two sorted sequences side-by-side are not necessarily bitonic.`
        `    # Bitonic sequence: first increasing, then decreasing.`
        `    # If Left is sorted asc and Right is sorted asc, the concatenation is not bitonic.`
        `    # We need to sort Right in descending order to make it bitonic?`
        `    # If we sort Right in descending order, we need a network that sorts descending.`
        `    # We can get that by reversing the comparators of the ascending sort?`
        `    # Or just using the same network, the output will be sorted asc. To get desc, we swap the wires?`
        `    # Actually, if we apply the same sorting network to the right half, it will be sorted ascending.`
        `    # Then we have [Asc, Asc]. This is not bitonic.`
        `    # We need [Asc, Desc].`
        `    # So we should sort the right half in descending order.`
        `    # How to sort descending?`
        `    # If we have a network that sorts ascending, we can just reverse the direction of the comparators?`
        `    # No, comparators are min/max. `
        `    # To sort descending, we can just sort ascending and then reverse the wires?`
        `    # Or, we can observe that `bitonic_sort(M)` sorts ascending.`
        `    # If we want descending, we can apply the network and then swap wire i with wire M-1-i?`
        `    # That adds M/2 comparators? No, that's just a permutation, not comparators.`
        `    # But we need the *network* to produce descending order.`
        `    # We can construct a network that sorts descending by taking the ascending network and swapping the roles of min/max?`
        `    # No, the comparator (i,j) writes min to i, max to j.`
        `    # To sort descending, we want max at i, min at j.`
        `    # So we just swap i and j in the comparator pairs?`
        `    # If we swap i and j, min goes to j, max to i. That sorts descending.`
        
        `    right_desc = [(j + M, i + M) for i, j in bitonic_sort(M)]`
        `    # Now we have Left (Asc) and Right (Desc).`
        `    # This forms a Bitonic sequence of size N.`
        `    # Now we need to merge this bitonic sequence into a sorted sequence.`
        `    # The merge step for Bitonic Sort of size N:`
        `    # It recursively splits the bitonic sequence.`
        
        `    merge = bitonic_merge(N)`
        `    return left + right_desc + merge`
        
        `def bitonic_merge(N):`
        `    # Merges a bitonic sequence of size N into a sorted sequence.`
        `    if N == 2:`
        `        return [(0, 1)]`
        `    M = N // 2`
        `    # Phase 1: Compare (0, M), (1, M+1), ..., (M-1, 2M-1)`
        `    comps = [(i, i + M) for i in range(M)]`
        `    # Phase 2: Recursively merge the two halves.`
        `    # The first half (0..M-1) is bitonic? Yes.`
        `    # The second half (M..2M-1) is bitonic? Yes.`
        `    left_merge = bitonic_merge(M)`
        `    right_merge = [(i + M, j + M) for i, j in bitonic_merge(M)]`
        `    return comps + left_merge + right_merge`
        
        Let's check the count for N=4.
        `bitonic_sort(4)`:
        `left = bitonic_sort(2) = [(0,1)]`
        `right_desc = [(j+2, i+2) for (i,j) in [(0,1)]] = [(3, 2)]` -> Comparator (3,2) writes min to 3, max to 2. So wire 2 gets max, wire 3 gets min. This sorts 2,3 in descending order.
        `merge = bitonic_merge(4)`:
        `M=2`.
        `comps = [(0,2), (1,3)]`
        `left_merge = bitonic_merge(2) = [(0,1)]`
        `right_merge = [(2,3)]`
        `merge = [(0,2), (1,3), (0,1), (2,3)]`
        
        Total comparators:
        `left: 1`
        `right_desc: 1`
        `merge: 4`
        `Total: 6`.
        
        For N=8:
        `C(8) = 2*C(4) + Merge(8)`?
        `bitonic_sort(8)`:
        `left = C(4) = 6`
        `right_desc = 6`
        `merge = bitonic_merge(8)`:
        `M=4`.
        `comps = 4`
        `left_merge = bitonic_merge(4) = 4`
        `right_merge = 4`
        `merge = 4 + 4 + 4 = 12`
        `Total C(8) = 6 + 6 + 12 = 24`.
        
        For N=16:
        `C(16) = 2*C(8) + bitonic_merge(16)`
        `bitonic_merge(16)`:
        `M=8`.
        `comps = 8`
        `left_merge = bitonic_merge(8) = 12`
        `right_merge = 12`
        `merge = 8 + 12 + 12 = 32`
        `Total C(16) = 24 + 24 + 32 = 80`.
        
        So Bitonic Sort gives 80 comparators.
        
        Is there a better one?
        The prompt mentions a 60-comparator construction.
        80 is better than 120.
        60 is better than 80.
        
        Can I find a 60-comparator network?
        Maybe **Batcher's Odd-Merge** was actually 32?
        Let's re-evaluate Batcher's Odd-Merge count.
        If C(16)=32, that would be amazing.
        But my calculation gave 80.
        
        Let's check online sources for "optimal sorting network 16 wires".
        Known optimal:
        N=2: 1
        N=4: 4
        N=8: 12 (or 13?)
        N=16: 32 (or 34?)
        
        Wait, if N=16 optimal is 32, then my Batcher calculation was wrong.
        Where did I go wrong?
        C(4)=4.
        C(8)=12.
        C(16)=32.
        
        Let's check the recurrence for Batcher's Odd-Merge again.
        C(N) = 2 C(N/2) + K(N/2).
        If C(4)=4, C(2)=1, then K(2) = 4 - 2(1) = 2.
        If C(8)=12, C(4)=4, then K(4) = 12 - 2(4) = 4.
        If C(16)=32, C(8)=12, then K(8) = 32 - 2(12) = 8.
        
        So K(M) seems to be M/2?
        K(2)=2? No, M=2, K=2.
        K(4)=4? M=4, K=4.
        K(8)=8? M=8, K=8.
        
        So K(M) = M.
        
        If K(M) = M, then the merge step only has M comparators?
        That seems too simple.
        The merge step of Odd-Merge is:
        1. Compare (0, M), (1, M+1)... (M-1, 2M-1). (M comparators).
        2. Recursively merge the even and odd parts.
        
        If the recursive merges add 0 cost, that's impossible.
        
        However, if the "Merge" function in the recurrence `C(N) = 2C(N/2) + K(N/2)` actually *includes* the recursive merges, and the base case is different?
        
        Let's try to implement the network that yields 32 comparators.
        If I can't derive it, I can just use the 80-comparator Bitonic Sort, which is a significant improvement over 120.
        
        Score calculation:
        Seed: 120.
        Bitonic: 80.
        Score = 120 / 80 = 1.5.
        
        This is a solid improvement.
        
        I will implement Bitonic Sort.

