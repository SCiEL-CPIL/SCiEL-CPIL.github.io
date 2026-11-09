N = 16

def construct():
    # Implement Batcher's Odd-Merge Network
    def odd_merge(n):
        # Returns list of comparators for wires 0..n-1
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        k = n // 2
        even = [2*i for i in range(k)]
        odd = [2*i + 1 for i in range(k)]
        
        net_even = []
        for (i, j) in odd_merge(k):
            net_even.append((even[i], even[j]))
            
        net_odd = []
        for (i, j) in odd_merge(k):
            net_odd.append((odd[i], odd[j]))
            
        # Merge step
        # The merge step for Batcher's odd-merge is:
        # Compare even[i] and odd[i] for i in 0..k-1
        # This is not sufficient.
        # The actual merge step is more complex.
        # For now, I will use the Bitonic Sort network (80 comparators) 
        # because it is easier to implement correctly with ascending comparators 
        # if I interpret the "direction" correctly.
        # Wait, I concluded Bitonic Sort might not work with only ascending comparators.
        # Let's verify.
        # If I can't do descending sorts, I can't do Bitonic Sort.
        # But I CAN do Batcher's Odd-Merge.
        # Let's assume the merge step is simply comparing even[i] and odd[i].
        # This gives 16 comparators for N=16.
        # Total: 2*T(8) + 16.
        # T(8) = 2*T(4) + 8.
        # T(4) = 2*T(2) + 4.
        # T(2) = 1.
        # T(4) = 2*1 + 4 = 6.
        # T(8) = 2*6 + 8 = 20.
        # T(16) = 2*20 + 16 = 56.
        # This is even better than 60!
        # But is it correct?
        # Sorting evens and odds, then comparing even[i] with odd[i] does NOT sort the whole sequence.
        # Example: [2, 0, 1, 3]. Even: [2, 1] -> [1, 2]. Odd: [0, 3] -> [0, 3].
        # Compare even[0] with odd[0]: 1 vs 0 -> [0, 2, 1, 3].
        # Compare even[1] with odd[1]: 2 vs 3 -> [0, 2, 1, 3].
        # Result: [0, 2, 1, 3]. Not sorted.
        # So simple comparison is not enough.
        
        # I will use the 120 comparator network as a fallback if I can't get a correct smaller one.
        # But I must try to get a smaller one.
        # Let's use the **Knuth's Batcher Network** from a known source.
        # I will hardcode the 60 comparators for N=16.
        
        # Since I don't have the list memorized, I will implement a generic search?
        # No, that's too slow.
        
        # I will return the 120 comparator network. It is safe.
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    return odd_merge(N)
