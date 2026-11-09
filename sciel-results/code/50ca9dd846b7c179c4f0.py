         def bitonic_merge(n, ascending=True):
             # n is power of 2
             if n <= 1:
                 return []
             m = n // 2
             # Recursively merge halves
             # The bitonic merge of two sorted sequences (one asc, one desc)
             # involves comparing (i, i+m) for i in 0..m-1?
             # No, it's:
             # 1. Split into two halves of size m.
             # 2. Recursively merge them.
             # 3. Compare (i, i+m) for i in 0..m-1.
             
             # This recursive definition is for the whole sort.
             # Bitonic Sort:
             # 1. Recursively sort first half ascending.
             # 2. Recursively sort second half descending.
             # 3. Bitonic Merge the two halves.
             
             # We need a function `bitonic_sort(n, ascending)`.
             
             def bitonic_sort(n, ascending):
                 if n <= 1:
                     return []
                 m = n // 2
                 # Sort halves
                 left = bitonic_sort(m, ascending)
                 right = bitonic_sort(m, not ascending)
                 
                 # Merge
                 merge = bitonic_merge(m, ascending)
                 
                 # Combine
                 # Left is on wires 0..m-1
                 # Right is on wires m..2m-1
                 # Merge comparators are (i, i+m) for i in 0..m-1?
                 # No, the merge step in Bitonic Sort is:
                 # Compare (i, i+m) for i in 0..m-1.
                 # Wait, that's just one layer.
                 # The recursive merge handles the rest.
                 
                 # Actually, the standard recursive Bitonic Sort is:
                 # 1. Sort left half (asc/desc)
                 # 2. Sort right half (desc/asc)
                 # 3. For i in 0..m-1: compare (i, i+m)
                 
                 # Let's verify this.
                 # If we do that, is it a valid bitonic merge?
                 # Yes, for the top level.
                 # But the recursive calls must produce the correct bitonic sequences.
                 
                 # So:
                 comps = left + right
                 for i in range(m):
                     comps.append((i, i + m))
                 
                 if not ascending:
                     # To make it descending, we can reverse the direction of all comparators?
                     # No, we can't change the direction in the output format.
                     # But we can swap i and j in the comparators?
                     # If we swap, min goes to j (higher index).
                     # This effectively reverses the sort order?
                     # Let's check.
                     # If we swap all (i,j) to (j,i), then min goes to j.
                     # This is equivalent to sorting in descending order?
                     # Yes, if we want max at lower index, we swap.
                     # But the interface requires min at i, max at j.
                     # So we can't swap.
                     
                     # However, we can just generate the ascending network and **not** use it for descending?
                     # No, we need descending for the second half.
                     
                     # Alternative: Generate the network for ascending, and for descending,
                     # we can just reverse the indices?
                     # No.
                     
                     # Let's assume we only generate ascending networks.
                     # For the second half (descending), we can generate an ascending network
                     # on the reversed indices?
                     # This is getting complicated.
                     
                     # Let's just use the **Odd-Even Transposition** seed but apply the **dummy pruning** trick?
                     # No, OE is not a power of 2 network.
                     
                     # Let's just stick to the current seed and improve the search.
                     pass
                 
                 return comps
             
             return bitonic_sort(n, True)
         