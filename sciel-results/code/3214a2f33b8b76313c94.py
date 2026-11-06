def construct():
    N = 16
    
    def akers_merge(n, offset=0, stride=1):
        """
        Generate Akers' merge comparators for two sorted lists of size n.
        The input wires are located at:
        List 1: offset + 0, offset + stride, ..., offset + (n-1)*stride
        List 2: offset + n*stride, offset + (n+1)*stride, ..., offset + (2n-1)*stride
        """
        if n == 1:
            # Compare the two single elements
            return [(offset, offset + stride)]
        
        half = n // 2
        
        comparators = []
        
        # 1. Merge even-indexed elements
        # Logical even indices: 0, 2, 4, ...
        # Physical offsets for List 1 evens: 0, 2, ... relative to offset
        # Physical offsets for List 2 evens: n, n+2, ...
        
        # The recursive merge operates on the sub-lists of evens.
        # The "stride" for the recursive call doubles?
        # In the standard formulation, we treat the evens as a contiguous block?
        # No, they are interleaved.
        
        # Let's use the mapping approach.
        # The evens of List 1 are at logical indices 0, 1, ..., half-1 in the "even sub-network".
        # Physical index for logical i in List 1 evens: offset + i * 2 * stride
        # Physical index for logical i in List 2 evens: offset + (n + i * 2) * stride
        
        # We call akers_merge(half, ...) with a new offset and stride?
        # The elements are not contiguous in memory if stride is 1.
        # However, the comparator logic only depends on the relative ordering.
        
        # Let's define a helper that returns comparators (i, j) based on logical indices,
        # then map them to physical indices.
        
        def merge_logical(n):
            if n == 1:
                return [(0, 1)]
            
            half = n // 2
            comps = []
            
            # Merge evens (logical 0, 2, 4...)
            # The evens form two lists of size half.
            # List 1 evens: logical 0, 2, ..., 2*half-2
            # List 2 evens: logical n, n+2, ..., n+2*half-2
            
            # Recursively merge these.
            # We need to map the logical indices of the sub-merge to the current logical indices.
            
            # Sub-merge for evens:
            # Input size 2*half.
            # List 1: 0..half-1 (logical in sub-merge)
            # List 2: half..2*half-1 (logical in sub-merge)
            
            sub_even = merge_logical(half)
            for i, j in sub_even:
                # Map sub-merge logical i to current logical index
                # If i < half: it's from List 1 evens.
                # Current logical index = i * 2
                # If i >= half: it's from List 2 evens.
                # Current logical index = n + (i - half) * 2
                
                if i < half:
                    pi = i * 2
                else:
                    pi = n + (i - half) * 2
                    
                if j < half:
                    pj = j * 2
                else:
                    pj = n + (j - half) * 2
                    
                comps.append((pi, pj))
                
            # Merge odds (logical 1, 3, 5...)
            sub_odd = merge_logical(half)
            for i, j in sub_odd:
                if i < half:
                    pi = i * 2 + 1
                else:
                    pi = n + (i - half) * 2 + 1
                    
                if j < half:
                    pj = j * 2 + 1
                else:
                    pj = n + (j - half) * 2 + 1
                    
                comps.append((pi, pj))
                
            # Final step: Compare (2i+1, 2i+2) for i in 0..n-1
            for i in range(n):
                comps.append((2*i + 1, 2*i + 2))
                
            return comps
        
        # Get logical comparators
        logical_comps = merge_logical(n)
        
        # Map logical indices to physical indices
        # Logical k corresponds to:
        # if k < n: offset + k * stride
        # if k >= n: offset + k * stride
        # Wait, the logical indices in merge_logical are 0..2n-1.
        # List 1 is 0..n-1. List 2 is n..2n-1.
        # Physical index for logical k:
        # If k < n: offset + k * stride
        # If k >= n: offset + k * stride
        
        # This assumes the lists are contiguous in the "logical" space of the merge call,
        # but in the physical space, they might be interleaved or separated.
        
        # In our top-level call:
        # List 1: 0..n-1 (stride 1)
        # List 2: n..2n-1 (stride 1)
        # So physical = logical.
        
        # In recursive calls:
        # We are merging evens.
        # List 1 evens are at physical 0, 2, 4...
        # List 2 evens are at physical n, n+2...
        # The recursive call `merge_logical(half)` returns pairs of logical indices for the sub-merge.
        # We mapped them to `pi, pj` which are the indices in the *current* logical space (0..2n-1).
        # So `pi` and `pj` are in 0..2n-1.
        # We just need to map `pi` and `pj` to physical indices.
        
        physical_comps = []
        for i, j in logical_comps:
            # i, j are in 0..2n-1
            phys_i = offset + i * stride
            phys_j = offset + j * stride
            physical_comps.append((phys_i, phys_j))
            
        return physical_comps

    def akers_sort(n, offset=0, stride=1):
        if n == 1:
            return []
        if n == 2:
            return [(offset, offset + stride)]
            
        half = n // 2
        
        # Sort left half
        left = akers_sort(half, offset, stride)
        
        # Sort right half
        right = akers_sort(half, offset + half * stride, stride)
        
        # Merge
        merge = akers_merge(half, offset, stride)
        
        return left + right + merge

    return akers_sort(N)
