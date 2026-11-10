import time

def construct():
    deadline = time.time() + 85
    
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)
        
    best_size = [11] * 101
    best_subset = [None] * 101
    
    # Precompute all subsets' sums and sizes to avoid repeated work if we needed it,
    # but 1023 is small enough to just iterate.
    
    for mask in range(1, 1 << 10):
        size = 0
        s_sum = 0
        # Efficient bit manipulation
        m = mask
        idx = 0
        while m:
            if m & 1:
                size += 1
                s_sum += inv[idx + 1]
            idx += 1
            m >>= 1
            
        residue = s_sum % 101
        
        if size < best_size[residue]:
            best_size[residue] = size
            # Reconstruct subset
            subset = []
            m = mask
            idx = 0
            while m:
                if m & 1:
                    subset.append(idx + 1)
                idx += 1
                m >>= 1
            best_subset[residue] = subset
        
        if time.time() > deadline:
            break

    result = []
    for r in range(101):
        if best_subset[r]:
            result.append(best_subset[r])
        else:
            # Should not happen, but handle gracefully
            # Find any subset for r
            for mask in range(1, 1 << 10):
                s_sum = 0
                m = mask
                idx = 0
                while m:
                    if m & 1:
                        s_sum += inv[idx + 1]
                    idx += 1
                    m >>= 1
                if s_sum % 101 == r:
                    subset = []
                    m = mask
                    idx = 0
                    while m:
                        if m & 1:
                            subset.append(idx + 1)
                        idx += 1
                        m >>= 1
                    result.append(subset)
                    break
            else:
                result.append([1])
                
    return result
