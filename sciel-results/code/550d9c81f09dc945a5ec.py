    # Precompute masks by size
    masks_by_size = [[] for _ in range(19)]
    for m in range(1 << 18):
        pc = bin(m).count('1')
        masks_by_size[pc].append(m)
        
    def max_tf_size(adj):
        for k in range(18, 0, -1):
            for mask in masks_by_size[k]:
                if is_tf_mask(mask, adj):
                    return k
        return 0
