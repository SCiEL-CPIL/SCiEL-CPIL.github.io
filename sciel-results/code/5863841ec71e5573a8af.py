def construct():
    # Strategy: Split into L (small) and H (large)
    # H = {K, ..., 128}
    # L = sum-free subset of {1, ..., K-1} with max sum < K
    # Optimal K found around 64
    
    K = 64
    H = list(range(K, 129))
    
    # L must be sum-free and max(L) + max(L) < K is sufficient to ensure a+b not in H
    # We pick the largest sum-free subset of {1, ..., floor((K-1)/2)}?
    # Actually, we need a+b < K.
    # Max element m such that 2m < K => m < K/2.
    # So L subset of {1, ..., floor((K-1)/2)}.
    max_L = (K - 1) // 2
    # Largest sum-free set in {1, ..., max_L} is {floor(max_L/2)+1, ..., max_L}
    start_L = max_L // 2 + 1
    L = list(range(start_L, max_L + 1))
    
    S = L + H
    # Convert to list of ints
    return [int(x) for x in S]
