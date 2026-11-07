"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Based on analysis, the optimal structure is S = L U H
    # where H = {K, ..., 128} and L is a sum-free subset of {1, ..., K-1}
    # such that max(L) + second_max(L) < K to ensure no sums of two L elements fall in H.
    
    # For K=64:
    # H = {64, ..., 128}, size 65.
    # L must satisfy x+y < 64 for all distinct x,y in L.
    # Largest sum-free subset of {1, ..., 31} is {16, ..., 31}, size 16.
    # Max sum 30+31=61 < 64.
    # Total size 81.
    
    L = list(range(16, 32))
    H = list(range(64, 129))
    S = L + H
    
    # Verify validity to be safe
    if is_valid(S):
        return S
    
    # Fallback to known valid set if verification fails (should not happen)
    return list(range(63, 129))

def is_valid(S):
    """Check if set S has no violating triples."""
    S_set = set(S)
    S_list = sorted(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i+1, n):
            b = S_list[j]
            ab = a + b
            if ab not in S_set:
                continue
            for k in range(j+1, n):
                c = S_list[k]
                ac = a + c
                bc = b + c
                if ac in S_set and bc in S_set:
                    return False
    return True

# EVOLVE-BLOCK-END
