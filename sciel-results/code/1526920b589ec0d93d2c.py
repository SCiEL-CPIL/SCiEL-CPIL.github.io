def _try_swap(S):
    # Try to swap y in S with x not in S to get a new maximal set of same size
    # This might help escape local minima for removals
    S_set = set(S)
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            sums.add(S[i] + S[j])
            
    candidates = []
    for x in range(1, N + 1):
        if x in S_set:
            continue
        # Check if S U {x} is Sidon? No, we are swapping.
        # We want S' = S \ {y} U {x} to be Sidon and Maximal.
        # This is expensive.
        pass
    return None
