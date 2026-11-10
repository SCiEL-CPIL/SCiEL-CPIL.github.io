import itertools

def construct(n):
    if n <= 1:
        return "1"
        
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Pre-compute prefix maps for lengths 1 to n-1
    # prefix_map[k] is a dict: tuple of length k -> list of indices of perms starting with this prefix
    prefix_maps = [None] * (n)  # index 0 unused, 1..n-1 used
    for k in range(1, n):
        prefix_maps[k] = {}
        for i, p in enumerate(perms):
            prefix = p[:k]
            if prefix not in prefix_maps[k]:
                prefix_maps[k][prefix] = []
            prefix_maps[k][prefix].append(i)
            
    # Track used permutations
    used = [False] * N
    
    # Start with the first permutation
    current_word = list(perms[0])
    used[0] = True
    
    for step in range(1, N):
        found = False
        # Try to find a permutation with maximum overlap
        # Check overlap from n-1 down to 1
        for k in range(n - 1, 0, -1):
            suffix = tuple(current_word[-k:])
            if suffix in prefix_maps[k]:
                candidates = prefix_maps[k][suffix]
                # Find an unused candidate
                for idx in candidates:
                    if not used[idx]:
                        # Use this permutation
                        used[idx] = True
                        # Append the remaining part
                        # The overlap is k, so we append perms[idx][k:]
                        current_word.extend(perms[idx][k:])
                        found = True
                        break
                if found:
                    break
            
        if not found:
            # Fallback: pick any unused permutation
            # Since no positive overlap was found, we append the whole permutation
            for i in range(N):
                if not used[i]:
                    used[i] = True
                    current_word.extend(perms[i])
                    break

    return "".join(map(str, current_word))
