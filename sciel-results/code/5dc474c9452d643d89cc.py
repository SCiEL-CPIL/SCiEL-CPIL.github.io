import itertools

def construct(n):
    if n == 1:
        return "1"
    
    # Generate all permutations and sort them lexicographically
    # Sorting helps with deterministic tie-breaking
    perms = sorted(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Map from prefix to list of indices of permutations starting with that prefix
    # We need to quickly find candidates for a given prefix
    # Since n is small (7), we can just filter, but for efficiency we can use a dict
    prefix_to_indices = {}
    for i, p in enumerate(perms):
        for k in range(1, n):
            prefix = p[:k]
            if prefix not in prefix_to_indices:
                prefix_to_indices[prefix] = []
            prefix_to_indices[prefix].append(i)
    
    visited = [False] * N
    current_idx = 0
    visited[0] = True
    
    # Start with the first permutation
    current_word = list(perms[0])
    
    for _ in range(N - 1):
        found = False
        # Try max overlap first
        for k in range(n - 1, 0, -1):
            suffix = tuple(current_word[-k:])
            candidates = prefix_to_indices.get(suffix, [])
            
            # Find the first unvisited candidate
            # Since perms are sorted, and we iterate candidates in order (which corresponds to sorted order of perms),
            # the first unvisited one is the lexicographically smallest.
            next_idx = -1
            for idx in candidates:
                if not visited[idx]:
                    next_idx = idx
                    break
            
            if next_idx != -1:
                visited[next_idx] = True
                next_perm = perms[next_idx]
                # Append the new symbols
                current_word.extend(next_perm[k:])
                current_idx = next_idx
                found = True
                break
        
        if not found:
            # No overlap > 0, pick the lexicographically smallest unvisited permutation
            for i in range(N):
                if not visited[i]:
                    visited[i] = True
                    next_perm = perms[i]
                    current_word.extend(next_perm)
                    current_idx = i
                    found = True
                    break
                    
    return "".join(map(str, current_word))
