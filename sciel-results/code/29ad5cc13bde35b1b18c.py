import itertools

def construct(n):
    if n == 1:
        return "1"
    
    # Generate all permutations and sort them lexicographically
    perms = sorted(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Map from prefix to list of indices of permutations starting with that prefix
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
            
            # Find unvisited candidates
            unvisited_candidates = [idx for idx in candidates if not visited[idx]]
            
            if unvisited_candidates:
                # Heuristic: Prefer candidates that have a "good" next step (look-ahead)
                # Specifically, check if the suffix of the candidate (length n-1)
                # leads to another unvisited permutation with overlap n-1.
                
                best_idx = -1
                best_score = -1
                
                for idx in unvisited_candidates:
                    next_perm = perms[idx]
                    # The suffix of the new word will be next_perm[n-1:]
                    # Check if there is an unvisited perm starting with this suffix
                    next_suffix = next_perm[n-1:]
                    next_candidates = prefix_to_indices.get(next_suffix, [])
                    
                    # Count unvisited next candidates
                    # If there are any, it's a "safe" move
                    has_next = any(not visited[i] for i in next_candidates)
                    
                    # Score: 
                    # 1. Prefer if it has a next move (continuity)
                    # 2. Tie-break: Lexicographical order (index)
                    score = 1 if has_next else 0
                    
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                    elif score == best_score:
                        # Keep the first one (smallest index) due to loop order
                        if best_idx == -1:
                            best_idx = idx
                
                if best_idx != -1:
                    visited[best_idx] = True
                    next_perm = perms[best_idx]
                    current_word.extend(next_perm[k:])
                    current_idx = best_idx
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
