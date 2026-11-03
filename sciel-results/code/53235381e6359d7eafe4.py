import itertools
import collections

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
    
    # Heuristic state: track last seen time for each symbol
    # Time is the index in current_word
    last_seen = {sym: 0 for sym in range(1, n + 1)}
    for i, sym in enumerate(current_word):
        last_seen[sym] = i
        
    current_time = len(current_word)
    
    for _ in range(N - 1):
        found = False
        # Try max overlap first
        best_candidate = -1
        best_score = -1
        best_append_sym = -1
        
        for k in range(n - 1, 0, -1):
            suffix = tuple(current_word[-k:])
            candidates = prefix_to_indices.get(suffix, [])
            
            # Find the best candidate among unvisited ones
            # Heuristic: Pick the one with the "least recently used" appended symbol
            # The appended symbol is next_perm[k] (the first symbol not in overlap)
            # Wait, if overlap is k, we append next_perm[k:]
            # The NEW symbols are next_perm[k], ..., next_perm[n-1]
            # The "appended" symbol in the sense of the transition is often considered the last one, or the first new one.
            # Let's use the last symbol of the new permutation as the "identity" for LRU.
            
            for idx in candidates:
                if not visited[idx]:
                    next_perm = perms[idx]
                    # The new symbols are next_perm[k:]
                    # Let's score based on the last symbol next_perm[-1]
                    # Or the first new symbol next_perm[k]
                    # Let's try the last symbol of the new permutation
                    append_sym = next_perm[-1]
                    # Score: time since last_seen[append_sym]
                    # Higher is better (less recently used)
                    score = current_time - last_seen[append_sym]
                    
                    # Tie-breaker: Lexicographic order (lower index is better)
                    # So if scores are equal, we prefer lower idx?
                    # Or we can just pick the first one with max score.
                    
                    if score > best_score:
                        best_score = score
                        best_candidate = idx
                        best_append_sym = append_sym
                        # Store k and next_perm for later
                        # We need to remember which k led to this best candidate
                        # But k is fixed in the outer loop.
                        # We break the k loop if we find a candidate?
                        # No, we want the MAX overlap.
                        # So we iterate k from n-1 down to 1.
                        # The FIRST k that has ANY candidate is the max overlap.
                        # So we only need to search the first k that has candidates.
                        pass
            
            if best_candidate != -1:
                # We found a candidate with this k (max overlap)
                # Apply the choice
                visited[best_candidate] = True
                next_perm = perms[best_candidate]
                current_word.extend(next_perm[k:])
                current_idx = best_candidate
                
                # Update last_seen for all new symbols
                for i, sym in enumerate(next_perm[k:]):
                    last_seen[sym] = current_time + i
                current_time += len(next_perm) - k
                
                found = True
                break
        
        if not found:
            # No overlap > 0, pick the lexicographically smallest unvisited permutation
            # Or apply LRU here too?
            # The directive says "greedily appending...".
            # Let's just pick the first unvisited to ensure termination.
            for i in range(N):
                if not visited[i]:
                    visited[i] = True
                    next_perm = perms[i]
                    current_word.extend(next_perm)
                    current_idx = i
                    for i, sym in enumerate(next_perm):
                        last_seen[sym] = current_time + i
                    current_time += n
                    found = True
                    break
                    
    return "".join(map(str, current_word))
