import itertools
import time

def construct(n):
    if n == 1:
        return "1"
    
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
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
    visited[0] = True
    
    # Start with the first permutation
    current_word = list(perms[0])
    current_idx = 0
    
    def get_candidates_for_suffix(suffix_tuple):
        """Get unvisited candidates for a given suffix tuple, sorted by index."""
        return [idx for idx in prefix_to_indices.get(suffix_tuple, []) if not visited[idx]]
    
    def lookahead_score(next_idx, current_word_len, overlap_k):
        """
        Evaluate how good a choice is by looking at how many unvisited permutations
        can be reached with good overlap from the resulting state.
        Higher is better.
        """
        next_perm = perms[next_idx]
        new_suffix = tuple(next_perm[n - 1:])  # last symbol
        # Count unvisited perms that start with the last (n-1) symbols
        last_n_minus_1 = tuple(next_perm[n - 1:]) if n >= 2 else ()
        
        # Primary: how many unvisited perms share prefix with the new suffix of length n-1
        suffix_n1 = tuple(next_perm[-(n-1):]) if n > 1 else ()
        future_candidates = prefix_to_indices.get(suffix_n1, [])
        unvisited_future = sum(1 for idx in future_candidates if not visited[idx] and idx != next_idx)
        
        # Secondary: total unvisited perms sharing any prefix of length n-2
        if n >= 3:
            suffix_n2 = tuple(next_perm[-(n-2):])
            future_candidates2 = prefix_to_indices.get(suffix_n2, [])
            unvisited_future2 = sum(1 for idx in future_candidates2 if not visited[idx] and idx != next_idx)
        else:
            unvisited_future2 = 0
        
        # We want to maximize future options
        return (unvisited_future, unvisited_future2)
    
    for step in range(N - 1):
        if time.time() > deadline:
            break
        
        found = False
        best_choice = None
        best_score = None
        
        # Try max overlap first
        for k in range(n - 1, 0, -1):
            suffix = tuple(current_word[-k:])
            candidates = get_candidates_for_suffix(suffix)
            
            if not candidates:
                continue
            
            # For each candidate, evaluate with lookahead
            for idx in candidates:
                score = lookahead_score(idx, len(current_word), k)
                if best_score is None or score > best_score:
                    best_score = score
                    best_choice = (idx, k)
            
            if best_choice is not None:
                break
        
        if best_choice is None:
            # No overlap > 0, pick the lexicographically smallest unvisited permutation
            for i in range(N):
                if not visited[i]:
                    best_choice = (i, 0)
                    break
        
        if best_choice is None:
            break
        
        next_idx, overlap_k = best_choice
        visited[next_idx] = True
        next_perm = perms[next_idx]
        
        if overlap_k > 0:
            current_word.extend(next_perm[overlap_k:])
        else:
            current_word.extend(next_perm)
        
        current_idx = next_idx
        found = True
    
    return "".join(map(str, current_word))
