import itertools
import heapq

def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Map from prefix (tuple) to list of indices
    prefix_to_indices = {}
    for i, p in enumerate(perms):
        for k in range(1, n):
            prefix = p[:k]
            if prefix not in prefix_to_indices:
                prefix_to_indices[prefix] = []
            prefix_to_indices[prefix].append(i)
    
    visited = [False] * N
    current_word = list(perms[0])
    visited[0] = True
    
    # Track last seen position for LRU heuristic
    # Initialize with -1 or 0? 
    # Symbols in perms[0] are at 0..n-1
    last_seen = [-1] * (n + 1)
    for i, sym in enumerate(perms[0]):
        last_seen[sym] = i
        
    current_len = n
    
    for _ in range(N - 1):
        found = False
        best_candidate = -1
        best_overlap = 0
        best_lru_score = -1 # We want min last_seen, so we maximize -last_seen or just store min
        best_lru_val = 10**9
        
        # Try max overlap first
        for k in range(n - 1, 0, -1):
            suffix = tuple(current_word[-k:])
            candidates = prefix_to_indices.get(suffix, [])
            
            if not candidates:
                continue
                
            # Check if any unvisited
            unvisited_candidates = []
            for idx in candidates:
                if not visited[idx]:
                    unvisited_candidates.append(idx)
                    
            if not unvisited_candidates:
                continue
            
            # We have at least one candidate with overlap k
            # Evaluate LRU for these candidates
            # The appended symbol is next_perm[k]
            # We want the symbol with the smallest last_seen value
            
            min_lru_val = 10**9
            best_idx_in_group = -1
            # Tie break: if LRU values are same, pick lex smallest perm?
            # Or just first one?
            
            for idx in unvisited_candidates:
                next_perm = perms[idx]
                append_sym = next_perm[k]
                ls = last_seen[append_sym]
                
                if ls < min_lru_val:
                    min_lru_val = ls
                    best_idx_in_group = idx
                elif ls == min_lru_val:
                    # Tie break: lexicographically smallest permutation?
                    # Or maybe we don't care as long as we pick one
                    if perms[idx] < perms[best_idx_in_group]:
                        best_idx_in_group = idx

            if best_idx_in_group != -1:
                # This k is the max overlap found so far? 
                # Since we iterate k from n-1 down to 1, the first k with candidates is the max overlap.
                # So we can break out of the k loop immediately after processing the first valid k.
                best_candidate = best_idx_in_group
                best_overlap = k
                break # Found max overlap k, so no need to check smaller k
            
        if best_candidate != -1:
            visited[best_candidate] = True
            next_perm = perms[best_candidate]
            append_sym = next_perm[best_overlap]
            current_word.extend(next_perm[best_overlap:])
            # Update last_seen for appended symbols
            for i in range(best_overlap, n):
                sym = next_perm[i]
                last_seen[sym] = current_len + i - best_overlap # Wait, indices
                # current_word length increases by n - best_overlap
                # The new symbols are at indices current_len ... current_len + (n - best_overlap) - 1
                # next_perm[best_overlap] is the first new symbol.
                # Its index in current_word will be current_len.
                # next_perm[best_overlap + j] index is current_len + j.
            # Let's just update last_seen for the appended part
            # The appended part is next_perm[best_overlap:]
            # Length of appended part is n - best_overlap
            for j in range(n - best_overlap):
                sym = next_perm[best_overlap + j]
                last_seen[sym] = current_len + j
            current_len += (n - best_overlap)
        else:
            # No overlap > 0
            # Pick LRU among all unvisited?
            # Or lex smallest?
            # Let's pick the one with the min last_seen of its first symbol?
            # Or just lex smallest unvisited.
            best_idx = -1
            min_lru_val = 10**9
            for i in range(N):
                if not visited[i]:
                    # The appended symbol is the whole perm, so the "new" symbol is effectively the first one?
                    # No, if overlap is 0, we append the whole perm.
                    # The "last seen" of the first symbol of the perm is what matters for the "newness"?
                    # Actually, if overlap is 0, we are appending a disjoint block.
                    # Let's just pick lex smallest unvisited to be safe/deterministic.
                    if best_idx == -1 or perms[i] < perms[best_idx]:
                        best_idx = i
            visited[best_idx] = True
            next_perm = perms[best_idx]
            current_word.extend(next_perm)
            for j in range(n):
                sym = next_perm[j]
                last_seen[sym] = current_len + j
            current_len += n

    return "".join(map(str, current_word))
