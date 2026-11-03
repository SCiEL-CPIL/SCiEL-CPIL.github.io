import itertools
import time

def construct(n):
    if n == 1:
        return "1"
    
    # Deadline guard: return best valid result before 140s (safety margin for 150s limit)
    start_time = time.time()
    DEADLINE = 140.0  # seconds
    
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
    
    def try_greedy(initial_idx, use_lru, use_lookahead, lru_weight=1.0, la_weight=0.5):
        """
        Greedy construction with optional LRU and lookahead heuristics.
        Returns (word_list, valid, length)
        """
        visited = [False] * N
        visited[initial_idx] = True
        current_word = list(perms[initial_idx])
        
        # LRU state: last_seen time for each symbol
        if use_lru:
            last_seen = {sym: 0 for sym in range(1, n + 1)}
            for i, sym in enumerate(current_word):
                last_seen[sym] = i
            current_time = len(current_word)
        else:
            last_seen = None
            current_time = 0
        
        for _ in range(N - 1):
            if time.time() - start_time > DEADLINE - 5:
                # Check if current word is valid
                if _validate(current_word, n):
                    return current_word, True, len(current_word)
                return None, False, float('inf')
            
            found = False
            best_candidate = -1
            best_score = -float('inf')
            best_k = -1
            best_next_perm = None
            
            # Try max overlap first
            for k in range(n - 1, 0, -1):
                suffix = tuple(current_word[-k:])
                candidates = prefix_to_indices.get(suffix, [])
                
                if not candidates:
                    continue
                
                # Find the best candidate among unvisited ones
                for idx in candidates:
                    if visited[idx]:
                        continue
                    
                    next_perm = perms[idx]
                    score = 0.0
                    
                    # LRU heuristic: prefer candidates whose last symbol was seen least recently
                    if use_lru:
                        append_sym = next_perm[-1]
                        lru_score = current_time - last_seen[append_sym]
                        score += lru_weight * lru_score
                    
                    # Lookahead heuristic: prefer candidates that lead to more options
                    if use_lookahead:
                        next_suffix = next_perm[n-1:]
                        next_candidates = prefix_to_indices.get(next_suffix, [])
                        unvisited_next = sum(1 for i in next_candidates if not visited[i])
                        score += la_weight * unvisited_next
                    
                    # Tie-breaker: lexicographic order (lower index is better)
                    # Add a tiny penalty for higher index
                    score -= 1e-9 * idx
                    
                    if score > best_score:
                        best_score = score
                        best_candidate = idx
                        best_k = k
                        best_next_perm = next_perm
                
                if best_candidate != -1:
                    # We found a candidate with this k (max overlap)
                    visited[best_candidate] = True
                    current_word.extend(best_next_perm[best_k:])
                    
                    # Update LRU state
                    if use_lru:
                        for i, sym in enumerate(best_next_perm[best_k:]):
                            last_seen[sym] = current_time + i
                        current_time += len(best_next_perm) - best_k
                    
                    found = True
                    break
            
            if not found:
                # No overlap > 0, pick the lexicographically smallest unvisited permutation
                for i in range(N):
                    if not visited[i]:
                        visited[i] = True
                        next_perm = perms[i]
                        current_word.extend(next_perm)
                        
                        if use_lru:
                            for i2, sym in enumerate(next_perm):
                                last_seen[sym] = current_time + i2
                            current_time += n
                        
                        found = True
                        break
        
        if found:
            # Validate the complete word
            if _validate(current_word, n):
                return current_word, True, len(current_word)
            else:
                return None, False, float('inf')
        else:
            return None, False, float('inf')
    
    def _validate(word, n):
        """Check if word contains all permutations of 1..n as consecutive windows."""
        if len(word) < n:
            return False
        
        seen = set()
        for i in range(len(word) - n + 1):
            window = tuple(word[i:i+n])
            if len(set(window)) == n:  # All symbols distinct
                seen.add(window)
        
        return len(seen) == N
    
    # Try different heuristic combinations
    best_word = None
    best_length = float('inf')
    
    # Strategy 1: Pure greedy (lexicographic tie-break)
    word, valid, length = try_greedy(0, use_lru=False, use_lookahead=False)
    if valid and length < best_length:
        best_length = length
        best_word = word
    
    # Strategy 2: LRU only
    word, valid, length = try_greedy(0, use_lru=True, use_lookahead=False)
    if valid and length < best_length:
        best_length = length
        best_word = word
    
    # Strategy 3: Lookahead only
    word, valid, length = try_greedy(0, use_lru=False, use_lookahead=True)
    if valid and length < best_length:
        best_length = length
        best_word = word
    
    # Strategy 4: LRU + Lookahead
    word, valid, length = try_greedy(0, use_lru=True, use_lookahead=True)
    if valid and length < best_length:
        best_length = length
        best_word = word
    
    # If no valid word found, fall back to a simple construction
    if best_word is None:
        # Use the classic recursive construction as a fallback
        word = [1]
        for m in range(2, n + 1):
            perms_m, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms_m.append(window)
            new = []
            for p in perms_m:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        best_word = word
        best_length = len(word)
    
    return "".join(map(str, best_word))
