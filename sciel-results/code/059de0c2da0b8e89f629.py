import itertools
import random
import time

def construct(n):
    if n <= 1:
        return "1"
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Map permutation to index
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Precompute prefix maps for fast lookup of overlapping permutations
    # prefix_map[k] maps a tuple of length k to a list of permutation indices starting with that tuple
    prefix_map = [dict() for _ in range(n)]
    for i, p in enumerate(perms):
        for k in range(1, n):
            prefix = p[:k]
            if prefix not in prefix_map[k]:
                prefix_map[k][prefix] = []
            prefix_map[k][prefix].append(i)
    
    best_len = float('inf')
    best_word = None
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Greedy Heuristic with Random Restarts
    # We try multiple random starting points.
    # To improve quality, we can use a "weighted" greedy or just simple nearest neighbor.
    # Given time constraints, simple nearest neighbor with random restarts is robust.
    
    num_restarts = 200
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        visited = [False] * N
        word = []
        
        # Pick a random start
        start_idx = random.randint(0, N - 1)
        current = start_idx
        visited[current] = True
        word.extend(perms[current])
        unvisited = N - 1
        
        while unvisited > 0:
            # Find the best next permutation
            # Check overlaps from n-1 down to 0
            next_idx = -1
            best_k = -1
            
            u = perms[current]
            
            for k in range(n - 1, -1, -1):
                if k == 0:
                    # No overlap, any unvisited works
                    # Find first unvisited
                    for v in range(N):
                        if not visited[v]:
                            next_idx = v
                            best_k = 0
                            break
                    if next_idx != -1:
                        break
                else:
                    suffix = u[n - k:]
                    candidates = prefix_map[k].get(suffix, [])
                    for v in candidates:
                        if not visited[v]:
                            next_idx = v
                            best_k = k
                            break
                    if next_idx != -1:
                        break
            
            if next_idx == -1:
                # Should not happen if unvisited > 0
                break
                
            # Append the new characters
            # Overlap is best_k
            # We need to append perms[next_idx][best_k:]
            current = next_idx
            visited[current] = True
            unvisited -= 1
            word.extend(perms[current][best_k:])
            
        if len(word) < best_len:
            best_len = len(word)
            best_word = word.copy()
            
    # If no valid word was found (should not happen), return a fallback
    if best_word is None:
        # Fallback: just concatenate all permutations (very long, but valid)
        # This is O(n * n!) which is 7 * 5040 = 35280 chars.
        # The greedy should find ~6000.
        fallback = []
        for p in perms:
            fallback.extend(p)
        return "".join(map(str, fallback))
        
    return "".join(map(str, best_word))
