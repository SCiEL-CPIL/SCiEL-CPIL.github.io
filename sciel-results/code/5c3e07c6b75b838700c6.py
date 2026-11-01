import itertools
import time
import random

def construct(n):
    if n == 1:
        return "1"
    
    start_time = time.monotonic()
    # Leave 10 seconds buffer for the evaluator
    deadline = start_time + 140.0
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Build prefix map: prefix tuple -> list of indices
    prefix_map = {}
    for i, p in enumerate(perms):
        for k in range(1, n):
            pref = p[:k]
            if pref not in prefix_map:
                prefix_map[pref] = []
            prefix_map[pref].append(i)
            
    # Recursive construction baseline
    def recursive_construct(n):
        word = [1]
        for m in range(2, n + 1):
            perms_prev, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms_prev.append(window)
            new = []
            for p in perms_prev:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
        
    best_word = recursive_construct(n)
    best_len = len(best_word)
    
    # Greedy TSP heuristic
    def run_greedy(start_idx):
        unvisited = set(range(N))
        unvisited.remove(start_idx)
        path = [start_idx]
        current = start_idx
        
        while unvisited:
            p_curr = perms[current]
            found = None
            # Try longest overlap first
            for k in range(n - 1, 0, -1):
                suf = p_curr[n - k:]
                cands = prefix_map.get(suf, [])
                for c in cands:
                    if c in unvisited:
                        found = c
                        break
                if found is not None:
                    break
            
            if found is None:
                # Fallback: pick random
                # To avoid O(N) list creation, we can just pick an arbitrary one
                # but random is better for diversity.
                # Since this is rare, O(N) is acceptable.
                found = random.choice(list(unvisited))
            
            path.append(found)
            unvisited.remove(found)
            current = found
            
        # Calculate length
        total_len = n
        for i in range(len(path) - 1):
            p1 = perms[path[i]]
            p2 = perms[path[i+1]]
            # Find max overlap
            max_ov = 0
            for k in range(n - 1, 0, -1):
                if p1[n-k:] == p2[:k]:
                    max_ov = k
                    break
            total_len += (n - max_ov)
        return total_len, path

    # Run multiple restarts
    while time.monotonic() < deadline:
        start_idx = random.randint(0, N - 1)
        length, path = run_greedy(start_idx)
        if length < best_len:
            best_len = length
            # Reconstruct word from path
            word = list(perms[path[0]])
            for i in range(1, len(path)):
                p_prev = perms[path[i-1]]
                p_curr = perms[path[i]]
                # Find overlap
                ov = 0
                for k in range(n - 1, 0, -1):
                    if p_prev[n-k:] == p_curr[:k]:
                        ov = k
                        break
                word.extend(p_curr[ov:])
            best_word = "".join(map(str, word))
            
    return best_word
