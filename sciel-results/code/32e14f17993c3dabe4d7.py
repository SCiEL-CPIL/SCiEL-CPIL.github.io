import itertools
import time

def construct(n):
    if n <= 1:
        return "1"
    
    # Recursive construction for fallback
    def recursive_superperm(n):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    fallback = recursive_superperm(n)
    
    if n < 4:
        return fallback

    # Greedy construction
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    perms = list(itertools.permutations(range(1, n + 1)))
    # Convert to strings for easier slicing? Or keep tuples.
    # Strings are faster for slicing in Python.
    perm_strs = ["".join(map(str, p)) for p in perms]
    
    # Map prefix to list of perm indices
    prefix_map = {}
    for i, p in enumerate(perm_strs):
        for k in range(1, n):
            prefix = p[:k]
            if prefix not in prefix_map:
                prefix_map[prefix] = []
            prefix_map[prefix].append(i)
    
    # Start with first perm
    current_idx = 0
    word = perm_strs[0]
    visited = [False] * len(perm_strs)
    visited[0] = True
    
    for step in range(1, len(perm_strs)):
        if time.time() > deadline:
            return fallback
            
        # Try to find next perm with max overlap
        found = False
        # Check k from n-1 down to 1
        for k in range(n - 1, 0, -1):
            suffix = word[-k:]
            candidates = prefix_map.get(suffix, [])
            for idx in candidates:
                if not visited[idx]:
                    # Append the non-overlapping part
                    word += perm_strs[idx][k:]
                    visited[idx] = True
                    current_idx = idx
                    found = True
                    break
            if found:
                break
        
        if not found:
            # Should not happen, but fallback if stuck
            return fallback
            
    return word
