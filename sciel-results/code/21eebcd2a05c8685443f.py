import itertools
from collections import defaultdict

def construct(n):
    if n == 1:
        return "1"
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Build prefix maps
    # prefix_maps[k][prefix] = set of perms starting with prefix
    prefix_maps = [defaultdict(set) for _ in range(n)]
    
    for p in perms:
        for k in range(1, n):
            prefix = p[:k]
            prefix_maps[k][prefix].add(p)
            
    unvisited = set(perms)
    
    # Start with first perm
    current = perms[0]
    unvisited.remove(current)
    for k in range(1, n):
        prefix_maps[k][current[:k]].remove(current)
        
    sequence = [(current, 0)]
    
    # LRU tracking: last time a permutation was selected
    last_used = {p: 0 for p in perms}
    step = 0
    
    for _ in range(N - 1):
        found = False
        
        # Try max overlap first
        for k in range(n - 1, 0, -1):
            suffix = current[-k:]
            candidates = prefix_maps[k].get(suffix)
            if candidates:
                # Pick the least recently used candidate
                next_perm = min(candidates, key=lambda x: last_used[x])
                # Remove from unvisited and maps
                unvisited.remove(next_perm)
                for kk in range(1, n):
                    prefix_maps[kk][next_perm[:kk]].remove(next_perm)
                
                sequence.append((next_perm, k))
                current = next_perm
                last_used[next_perm] = step
                found = True
                break
        
        if not found:
            # No overlap > 0, pick the least recently used unvisited
            next_perm = min(unvisited, key=lambda x: last_used[x])
            unvisited.remove(next_perm)
            for kk in range(1, n):
                prefix_maps[kk][next_perm[:kk]].remove(next_perm)
                
            sequence.append((next_perm, 0))
            current = next_perm
            last_used[next_perm] = step
            
        step += 1
        
    # Construct word
    word = list(sequence[0][0])
    for perm, k in sequence[1:]:
        word.extend(perm[k:])
        
    return "".join(map(str, word))
