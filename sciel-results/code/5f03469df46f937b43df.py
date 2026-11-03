import itertools
import sys
from collections import defaultdict

def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Precompute neighbors
    # Neighbor j of i if perms[i][1:] == perms[j][:-1]
    neighbors = [[] for _ in range(N)]
    for i, p in enumerate(perms):
        suffix = p[1:]
        # The next permutation must start with a symbol such that its first n-1 elements match suffix.
        # Actually, perms[j][:-1] == suffix.
        # So perms[j] is (x, suffix[0], ..., suffix[n-2]) where x is the first element.
        # We can just iterate all perms? No, that's O(N^2).
        # Better: For each permutation p, the possible next permutations are determined by the first symbol of the next perm.
        # Let next_perm = (c, p[1], ..., p[n-1]).
        # This is a valid permutation if c is not in p[1:].
        # So for each p, there are n-1 possible next permutations.
        for c in range(1, n + 1):
            if c in p[1:]:
                continue
            next_p = (c,) + p[1:]
            j = perm_to_idx[next_p]
            neighbors[i].append(j)
            
    # Greedy Hamiltonian Path with Warnsdorff's rule
    visited = [False] * N
    current = 0
    word = list(perms[0])
    visited[0] = True
    count = 1
    
    while count < N:
        # Get unvisited neighbors
        unvisited_neighbors = [j for j in neighbors[current] if not visited[j]]
        
        if not unvisited_neighbors:
            # Stuck. This shouldn't happen often for small n with Warnsdorff, 
            # but if it does, we might need a different start or heuristic.
            # For now, break and return what we have (invalid), or try a simple fallback.
            # To ensure validity, if we get stuck, we should probably fall back to a known valid construction 
            # or just raise an error. But the task requires a valid word.
            # Let's try a different start if stuck? Or just use the classic construction as fallback.
            break
            
        # Warnsdorff: choose neighbor with fewest unvisited neighbors
        def degree(j):
            return sum(1 for k in neighbors[j] if not visited[k])
            
        # Sort by degree
        unvisited_neighbors.sort(key=degree)
        next_node = unvisited_neighbors[0]
        
        visited[next_node] = True
        count += 1
        # Append the last character of the next permutation
        word.append(perms[next_node][-1])
        current = next_node
        
    if count < N:
        # Fallback to classic construction if greedy failed
        # This ensures we return a valid answer, even if not optimal
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
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
        
    return "".join(map(str, word))
