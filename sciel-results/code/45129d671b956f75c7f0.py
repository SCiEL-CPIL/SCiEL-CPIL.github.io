import itertools
import random
import numpy as np

def construct(n):
    if n < 7:
        return standard_construct(n)
    
    # For n=7, use TSP on permutations of 1..6
    symbols = list(range(1, 7))
    perms = list(itertools.permutations(symbols))
    num_perms = len(perms)
    
    # Precompute overlap matrix using numpy for speed
    # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    overlap = np.zeros((num_perms, num_perms), dtype=np.int32)
    perm_array = np.array(perms)  # shape (num_perms, 6)
    
    for k in range(6, 0, -1):
        # For each k, find pairs where suffix of i matches prefix of j
        suffixes = perm_array[:, -k:]  # shape (num_perms, k)
        prefixes = perm_array[:, :k]   # shape (num_perms, k)
        # For each i, find j where suffixes[i] == prefixes[j]
        # Use broadcasting: suffixes[:, None, :] == prefixes[None, :, :]
        # This creates a (num_perms, num_perms, k) boolean array
        matches = np.all(suffixes[:, None, :] == prefixes[None, :, :], axis=2)
        # For each i, find if any j matches
        # We want to set overlap[i][j] = k for matching pairs
        # But we want the MAX k, so we process from k=6 down to k=1
        # and only set if not already set
        for i in range(num_perms):
            matching_js = np.where(matches[i])[0]
            for j in matching_js:
                if overlap[i][j] == 0:
                    overlap[i][j] = k
    
    best_length = float('inf')
    best_path = None
    
    # Greedy with multiple restarts and improved heuristic
    num_restarts = 200
    for restart in range(num_restarts):
        start = random.randint(0, num_perms - 1)
        visited = np.zeros(num_perms, dtype=bool)
        visited[start] = True
        path = [start]
        total_overlap = 0
        
        current = start
        for step in range(num_perms - 1):
            best_next = -1
            best_score = -1
            # Find unvisited with max overlap, with tie-breaking
            unvisited = np.where(~visited)[0]
            if len(unvisited) == 0:
                break
            ovs = overlap[current][unvisited]
            # Primary: max overlap
            # Secondary: random tie-breaking for diversity
            max_ov = np.max(ovs)
            candidates = unvisited[ovs == max_ov]
            if len(candidates) > 1:
                # Pick randomly among best candidates
                best_next = candidates[random.randint(0, len(candidates) - 1)]
            else:
                best_next = candidates[0]
            best_ov = max_ov
            
            visited[best_next] = True
            path.append(best_next)
            total_overlap += best_ov
            current = best_next
        
        # Calculate length
        length = 13 + (num_perms - 1) * 13 - total_overlap
        
        if length < best_length:
            best_length = length
            best_path = path[:]
            
    # Local search improvement: 2-opt
    if best_path is not None:
        improved = True
        while improved:
            improved = False
            for i in range(len(best_path) - 2):
                for j in range(i + 2, len(best_path)):
                    # Try reversing segment i+1..j
                    new_path = best_path[:i+1] + best_path[i+1:j+1][::-1] + best_path[j+1:]
                    # Calculate overlap for new path
                    new_total = 0
                    for idx in range(1, len(new_path)):
                        new_total += overlap[new_path[idx-1]][new_path[idx]]
                    new_length = 13 + (num_perms - 1) * 13 - new_total
                    if new_length < best_length:
                        best_length = new_length
                        best_path = new_path
                        improved = True
                        break
                if improved:
                    break
    
    # Construct string from best_path
    if best_path is None:
        return standard_construct(n)
        
    # Build the word
    word = []
    # First block
    p = perms[best_path[0]]
    block = list(p) + [7] + list(p)
    word.extend(block)
    
    for idx in range(1, len(best_path)):
        p = perms[best_path[idx]]
        block = list(p) + [7] + list(p)
        ov = int(overlap[best_path[idx-1]][best_path[idx]])
        word.extend(block[ov:])
        
    return "".join(map(str, word))

def standard_construct(n):
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
