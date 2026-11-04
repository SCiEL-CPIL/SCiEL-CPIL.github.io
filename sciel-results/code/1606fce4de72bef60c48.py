import itertools
import random
import time

def construct(n):
    if n <= 1:
        return "1"
    
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Precompute shift
    total_sum = n * (n + 1) // 2
    shift = [0] * N
    for i in range(N):
        p = perms[i]
        sum_suffix = sum(p[1:])
        missing = total_sum - sum_suffix
        q = p[1:] + (missing,)
        shift[i] = perm_to_idx[q]
        
    # Precompute prefix maps
    # prefix_map[k] is a dict: prefix_tuple -> list of indices
    prefix_maps = [{} for _ in range(n + 1)]
    for k in range(1, n):
        for i, p in enumerate(perms):
            prefix = p[:k]
            if prefix not in prefix_maps[k]:
                prefix_maps[k][prefix] = []
            prefix_maps[k][prefix].append(i)
            
    best_len = float('inf')
    best_word = None
    
    start_time = time.time()
    deadline = start_time + 110
    
    for restart in range(200):
        if time.time() > deadline:
            break
            
        visited = [False] * N
        word = []
        
        start_idx = random.randint(0, N - 1)
        current = start_idx
        visited[current] = True
        word.extend(perms[current])
        unvisited = N - 1
        
        while unvisited > 0:
            next_idx = shift[current]
            
            if not visited[next_idx]:
                current = next_idx
                visited[current] = True
                unvisited -= 1
                word.append(perms[current][n-1])
            else:
                # Jump
                # Try overlaps from n-2 down to 0
                best_v = -1
                best_overlap = -1
                u = perms[current]
                
                found_best = False
                for k in range(n - 2, -1, -1):
                    if k == 0:
                        # Overlap 0: any unvisited v
                        # We can just pick the first unvisited one
                        # But we need to find it.
                        # We can maintain a list of unvisited?
                        # Or just scan.
                        # If we reach k=0, we just need ANY unvisited v.
                        # We can pre-store unvisited indices in a set?
                        # Let's just scan for the first unvisited.
                        for v in range(N):
                            if not visited[v]:
                                best_v = v
                                best_overlap = 0
                                found_best = True
                                break
                        break
                        
                    prefix = u[n-k:]
                    candidates = prefix_maps[k].get(prefix, [])
                    for v in candidates:
                        if not visited[v]:
                            best_v = v
                            best_overlap = k
                            found_best = True
                            break # Found one with this overlap k. Since we iterate k desc, this is best for this k.
                            # But is it the best overall?
                            # Yes, because we are checking k from high to low.
                            # The first unvisited candidate for the highest k is the best.
                            # Wait, if k=n-2 has no unvisited, we go to k=n-3.
                            # So the first found in the outer loop is the global best.
                            
                    if found_best:
                        break
                        
                # Append
                current = best_v
                visited[current] = True
                unvisited -= 1
                # Append v[best_overlap:]
                word.extend(perms[current][best_overlap:])
                
        word_str = "".join(map(str, word))
        if len(word_str) < best_len:
            best_len = len(word_str)
            best_word = word_str
            
    return best_word if best_word else "1"
