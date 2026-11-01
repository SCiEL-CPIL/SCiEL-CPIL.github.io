import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0

    def classical_superpermutation(n):
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
                max_overlap = 0
                limit = min(len(new), len(block))
                for o in range(limit, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        max_overlap = o
                        break
                new += block[max_overlap:]
            word = new
        return "".join(map(str, word))

    if n <= 3:
        return classical_superpermutation(n)

    all_perms = list(permutations(range(1, n + 1)))
    N = len(all_perms)
    
    prefix_map = {}
    for i, p in enumerate(all_perms):
        pref = tuple(p[:n-1])
        if pref not in prefix_map:
            prefix_map[pref] = set()
        prefix_map[pref].add(i)
    
    best_word = classical_superpermutation(n)
    best_len = len(best_word)
    
    while time.time() < deadline:
        unused = set(range(N))
        start_idx = random.choice(list(unused))
        unused.remove(start_idx)
        
        curr_word = list(all_perms[start_idx])
        curr_suffix = tuple(curr_word[-(n-1):])
        
        current_len = n
        path = [start_idx]
        
        while unused:
            candidates = prefix_map.get(curr_suffix, set()) & unused
            
            if candidates:
                next_idx = random.choice(list(candidates))
                unused.remove(next_idx)
                path.append(next_idx)
                current_len += 1
                curr_suffix = tuple(all_perms[next_idx][-(n-1):])
            else:
                if time.time() > deadline - 5.0:
                    break
                
                best_overlap = -1
                best_idx = -1
                
                for idx in unused:
                    p = all_perms[idx]
                    max_k = n - 2
                    for k in range(max_k, 0, -1):
                        if curr_suffix[-k:] == tuple(p[:k]):
                            if k > best_overlap:
                                best_overlap = k
                                best_idx = idx
                            break
                    if best_idx != -1 and best_overlap == n - 2:
                        break # Can't do better than n-2 if n-1 failed
                
                if best_idx == -1:
                    best_idx = random.choice(list(unused))
                    best_overlap = 0
                
                unused.remove(best_idx)
                path.append(best_idx)
                current_len += (n - best_overlap)
                curr_suffix = tuple(all_perms[best_idx][-(n-1):])
        
        if current_len < best_len:
            best_len = current_len
            word_chars = list(all_perms[path[0]])
            for i in range(1, len(path)):
                p_curr = all_perms[path[i]]
                suffix = tuple(word_chars[-(n-1):])
                overlap = 0
                for k in range(n-1, 0, -1):
                    if suffix[-k:] == tuple(p_curr[:k]):
                        overlap = k
                        break
                word_chars.extend(p_curr[overlap:])
            
            best_word = "".join(map(str, word_chars))
            
            if best_len <= 5905:
                break

    return best_word
