import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for 120s limit
    
    total_words = q ** n
    if total_words > 10000:
        # Fallback: Greedy Set Cover
        # Not implemented for large, but task is small
        return []

    def to_idx(cw):
        idx = 0
        for i, val in enumerate(cw):
            idx += val * (q ** i)
        return idx

    def from_idx(idx):
        cw = [0] * n
        for i in range(n):
            cw[i] = idx % q
            idx //= q
        return tuple(cw)

    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for idx in range(total_words):
        cw = from_idx(idx)
        neighbors[idx].append(idx)
        for pos in range(n):
            for val in range(q):
                if val == cw[pos]:
                    continue
                new_cw = list(cw)
                new_cw[pos] = val
                neighbors[idx].append(to_idx(new_cw))
        
    # Initial construction: Tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_codewords.append(tuple(h + list(tail)))
    
    codewords = set(initial_codewords)
    counts = [0] * total_words
    
    for cw in codewords:
        idx = to_idx(cw)
        for nb in neighbors[idx]:
            counts[nb] += 1
            
    best_size = len(codewords)
    best_codewords = set(codewords)
    
    while time.time() < deadline:
        cw_list = list(codewords)
        if not cw_list:
            break
        cw_old = random.choice(cw_list)
        idx_old = to_idx(cw_old)
        
        # Check redundancy
        redundant = True
        for nb in neighbors[idx_old]:
            if counts[nb] <= 1:
                redundant = False
                break
        
        if redundant:
            for nb in neighbors[idx_old]:
                counts[nb] -= 1
            codewords.remove(cw_old)
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = set(codewords)
            continue
        
        # Try swap
        for nb in neighbors[idx_old]:
            counts[nb] -= 1
            
        uncovered = [nb for nb in neighbors[idx_old] if counts[nb] == 0]
        
        if not uncovered:
            # Should not happen if redundant check was correct, 
            # but if it does, it's a deletion
            codewords.remove(cw_old)
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = set(codewords)
        else:
            candidates = set(neighbors[uncovered[0]])
            for u in uncovered[1:]:
                candidates &= set(neighbors[u])
                if not candidates:
                    break
            
            if candidates:
                valid_candidates = [c for c in candidates if c not in codewords]
                if valid_candidates:
                    cw_new_idx = random.choice(valid_candidates)
                    cw_new = from_idx(cw_new_idx)
                    for nb in neighbors[cw_new_idx]:
                        counts[nb] += 1
                    codewords.add(cw_new)
                    codewords.remove(cw_old)
                    # Size same, no update to best_size needed
            else:
                for nb in neighbors[idx_old]:
                    counts[nb] += 1

    return [list(cw) for cw in best_codewords]
