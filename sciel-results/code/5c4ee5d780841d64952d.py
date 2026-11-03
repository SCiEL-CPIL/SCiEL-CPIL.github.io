import time
import numpy as np
import random

def _construct_ternary_6_1():
    start_time = time.perf_counter()
    deadline = start_time + 115.0  # Safety margin (150s limit, leave 35s)
    
    q = 3
    n = 6
    total = q ** n  # 729
    
    # Precompute balls as lists of indices
    balls = []
    for idx in range(total):
        w = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            w[i] = temp % q
            temp //= q
        ball = [idx]
        for i in range(n):
            old_val = w[i]
            for new_val in range(q):
                if new_val != old_val:
                    power = q ** (n - 1 - i)
                    new_idx = idx + (new_val - old_val) * power
                    ball.append(new_idx)
        balls.append(ball)
    
    # Precompute distance-2 graph: edges between words at Hamming distance 2
    dist2_neighbors = [[] for _ in range(total)]
    for idx in range(total):
        w = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            w[i] = temp % q
            temp //= q
        for i in range(n):
            for j in range(i + 1, n):
                old_i, old_j = w[i], w[j]
                for ni in range(q):
                    if ni == old_i:
                        continue
                    for nj in range(q):
                        if nj == old_j:
                            continue
                        power_i = q ** (n - 1 - i)
                        power_j = q ** (n - 1 - j)
                        new_idx = idx + (ni - old_i) * power_i + (nj - old_j) * power_j
                        if new_idx != idx:
                            dist2_neighbors[idx].append(new_idx)
    
    # Convert to sets for fast membership
    dist2_neighbors_set = [set(nb) for nb in dist2_neighbors]
    
    def _greedy_cover():
        """Greedy set cover to get initial solution."""
        in_code = np.zeros(total, dtype=bool)
        cover_count = np.zeros(total, dtype=np.int32)
        
        while np.any(cover_count == 0):
            if time.perf_counter() > deadline:
                break
            uncovered = np.where(cover_count == 0)[0]
            best_idx = -1
            best_gain = 0
            for i in range(total):
                if in_code[i]:
                    continue
                gain = sum(1 for u in uncovered if u in balls[i])
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_gain == 0:
                break
            in_code[best_idx] = True
            cover_count[balls[best_idx]] += 1
        
        return in_code, cover_count, int(np.sum(in_code))
    
    def _try_remove(in_code, cover_count, size):
        """Try to remove a codeword. Return (new_size, success)."""
        current_codewords = np.where(in_code)[0]
        if len(current_codewords) == 0:
            return size, False
        
        c_to_remove = current_codewords[random.randint(0, len(current_codewords) - 1)]
        
        in_code[c_to_remove] = False
        cover_count[balls[c_to_remove]] -= 1
        
        uncovered_words = np.where(cover_count == 0)[0]
        
        if len(uncovered_words) == 0:
            return size - 1, True
        else:
            # Try to fix by adding a single codeword
            if len(uncovered_words) <= 12:
                best_add = -1
                best_gain = 0
                for w in range(total):
                    if in_code[w]:
                        continue
                    gain = sum(1 for u in uncovered_words if u in balls[w])
                    if gain > best_gain:
                        best_gain = gain
                        best_add = w
                    if best_gain == len(uncovered_words):
                        break
                if best_gain == len(uncovered_words):
                    in_code[best_add] = True
                    cover_count[balls[best_add]] += 1
                    return size, False  # Size unchanged
            # Revert
            in_code[c_to_remove] = True
            cover_count[balls[c_to_remove]] += 1
            return size, False
    
    def _local_search(in_code, cover_count, size):
        """Local search to minimize code size."""
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.perf_counter() < deadline:
            if no_improve_count > max_no_improve:
                break
            
            new_size, success = _try_remove(in_code, cover_count, size)
            if success:
                size = new_size
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        return size
    
    # Run multiple restarts with different strategies
    best_size = total  # Worst case: all words
    best_in_code = None
    best_cover_count = None
    
    # Strategy 1: Pure greedy set cover
    in_code, cover_count, size = _greedy_cover()
    if size < best_size:
        best_size = size
        best_in_code = in_code.copy()
        best_cover_count = cover_count.copy()
    
    # Local search on greedy result
    size = _local_search(in_code, cover_count, size)
    if size < best_size:
        best_size = size
        best_in_code = in_code.copy()
        best_cover_count = cover_count.copy()
    
    # Strategy 2: Greedy with random tie-breaking
    for restart in range(20):
        if time.perf_counter() > deadline - 10:
            break
        
        random.seed(restart * 7 + 13)
        in_code = np.zeros(total, dtype=bool)
        cover_count = np.zeros(total, dtype=np.int32)
        
        while np.any(cover_count == 0):
            if time.perf_counter() > deadline:
                break
            uncovered = np.where(cover_count == 0)[0]
            # Find all candidates with max gain, pick randomly
            best_gain = 0
            candidates = []
            for i in range(total):
                if in_code[i]:
                    continue
                gain = sum(1 for u in uncovered if u in balls[i])
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            if best_gain == 0:
                break
            chosen = random.choice(candidates)
            in_code[chosen] = True
            cover_count[balls[chosen]] += 1
        
        size = int(np.sum(in_code))
        
        # Local search
        size = _local_search(in_code, cover_count, size)
        if size < best_size:
            best_size = size
            best_in_code = in_code.copy()
            best_cover_count = cover_count.copy()
    
    # Strategy 3: Start from independent set on distance-2 graph, then cover
    for restart in range(10):
        if time.perf_counter() > deadline - 5:
            break
        
        random.seed(restart * 31 + 7)
        in_code = np.zeros(total, dtype=bool)
        cover_count = np.zeros(total, dtype=np.int32)
        
        available = set(range(total))
        while available and time.perf_counter() < deadline:
            v = random.choice(list(available))
            in_code[v] = True
            cover_count[balls[v]] += 1
            available.discard(v)
            for nb in dist2_neighbors[v]:
                if nb in available:
                    available.discard(nb)
        
        # Cover remaining
        while np.any(cover_count == 0):
            if time.perf_counter() > deadline:
                break
            uncovered = np.where(cover_count == 0)[0]
            best_idx = -1
            best_gain = 0
            for i in range(total):
                if in_code[i]:
                    continue
                gain = sum(1 for u in uncovered if u in balls[i])
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_gain == 0:
                break
            in_code[best_idx] = True
            cover_count[balls[best_idx]] += 1
        
        size = int(np.sum(in_code))
        size = _local_search(in_code, cover_count, size)
        if size < best_size:
            best_size = size
            best_in_code = in_code.copy()
            best_cover_count = cover_count.copy()
    
    # Final local search on best solution
    if best_in_code is not None and time.perf_counter() < deadline - 5:
        size = _local_search(best_in_code, best_cover_count, best_size)
        best_size = size
    
    # Reconstruct code
    final_code = np.where(best_in_code)[0].tolist()
    
    result = []
    for idx in final_code:
        w = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            w[i] = temp % q
            temp //= q
        result.append(w)
    
    return result

def _fallback(q, n, radius):
    total = q ** n
    if total <= 10000:
        res = []
        for i in range(total):
            w = [0] * n
            t = i
            for k in range(n - 1, -1, -1):
                w[k] = t % q
                t //= q
            res.append(w)
        return res
    return []

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        return _construct_ternary_6_1()
    return _fallback(q, n, radius)
