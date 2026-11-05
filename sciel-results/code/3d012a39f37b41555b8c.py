import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for 120s limit

    # Specific case optimization for the football pool problem
    if q == 3 and n == 6 and radius == 1:
        total = 3 ** 6  # 729
        powers = [3 ** p for p in range(n)]  # [1, 3, 9, 27, 81, 243]
        
        # Precompute neighbors: for each word, its ball of radius 1
        # Ball size = 1 + n*(q-1) = 1 + 6*2 = 13
        neighbors = np.empty((total, 13), dtype=np.int32)
        for i in range(total):
            d = [(i // powers[p]) % q for p in range(n)]
            neighbors[i, 0] = i
            k = 1
            for p in range(n):
                for alt in range(q):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        neighbors[i, k] = new_i
                        k += 1

        # Precompute neighbor masks for fast intersection/union checks
        # neighbor_mask[i, j] = True if j is in the ball of i
        neighbor_mask = np.zeros((total, total), dtype=np.bool_)
        for i in range(total):
            neighbor_mask[i, neighbors[i]] = True

        def idx_to_list(idx):
            return [(idx // powers[p]) % q for p in range(n)]

        def greedy_set_cover():
            """Run one greedy set cover with random tie-breaking using numpy."""
            uncovered = np.ones(total, dtype=np.int8)
            codewords = []
            
            while uncovered.sum() > 0:
                if time.time() > deadline - 30:
                    break
                # counts = sum of uncovered points covered by each candidate
                counts = np.sum(uncovered[neighbors], axis=1)
                max_count = np.max(counts)
                if max_count == 0:
                    break
                # Find all indices with max count
                max_indices = np.where(counts == max_count)[0]
                # Random tie break
                best_c = int(random.choice(max_indices))
                codewords.append(best_c)
                uncovered[neighbors[best_c]] = 0
            
            # Ensure valid
            if uncovered.sum() != 0:
                while uncovered.sum() > 0:
                    c = int(np.random.randint(total))
                    if c not in codewords:
                        codewords.append(c)
                        uncovered[neighbors[c]] = 0
            
            return codewords

        def local_search(code):
            """Try to remove elements from code while maintaining coverage."""
            # Build coverage: for each element u, how many centers in code cover it
            cover_count = np.zeros(total, dtype=np.int32)
            for w in code:
                cover_count[neighbors[w]] += 1
            
            improved = True
            while improved:
                improved = False
                # Try removing each element in reverse order
                for i in range(len(code) - 1, -1, -1):
                    w = code[i]
                    # Check if removing w still covers everything
                    if np.min(cover_count[neighbors[w]]) >= 2:
                        # Remove w
                        cover_count[neighbors[w]] -= 1
                        code.pop(i)
                        improved = True
                        break  # Restart after removal
            
            return code

        def swap_search(code, cover_count, max_iters=1000):
            """Try 1-for-1 swaps to escape local minima."""
            for _ in range(max_iters):
                if time.time() > deadline - 10:
                    break
                if not code:
                    break
                
                # Pick random codeword to remove
                idx_out = random.randint(0, len(code) - 1)
                c_out = code[idx_out]
                
                # Temporarily remove
                cover_count[neighbors[c_out]] -= 1
                
                # Find critical words: points covered ONLY by c_out (now uncovered)
                crit = neighbors[c_out][cover_count[neighbors[c_out]] == 0]
                
                if len(crit) == 0:
                    # No new holes, revert
                    cover_count[neighbors[c_out]] += 1
                    continue
                
                # Find candidates c_in that cover all crit
                valid_in = np.all(neighbor_mask[:, crit], axis=1)
                
                # Exclude existing codewords
                existing_mask = np.zeros(total, dtype=np.bool_)
                existing_mask[code] = True
                valid_in[existing_mask] = False
                valid_in[c_out] = False  # Just in case
                
                candidates = np.where(valid_in)[0]
                if len(candidates) > 0:
                    c_in = int(random.choice(candidates))
                    # Perform swap
                    code[idx_out] = c_in
                    cover_count[neighbors[c_out]] += 1
                    cover_count[neighbors[c_in]] += 1
                    return True
            
            # Revert if we didn't swap
            return False

        best_code = None
        best_len = total  # Upper bound

        # Run multiple random restarts
        while time.time() < deadline:
            code = greedy_set_cover()
            
            # Local search: try to remove redundant codewords
            code = local_search(code)
            
            # Build coverage counts for swap search
            cover_count = np.zeros(total, dtype=np.int32)
            for w in code:
                cover_count[neighbors[w]] += 1
            
            # Swap search to escape local minima
            swap_search(code, cover_count, max_iters=200)
            
            # Final local search after swaps
            code = local_search(code)
            
            if len(code) < best_len:
                best_len = len(code)
                best_code = code.copy()
                
                # Early termination if we hit the known optimum
                if best_len <= 73:
                    pass

        if best_code is None:
            # Fallback if something went wrong
            return [[0] * n for _ in range(total)]
        
        # Convert int indices back to words
        result = []
        for w in best_code:
            word = [0] * n
            temp = w
            for i in range(n - 1, -1, -1):
                word[i] = temp % q
                temp //= q
            result.append(word)
        
        return result
    
    # General case (should not be reached for the specific task)
    N = q ** n
    ball_size = 1 + n * (q - 1)
    
    balls = np.zeros((N, ball_size), dtype=np.int32)
    
    for idx in range(N):
        w = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            w[i] = temp % q
            temp //= q
        
        balls[idx, 0] = idx
        pos = 1
        for i in range(n):
            for delta in range(1, q):
                new_w = w.copy()
                new_w[i] = (new_w[i] + delta) % q
                new_idx = 0
                for j in range(n):
                    new_idx = new_idx * q + new_w[j]
                balls[idx, pos] = new_idx
                pos += 1
    
    best_code = None
    best_len = N
    
    def greedy_set_cover():
        uncovered = np.ones(N, dtype=np.int8)
        codewords = []
        
        while uncovered.sum() > 0:
            if time.time() > deadline - 30:
                break
            counts = np.sum(uncovered[balls], axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            max_indices = np.where(counts == max_count)[0]
            best_c = int(random.choice(max_indices))
            codewords.append(best_c)
            uncovered[balls[best_c]] = 0
        
        if uncovered.sum() != 0:
            while uncovered.sum() > 0:
                c = int(np.random.randint(N))
                if c not in codewords:
                    codewords.append(c)
                    uncovered[balls[c]] = 0
        
        return codewords

    def local_search(code):
        cover_count = np.zeros(N, dtype=np.int32)
        for w in code:
            cover_count[balls[w]] += 1
        
        improved = True
        while improved:
            improved = False
            for i in range(len(code) - 1, -1, -1):
                w = code[i]
                if np.min(cover_count[balls[w]]) >= 2:
                    cover_count[balls[w]] -= 1
                    code.pop(i)
                    improved = True
                    break
        
        return code

    while time.time() < deadline:
        code = greedy_set_cover()
        code = local_search(code)
        
        if len(code) < best_len:
            best_len = len(code)
            best_code = code.copy()
            if best_len <= 73:
                pass

    if best_code is None:
        return [[0] * n for _ in range(N)]
        
    result = []
    for w in best_code:
        word = [0] * n
        temp = w
        for i in range(n - 1, -1, -1):
            word[i] = temp % q
            temp //= q
        result.append(word)
    
    return result
