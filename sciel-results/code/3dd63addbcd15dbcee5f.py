import time
import random
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Standard construction for n <= 6
    if n <= 6:
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
                # Find max overlap
                overlap = 0
                min_len = min(len(new), len(block))
                for o in range(min_len, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    if n == 7:
        # 1. Generate S_6 standard
        word = [1]
        for m in range(2, 7):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = 0
                min_len = min(len(new), len(block))
                for o in range(min_len, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        
        # 2. Extract perms of 1..6 in order of appearance in S_6
        perms, seen = [], set()
        for i in range(len(word) - 5):
            window = tuple(word[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        N = len(perms)
        # N should be 720
        
        # 3. Precompute overlap matrix
        # ov[i][j] = max k such that perms[i][-k:] == perms[j][:k]
        # Max k is 5 (since length is 6, and they are distinct perms, k<6)
        # Actually if i==j, k=6. But we don't visit same node twice in path.
        # However, in the construction, we append P+[7]+P.
        # The end of the previous block is ... P_prev.
        # The start of the next block is P_new ...
        # So we need overlap between P_prev and P_new.
        
        # Convert perms to strings for faster slicing/comparison?
        # Or use numpy?
        # Let's use a list of strings for perms for fast comparison.
        perm_strs = ["".join(map(str, p)) for p in perms]
        
        ov = np.zeros((N, N), dtype=np.int8)
        for i in range(N):
            p_str = perm_strs[i]
            for j in range(N):
                if i == j:
                    ov[i][j] = 6
                    continue
                q_str = perm_strs[j]
                # Check k from 5 down to 0
                # p_str[-k:] == q_str[:k]
                # k=5
                if p_str[-5:] == q_str[:5]:
                    ov[i][j] = 5
                elif p_str[-4:] == q_str[:4]:
                    ov[i][j] = 4
                elif p_str[-3:] == q_str[:3]:
                    ov[i][j] = 3
                elif p_str[-2:] == q_str[:2]:
                    ov[i][j] = 2
                elif p_str[-1:] == q_str[:1]:
                    ov[i][j] = 1
                else:
                    ov[i][j] = 0
                    
        # 4. Local Search
        # Path is a permutation of range(N)
        path = list(range(N))
        
        def calc_total_overlap(path):
            total = 0
            for i in range(len(path) - 1):
                total += ov[path[i]][path[i+1]]
            return total
        
        current_overlap = calc_total_overlap(path)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 2-opt
            for i in range(N - 1):
                if time.time() > deadline - 5: break
                # We can skip some checks or use early termination
                # Reverse path[i+1 .. j]
                # Old: ov[i, i+1] + ov[j, j+1] (if j < N-1)
                # New: ov[i, j] + ov[i+1, j+1] (if j < N-1)
                for j in range(i + 1, N):
                    if j == N - 1:
                        old_ov = ov[path[i]][path[i+1]]
                        new_ov = ov[path[i]][path[j]]
                        if new_ov > old_ov:
                            path[i+1:j+1] = reversed(path[i+1:j+1])
                            current_overlap += new_ov - old_ov
                            improved = True
                            break
                    else:
                        old_ov = ov[path[i]][path[i+1]] + ov[path[j]][path[j+1]]
                        new_ov = ov[path[i]][path[j]] + ov[path[i+1]][path[j+1]]
                        if new_ov > old_ov:
                            path[i+1:j+1] = reversed(path[i+1:j+1])
                            current_overlap += new_ov - old_ov
                            improved = True
                            break
                if improved:
                    break
            
            # Or-Opt (move 1, 2, 3 elements)
            if not improved:
                for length in [1, 2, 3]:
                    if time.time() > deadline - 5: break
                    for i in range(N - length):
                        # Remove block path[i : i+length]
                        # Try inserting at k
                        # We need to be careful with indices.
                        # Let's just try all insertions.
                        block = path[i:i+length]
                        # To avoid O(N^2) list slicing overhead, we can simulate.
                        # But N=720, so 720*720*3 ~ 1.5M ops.
                        # List slicing is fast in C.
                        
                        best_gain = 0
                        best_k = -1
                        
                        # Calculate old cost of removing block
                        # Edges affected: (i-1, i) and (i+length-1, i+length)
                        # If i==0, no left edge. If i+length==N, no right edge.
                        
                        left_node = path[i-1] if i > 0 else None
                        right_node = path[i+length] if i+length < N else None
                        
                        cost_remove = 0
                        if left_node is not None:
                            cost_remove += ov[left_node][path[i]]
                        if right_node is not None:
                            cost_remove += ov[path[i+length-1]][right_node]
                            
                        # Try inserting at k (between k-1 and k, or start/end)
                        # Insertion position k means block goes before path[k]
                        # k ranges from 0 to N-length
                        # If k == i, it's the same position, gain 0.
                        
                        for k in range(N - length + 1):
                            if k == i: continue
                            
                            # Determine neighbors in new position
                            # Block is path[i:i+length]
                            # First of block: b0 = path[i]
                            # Last of block: bl = path[i+length-1]
                            
                            new_left = path[k-1] if k > 0 else None
                            new_right = path[k] if k < N - length else None
                            
                            # If we insert at k, the element at k shifts right.
                            # But we removed the block first.
                            # So the "gap" is between path[k-1] (in reduced list) and path[k] (in reduced list).
                            # This logic is tricky with indices.
                            
                            # Alternative: Just calculate gain by checking specific edges.
                            # This is error-prone.
                            
                            # Let's use a simpler, slightly slower but robust method:
                            # Reconstruct path, calc overlap, compare.
                            # Too slow for 1.5M iterations.
                            
                            # Let's stick to 2-opt if Or-Opt is too complex to optimize quickly.
                            # 2-opt is often sufficient for TSP to get close to optimal.
                            # But Or-Opt is known to improve significantly.
                            
                            pass
                        
        # Construct S_7 using the optimized path
        new_word = list(word) # Start with S_6
        for idx in path:
            p = perms[idx]
            block = list(p) + [7] + list(p)
            overlap = 0
            min_len = min(len(new_word), len(block))
            for o in range(min_len, -1, -1):
                if new_word[len(new_word)-o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
            
        return "".join(map(str, new_word))
    
    # Fallback for n > 7
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
            overlap = 0
            min_len = min(len(new), len(block))
            for o in range(min_len, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
