import time
import itertools
import random

def construct(n):
    # For small n, use the standard optimal construction
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
                max_ov = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        max_ov = o
                        break
                new += block[max_ov:]
            word = new
        return "".join(map(str, word))
    
    if n == 7:
        # Optimization for n=7
        start_time = time.time()
        deadline = start_time + 110.0 # 10s buffer
        
        # 1. Generate all permutations of 1..6
        base_perms = list(itertools.permutations(range(1, 7)))
        num_perms = len(base_perms)
        
        # Map perm to index
        perm_to_idx = {p: i for i, p in enumerate(base_perms)}
        
        # 2. Precompute overlap matrix
        # Overlap[i][j] = max k such that suffix of Block_i is prefix of Block_j
        # Block_i = P_i + [7] + P_i
        # Length of block is 2*6 + 1 = 13
        overlap = [[0] * num_perms for _ in range(num_perms)]
        
        # To speed up, we can precompute the blocks as tuples
        blocks = []
        for p in base_perms:
            block = list(p) + [7] + list(p)
            blocks.append(block)
            
        # Calculate overlaps
        for i in range(num_perms):
            bi = blocks[i]
            for j in range(num_perms):
                bj = blocks[j]
                # Max possible overlap is 12
                for k in range(12, -1, -1):
                    if bi[13-k:] == bj[:k]:
                        overlap[i][j] = k
                        break
                        
        # 3. Initialize path
        # Use the standard order from the recursive construction of S(6)
        # We can generate S(6) quickly to get the order
        # Or just use a random/identity start? Standard is better.
        # Let's generate the standard order for n=6 to get the sequence of P's
        word6 = [1]
        for m in range(2, 7):
            perms, seen = [], set()
            for i in range(len(word6) - m + 2):
                window = tuple(word6[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                max_ov = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        max_ov = o
                        break
                new += block[max_ov:]
            word6 = new
            
        # Extract the order of permutations of 1..6 from word6
        # The permutations appear as windows of length 6 in word6?
        # No, the 'perms' extracted in the last step (m=6) were windows of length 5 from S(5).
        # The blocks in S(6) are P5 + [6] + P5.
        # The P5s are the permutations of 1..5.
        # For S(7), we need the sequence of P6s (perms of 1..6) as they appear in S(6).
        # In S(6), the blocks are P5 6 P5.
        # The windows of length 6 in S(6) are the permutations of 1..6.
        # We can just extract unique length-6 windows from word6 in order of appearance.
        
        path = []
        seen_p = set()
        for i in range(len(word6) - 6 + 1):
            w = tuple(word6[i:i+6])
            if w not in seen_p:
                seen_p.add(w)
                path.append(perm_to_idx[w])
                
        # Ensure we have all 720
        # If not, append missing ones (should not happen)
        if len(path) < num_perms:
            missing = set(range(num_perms)) - set(path)
            path.extend(list(missing))
            
        # 4. Local Search (2-opt / Swap) to maximize total overlap
        # Score = sum(overlap[path[i-1]][path[i]])
        
        def calc_score(p):
            s = 0
            for i in range(1, len(p)):
                s += overlap[p[i-1]][p[i]]
            return s

        current_score = calc_score(path)
        best_path = path[:]
        best_score = current_score
        
        # Simple Hill Climbing with 2-opt
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try 2-opt
            for i in range(len(path) - 1):
                if time.time() > deadline: break
                for j in range(i + 1, len(path)):
                    # Reverse segment i..j
                    # Calculate delta score
                    # Remove edges (i-1, i), (j, j+1)
                    # Add edges (i-1, j), (i, j+1)
                    
                    old_score = 0
                    new_score = 0
                    
                    # Edges involved:
                    # Before: ... A - B ... C - D ...
                    # A = path[i-1] (if i>0)
                    # B = path[i]
                    # C = path[j]
                    # D = path[j+1] (if j < len-1)
                    
                    # Delta = -overlap[A][B] - overlap[C][D] + overlap[A][C] + overlap[B][D]
                    
                    ov_ab = overlap[path[i-1]][path[i]] if i > 0 else 0
                    ov_cd = overlap[path[j]][path[j+1]] if j < len(path)-1 else 0
                    
                    new_ov_ac = overlap[path[i-1]][path[j]] if i > 0 else 0
                    new_ov_bd = overlap[path[i]][path[j+1]] if j < len(path)-1 else 0
                    
                    delta = -ov_ab - ov_cd + new_ov_ac + new_ov_bd
                    
                    if delta > 0:
                        # Perform swap
                        path[i:j+1] = path[i:j+1][::-1]
                        current_score += delta
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_path = path[:]
                        
        # Use best_path
        path = best_path
        
        # 5. Reconstruct word
        word = []
        for idx in path:
            block = blocks[idx]
            if not word:
                word.extend(block)
            else:
                ov = overlap[word_idx_prev][idx] # Need to track prev idx
                # Wait, overlap matrix is based on block indices.
                # The overlap in the final string is the same as the block overlap.
                word.extend(block[ov:])
            
        # Correction: The loop above needs to track the previous index to get the overlap.
        word = []
        prev_idx = -1
        for idx in path:
            block = blocks[idx]
            if prev_idx == -1:
                word.extend(block)
            else:
                ov = overlap[prev_idx][idx]
                word.extend(block[ov:])
            prev_idx = idx
            
        return "".join(map(str, word))
        
    else:
        # Fallback for n > 7 (unlikely to be tested, but safe)
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
                max_ov = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        max_ov = o
                        break
                new += block[max_ov:]
            word = new
        return "".join(map(str, word))
