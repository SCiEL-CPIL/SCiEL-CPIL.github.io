import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0 # 115s limit to be safe

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
                    if new[len(new)-o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    if n == 7:
        # Build S_6
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
                    if new[len(new)-o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        
        # Extract perms of 1..6
        perms, seen = [], set()
        for i in range(len(word) - 5):
            window = tuple(word[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        N = len(perms)
        
        # Define blocks
        # The block for perm p is p + [7] + p[1:]? 
        # Let's check the structure of S_7 from S_6.
        # S_7 is formed by inserting 7 into S_6.
        # The standard construction says:
        # For each perm p in S_6 order:
        #   append 7
        #   append p without first element?
        # Actually, the block is just the sequence of characters added.
        # Let's use the explicit block definition that allows overlap calculation.
        # Block i corresponds to perm perms[i].
        # The block content is: perms[i] + [7] + perms[i][1:]
        # Wait, if we just concatenate these with max overlap, do we get a valid superperm?
        # Yes, this is the standard construction logic.
        
        blocks = []
        for p in perms:
            b = list(p) + [7] + list(p[1:])
            blocks.append(b)
        
        # Precompute overlap matrix
        # ov[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
        ov = [[0]*N for _ in range(N)]
        for i in range(N):
            bi = blocks[i]
            for j in range(N):
                if i == j:
                    ov[i][j] = len(bi)
                    continue
                bj = blocks[j]
                best = 0
                max_k = min(len(bi), len(bj))
                # We can limit max_k to n-1 or n? 
                # Overlap can be up to n-1 usually, but can be more?
                # If bi ends with 123456 and bj starts with 123456, overlap is 6.
                # Can it be 7? If bi ends with 1234567 and bj starts with 1234567?
                # Then bj is 1234567... which is a perm of 1..7? No, bj is a block derived from a perm of 1..6.
                # So bj contains 7 exactly once.
                # bi contains 7 exactly once.
                # If overlap is 7, then the last 7 of bi are the first 7 of bj.
                # This implies bi[-7:] == bj[:7].
                # Since bj is a perm of 1..7 (with 7 in pos 6? No, 7 is at index 6 in bj? 
                # bj = p + [7] + p[1:]. p is len 6. So 7 is at index 6.
                # So bj[:7] is p + [7].
                # bi[-7:] is [7] + p[1:] + ...?
                # bi = p + [7] + p[1:].
                # bi[-7:] = [7] + p[1:] + (something)? No, len(bi) = 6+1+5 = 12.
                # bi[-7:] is the last 7 elements.
                # It is possible to have overlap 6 or 7.
                for k in range(max_k, 0, -1):
                    if bi[-k:] == bj[:k]:
                        best = k
                        break
                ov[i][j] = best

        # Local Search to maximize sum of overlaps
        # Initial path: standard order
        path = list(range(N))
        
        def calc_overlap_sum(p):
            return sum(ov[p[i]][p[i+1]] for i in range(N-1))
        
        current_sum = calc_overlap_sum(path)
        best_path = path[:]
        best_sum = current_sum
        
        # 2-Opt Search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            best_gain = 0
            best_i, best_j = -1, -1
            
            # Sample random pairs or iterate all?
            # N=720. N^2/2 = 259,200.
            # Iterating all is fast enough in Python if optimized?
            # 250k iterations * ~10 ops = 2.5M ops. < 0.1s.
            # Let's iterate all.
            
            for i in range(N - 1):
                # Reverse segment i..j
                # Gain = (ov[left][j] + ov[i][right] + sum_bwd(i,j)) - (ov[left][i] + ov[j][right] + sum_fwd(i,j))
                # To make this O(1) per pair, we need prefix sums.
                # But updating prefix sums after a move is O(N).
                # If we do a full pass of 2-opt, we do N^2/2 evaluations.
                # If we accept a move, we update path and restart.
                
                # Let's just do a simple 2-opt scan.
                # To avoid O(N) update cost for every potential move, we can just check the best move in a full scan, then apply it.
                
                pass # Placeholder, we will implement a simpler Or-opt or 1-opt first for speed, then 2-opt.
            
            # Let's use 1-Opt (Relocate) and 2-Opt (Reverse) with full scan for best move.
            # This is O(N^2) per iteration.
            # With N=720, N^2 ~ 500k.
            # If we do 1000 iterations, 500M ops. Might be too slow in Python.
            # Let's use a random subset of moves or a smaller N?
            # Or optimize the inner loop.
            
            # Let's try a limited 2-opt: only reverse segments of length 2, 3, 4?
            # Or just 1-opt (move one block).
            
            # 1-Opt: Move block i to position j.
            # Gain = (ov[left_j][i] + ov[i][right_j] - ov[left_j][right_j]) + (ov[left_i][right_i] - ov[left_i][i] - ov[i][right_i])
            
            # Let's implement a fast 1-opt and 2-opt (length 2) search.
            
            found_move = False
            # 1-Opt
            for i in range(N):
                left_i = path[i-1] if i > 0 else -1
                right_i = path[i+1] if i < N-1 else -1
                item = path[i]
                
                gain_remove = 0
                if left_i != -1 and right_i != -1:
                    gain_remove = ov[left_i][right_i] - ov[left_i][item] - ov[item][right_i]
                elif left_i != -1:
                    gain_remove = -ov[left_i][item]
                elif right_i != -1:
                    gain_remove = -ov[item][right_i]
                
                for j in range(N):
                    if j == i or j == i+1: continue
                    
                    if j < i:
                        left_j = path[j-1] if j > 0 else -1
                        right_j = path[j]
                    else:
                        left_j = path[j]
                        right_j = path[j+1] if j < N-1 else -1
                    
                    gain_insert = 0
                    if left_j != -1 and right_j != -1:
                        gain_insert = ov[left_j][item] + ov[item][right_j] - ov[left_j][right_j]
                    elif left_j != -1:
                        gain_insert = ov[left_j][item]
                    elif right_j != -1:
                        gain_insert = ov[item][right_j]
                    
                    total_gain = gain_remove + gain_insert
                    if total_gain > best_gain:
                        best_gain = total_gain
                        best_i = i
                        best_j = j
            
            if best_gain > 0:
                i, j = best_i, best_j
                item = path[i]
                del path[i]
                if i < j: j -= 1
                path.insert(j, item)
                current_sum += best_gain
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_path = path[:]
                improved = True
                continue
            
            # 2-Opt (Reverse segment)
            # We will check reversals of length 2 and 3 to keep it fast.
            for length in [2, 3]:
                for i in range(N - length + 1):
                    # Segment is path[i : i+length]
                    # Reverse it
                    # Left is path[i-1], Right is path[i+length]
                    left_idx = i-1 if i > 0 else -1
                    right_idx = i+length if i+length < N else -1
                    
                    # Old cost: ov[left][i] + ov[i][i+1]... + ov[i+length-1][right]
                    # New cost: ov[left][i+length-1] + ov[i+length-1][i+length-2]... + ov[i][right]
                    
                    old_cost = 0
                    if left_idx != -1: old_cost += ov[path[left_idx]][path[i]]
                    for k in range(i, i+length-1):
                        old_cost += ov[path[k]][path[k+1]]
                    if right_idx != -1: old_cost += ov[path[i+length-1]][path[right_idx]]
                    
                    new_cost = 0
                    if left_idx != -1: new_cost += ov[path[left_idx]][path[i+length-1]]
                    for k in range(i+length-1, i, -1):
                        new_cost += ov[path[k]][path[k-1]]
                    if right_idx != -1: new_cost += ov[path[i]][path[right_idx]]
                    
                    gain = new_cost - old_cost
                    if gain > best_gain:
                        best_gain = gain
                        best_i = i
                        best_j = i + length - 1
            
            if best_gain > 0:
                i, j = best_i, best_j
                path[i:j+1] = reversed(path[i:j+1])
                current_sum += best_gain
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_path = path[:]
                improved = True
                continue

        # Build final word
        final_word = []
        for idx in best_path:
            block = blocks[idx]
            overlap = 0
            min_len = min(len(final_word), len(block))
            for o in range(min_len, -1, -1):
                if final_word[len(final_word)-o:] == block[:o]:
                    overlap = o
                    break
            final_word += block[overlap:]
        
        return "".join(map(str, final_word))
    
    # Fallback
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
                if new[len(new)-o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
