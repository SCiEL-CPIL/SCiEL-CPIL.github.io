import time
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Standard is 33. 
        # Optimization might not help or might be overkill.
        # But let's run the same logic.
        pass

    # Generate standard S_{n-1}
    word = [1]
    for m in range(2, n):
        perms, seen = [], set()
        w_size = m - 1
        for i in range(len(word) - w_size + 1):
            window = tuple(word[i:i + w_size])
            if len(set(window)) == w_size:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    
    # Extract perms of size n-1
    perms, seen = [], set()
    w_size = n - 1
    for i in range(len(word) - w_size + 1):
        window = tuple(word[i:i + w_size])
        if len(set(window)) == w_size:
            if window not in seen:
                seen.add(window)
                perms.append(window)
                
    N = len(perms)
    
    if n >= 5:
        # Precompute overlap matrix
        # W[i][j] = overlap(perms[i], perms[j])
        W = [[0]*N for _ in range(N)]
        for i in range(N):
            pi = perms[i]
            for j in range(N):
                if i == j: continue
                pj = perms[j]
                max_len = min(len(pi), len(pj))
                ov = 0
                for k in range(max_len, -1, -1):
                    if k == 0:
                        ov = 0
                        break
                    if pi[-k:] == pj[:k]:
                        ov = k
                        break
                W[i][j] = ov

        # Initial order
        order = list(range(N))
        
        # Calculate initial total overlap
        total_ov = 0
        for i in range(1, N):
            total_ov += W[order[i-1]][order[i]]
            
        # Cost is proportional to -total_ov. Minimize length = Maximize total_ov.
        # Length = N * (2n-1) - total_ov.
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # 2-Opt: Swap adjacent
            for i in range(N - 1):
                if time.time() > deadline - 10: break
                
                # Indices in order list
                idx_i = order[i]
                idx_ip1 = order[i+1]
                
                # Get neighbors
                # Left neighbor of i
                if i == 0:
                    left = -1
                else:
                    left = order[i-1]
                    
                # Right neighbor of i+1
                if i+1 == N-1:
                    right = -1
                else:
                    right = order[i+2]
                    
                # Old overlap sum for this segment
                # Edges: (left, idx_i), (idx_i, idx_ip1), (idx_ip1, right)
                old_ov = 0
                if left != -1:
                    old_ov += W[left][idx_i]
                old_ov += W[idx_i][idx_ip1]
                if right != -1:
                    old_ov += W[idx_ip1][right]
                
                # New overlap sum
                # Edges: (left, idx_ip1), (idx_ip1, idx_i), (idx_i, right)
                new_ov = 0
                if left != -1:
                    new_ov += W[left][idx_ip1]
                new_ov += W[idx_ip1][idx_i]
                if right != -1:
                    new_ov += W[idx_i][right]
                
                if new_ov > old_ov:
                    # Swap
                    order[i], order[i+1] = order[i+1], order[i]
                    total_ov += (new_ov - old_ov)
                    improved = True
                    break # Restart scan
            
            # Or-Opt: Move single element
            if not improved:
                for i in range(N):
                    if time.time() > deadline - 10: break
                    for j in range(N):
                        if i == j or i == j + 1: continue
                        
                        # Move element at i to position j
                        # Note: If i < j, the element shifts left.
                        # If i > j, the element shifts right.
                        
                        elem = order[i]
                        
                        # Neighbors of i
                        if i == 0:
                            left_i = -1
                        else:
                            left_i = order[i-1]
                            
                        if i == N-1:
                            right_i = -1
                        else:
                            right_i = order[i+1]
                            
                        # Neighbors of j
                        # We are inserting into position j.
                        # If i < j:
                        #   The element at j becomes j+1.
                        #   The element at j-1 is the left neighbor of the insertion point.
                        #   The element at j is the right neighbor of the insertion point.
                        #   BUT, we are removing i first.
                        
                        # Let's just calculate delta by reconstructing the local changes.
                        # It's safer to just compute the full delta.
                        
                        # Create a copy? No, N=720, copy is fast.
                        # But we want O(1) delta.
                        
                        # Let's stick to adjacent swaps if Or-opt is too complex to implement correctly with O(1) delta in short time.
                        # Adjacent swaps are sufficient for hill climbing.
                        # We can also try 3-opt or longer moves if time permits.
                        # For now, adjacent swaps are a good start.
                        pass

    # Construct final word
    final_word = []
    for idx in order:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        max_overlap = min(len(final_word), len(block))
        overlap = 0
        for o in range(max_overlap, -1, -1, ):
            if o == 0:
                overlap = 0
                break
            if final_word[len(final_word) - o:] == block[:o]:
                overlap = o
                break
        final_word += block[overlap:]
        
    return "".join(map(str, final_word))
