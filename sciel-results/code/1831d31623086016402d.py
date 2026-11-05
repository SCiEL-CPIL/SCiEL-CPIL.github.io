import time
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0
    
    if n <= 1:
        return "1"
    
    # Standard superpermutations for n <= 6 are known and small
    # We can hardcode or compute them
    # For n=7, we use the recursive construction with optimization
    
    # Base cases
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Standard construction for 4
        # S(4) = 33
        # We can generate it
        pass

    # We will build up to n
    # To save time, we can hardcode S_6
    # S_6 length 873
    # But generating S_6 via recursion is fast enough
    
    word = [1]
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
            
        # Extract permutations of length m-1 from current word
        perms = []
        seen = set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
                
        # Define blocks for each permutation
        # Block for p is p + [m] + p
        blocks = []
        for p in perms:
            blocks.append(tuple(p) + (m,) + tuple(p))
            
        # We want to order these blocks to maximize overlap
        # Start with the order from 'perms'
        current_order = list(range(len(perms)))
        
        # 2-Opt improvement
        # Calculate overlap matrix
        K = len(blocks)
        # Overlap[i][j] = max overlap between blocks[i] and blocks[j]
        # We compute this on the fly or precompute
        # K is (m-1)!
        # For m=7, K=720. 720x720 matrix is 518k entries.
        # Overlap calculation is O(L). L is ~13.
        # So 518k * 13 ~ 6.7M ops. Fast.
        
        overlap = [[0]*K for _ in range(K)]
        for i in range(K):
            b1 = blocks[i]
            for j in range(K):
                if i == j: continue
                b2 = blocks[j]
                max_ov = min(len(b1), len(b2))
                ov = 0
                for o in range(max_ov, -1, -1):
                    if b1[len(b1)-o:] == b2[:o]:
                        ov = o
                        break
                overlap[i][j] = ov
        
        # 2-Opt on the order
        # We want to maximize sum of overlaps
        # Score = sum(overlap[order[i]][order[i+1]])
        
        # Initial score
        def get_score(order):
            s = 0
            for i in range(len(order)-1):
                s += overlap[order[i]][order[i+1]]
            return s
            
        current_score = get_score(current_order)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(K-1):
                for j in range(i+1, K):
                    # Reverse segment i..j
                    new_order = current_order[:i] + current_order[i:j+1][::-1] + current_order[j+1:]
                    new_score = get_score(new_order)
                    if new_score > current_score:
                        current_order = new_order
                        current_score = new_score
                        improved = True
                        if time.time() > deadline - 5:
                            break
                if time.time() > deadline - 5:
                    break
                    
        # Construct final word
        new_word = []
        # Add first block
        new_word.extend(blocks[current_order[0]])
        
        for idx in range(1, K):
            b_prev = blocks[current_order[idx-1]]
            b_curr = blocks[current_order[idx]]
            ov = overlap[current_order[idx-1]][current_order[idx]]
            new_word.extend(b_curr[ov:])
            
        word = new_word
        
    return "".join(map(str, word))
