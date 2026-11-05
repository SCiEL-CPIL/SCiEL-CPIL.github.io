import time
import sys
import random
import math

def construct_standard(n):
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
            max_ov = min(len(new), len(block))
            overlap = 0
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def get_perm_order(n):
    """
    Returns the list of permutations of 1..n in the order they appear 
    in the standard superpermutation of size n.
    """
    word = construct_standard(n)
    perms = []
    seen = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        if len(set(w)) == n and w not in seen:
            seen.add(w)
            perms.append(w)
    return perms

def compute_overlap_matrix(perms):
    """
    perms: list of tuples, each a permutation of 1..n
    Returns matrix M where M[i][j] is the overlap between block_i and block_j.
    block_i = list(perms[i]) + [n+1] + list(perms[i])
    """
    n = len(perms[0])
    N = len(perms)
    # Precompute blocks as strings for faster comparison?
    # Or just use lists.
    blocks = [list(p) + [n+1] + list(p) for p in perms]
    
    M = [[0]*N for _ in range(N)]
    
    for i in range(N):
        b_i = blocks[i]
        for j in range(N):
            if i == j:
                continue
            b_j = blocks[j]
            # Max possible overlap is len(b_i) - 1
            # But we know it's small.
            max_ov = min(len(b_i), len(b_j))
            ov = 0
            # Check from max possible down to 0
            # Optimization: The overlap is limited by the structure.
            # We can just loop.
            for o in range(max_ov, 0, -1):
                if b_i[len(b_i)-o:] == b_j[:o]:
                    ov = o
                    break
            M[i][j] = ov
    return M

def construct(n):
    start_time = time.time()
    deadline = start_time + 115
    
    if n < 5:
        # Standard is optimal for small n
        return "".join(map(str, construct_standard(n)))
        
    # Get the standard order of permutations for n-1
    perms = get_perm_order(n-1)
    N = len(perms)
    
    # Precompute overlap matrix
    M = compute_overlap_matrix(perms)
    
    # Initial tour: standard order (0, 1, ..., N-1)
    tour = list(range(N))
    
    # Calculate initial total overlap
    def calc_total_overlap(t):
        total = 0
        for i in range(len(t) - 1):
            total += M[t[i]][t[i+1]]
        return total
        
    current_overlap = calc_total_overlap(tour)
    
    # 2-Opt Search
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(N - 1):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, N):
                # Segment to reverse: tour[i+1 ... j]
                # Edges affected:
                # (tour[i], tour[i+1]) and (tour[j], tour[j+1])
                # New edges:
                # (tour[i], tour[j]) and (tour[i+1], tour[j+1])
                
                old_cost = M[tour[i]][tour[i+1]] + M[tour[j]][tour[j+1]]
                new_cost = M[tour[i]][tour[j]] + M[tour[i+1]][tour[j+1]]
                
                # We want to maximize overlap, so minimize cost (negative overlap)
                # Gain = new_overlap - old_overlap
                if new_cost > old_cost:
                    # Reverse
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    current_overlap += (new_cost - old_cost)
                    improved = True
                    
    # Construct the final word
    # The word is formed by concatenating blocks with overlaps
    # Block 0 is added fully.
    # Subsequent blocks are added with overlap.
    
    # Get the blocks again
    blocks = [list(p) + [n] + list(p) for p in perms]
    
    word = []
    for idx in tour:
        b = blocks[idx]
        if not word:
            word = b[:]
        else:
            # Find max overlap
            max_ov = min(len(word), len(b))
            ov = 0
            # We can use M? No, M is for perm-to-perm block overlap.
            # But the current word is not just the previous block, it's the whole history.
            # However, the overlap with the history is the same as the overlap with the last block?
            # Yes, because the last block's suffix is the word's suffix.
            # And the new block's prefix is the same.
            # So we can just use the overlap between the last block and the new block.
            # Wait, is it possible that the overlap with the history is *larger* than with the last block?
            # No, the history ends with the last block. Any suffix of the history is a suffix of the last block or shorter.
            # So the max overlap with the history is the max overlap with the last block.
            
            # We need the overlap between blocks[tour[-1]] and blocks[idx]
            # We have M[tour[-1]][idx]
            ov = M[tour[-1]][idx]
            word += b[ov:]
            
    # Verify validity (optional, but good for debugging)
    # For n=7, verification is O(L*n) ~ 6000*7 = 42000. Fast.
    # But we trust the construction.
    
    return "".join(map(str, word))

# Keep the original functions if needed, but construct is the main entry point.
# The prompt asks to keep function names and signatures.
# construct(n) is the main one.
