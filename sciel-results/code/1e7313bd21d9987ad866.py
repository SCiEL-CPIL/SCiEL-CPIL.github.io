import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # For small n, use the standard construction which is optimal
    if n <= 3:
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
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # For n >= 4, we try to improve the standard construction
    # Generate standard superpermutation for n-1 to get the order
    # Actually, we just need the order of permutations of 1..n-1 as they appear in S_{n-1}
    
    def get_standard_order(n_prev):
        word = [1]
        for m in range(2, n_prev + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        # Extract the order of permutations of 1..n_prev in the final word
        # The permutations are the windows of size n_prev that are valid perms
        # But we want the order of the *blocks* used in the last step?
        # The last step used permutations of 1..n_prev-1 to build blocks for n_prev.
        # Wait, the order of blocks in S_n is the order of permutations of 1..n-1 in S_{n-1}.
        # So we need the order of permutations of 1..n-1 in S_{n-1}.
        # Let's just return the word S_{n-1} and extract the order from it.
        return word

    # Get S_{n-1}
    s_prev = get_standard_order(n - 1)
    
    # Extract the order of permutations of 1..n-1 in s_prev
    # A permutation of 1..n-1 is a window of size n-1 with distinct elements
    order = []
    seen = set()
    for i in range(len(s_prev) - (n - 1) + 1):
        window = tuple(s_prev[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            order.append(window)
            
    # Generate blocks for n
    blocks = []
    for p in order:
        blocks.append(list(p) + [n] + list(p))
        
    M = len(blocks)
    
    # Precompute overlap matrix
    # O[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    overlap_matrix = [[0] * M for _ in range(M)]
    for i in range(M):
        for j in range(M):
            max_ov = 0
            # Max possible overlap is len(blocks[i]) - 1
            for k in range(min(len(blocks[i]), len(blocks[j])) - 1, -1, -1):
                if blocks[i][-k:] == blocks[j][:k]:
                    max_ov = k
                    break
            overlap_matrix[i][j] = max_ov

    def evaluate(order_idx):
        # Calculate length
        length = len(blocks[order_idx[0]])
        for i in range(1, M):
            length += len(blocks[order_idx[i]]) - overlap_matrix[order_idx[i-1]][order_idx[i]]
        return length

    def build_string(order_idx):
        s = blocks[order_idx[0]]
        for i in range(1, M):
            ov = overlap_matrix[order_idx[i-1]][order_idx[i]]
            s = s + blocks[order_idx[i]][ov:]
        return s

    def is_valid(s):
        # Check if all permutations of 1..n are in s
        # n is small (7), so we can just check all windows
        seen_perms = set()
        for i in range(len(s) - n + 1):
            window = tuple(s[i:i+n])
            if len(set(window)) == n:
                seen_perms.add(window)
        return len(seen_perms) == n

    # Initial evaluation
    current_order = list(range(M))
    current_string = build_string(current_order)
    current_length = len(current_string)
    is_current_valid = is_valid(current_string)
    
    best_order = current_order
    best_string = current_string
    best_length = current_length
    
    if not is_current_valid:
        # Should not happen for standard construction
        return "".join(map(str, best_string))

    # Local Search
    # We will try to improve the length by swapping blocks
    # Since the standard construction is valid, we start from a valid state.
    # Swaps might make it invalid. We only accept if valid and shorter.
    
    # Use a simple random restart / hill climbing
    # Time limit check
    while time.time() < deadline:
        # Pick a random swap
        i = random.randint(0, M - 2)
        j = i + 1
        # Or random pair
        # i, j = random.sample(range(M), 2)
        
        # Try swapping i and j
        new_order = current_order[:]
        new_order[i], new_order[j] = new_order[j], new_order[i]
        
        new_length = evaluate(new_order)
        
        # Quick check: if length is not better, skip build
        if new_length >= best_length:
            continue
            
        new_string = build_string(new_order)
        
        if is_valid(new_string):
            best_order = new_order
            best_string = new_string
            best_length = new_length
            current_order = new_order
            current_string = new_string
        else:
            # Revert
            pass
            # Maybe try to repair? No, just revert.

    return "".join(map(str, best_string))
