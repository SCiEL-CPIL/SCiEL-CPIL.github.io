import time
import random
import math
from itertools import permutations

def construct(n):
    if n <= 1:
        return "1"
    
    if n == 2:
        return "121"
    
    if n == 3:
        return "123121321"
        
    if n < 7:
        # Standard recursive construction for smaller n
        # This is fast enough for n < 7
        if n == 1:
            return "1"
        prev_word = construct(n - 1)
        prev_list = [int(c) for c in prev_word]
        
        # Extract blocks
        seen = set()
        blocks = []
        for i in range(len(prev_list) - (n - 2)):
            window = tuple(prev_list[i:i + n - 1])
            if len(set(window)) == n - 1 and window not in seen:
                seen.add(window)
                blocks.append(window)
        
        # Reconstruct
        word = []
        for p in blocks:
            block = list(p) + [n] + list(p)
            max_k = min(len(word), len(block))
            overlap = 0
            for k in range(max_k, -1, -1):
                if k == 0:
                    overlap = 0
                    break
                if word[len(word) - k:] == block[:k]:
                    overlap = k
                    break
            word += block[overlap:]
        return "".join(map(str, word))
    
    # For n = 7
    start_time = time.time()
    time_limit = 145 # Safety margin
    
    # 1. Get superpermutation for n=6
    prev_word = construct(6)
    prev_list = [int(c) for c in prev_word]
    
    # 2. Extract blocks (permutations of 1..6)
    seen = set()
    blocks = []
    for i in range(len(prev_list) - 4):
        window = tuple(prev_list[i:i + 6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            blocks.append(window)
    
    num_blocks = len(blocks) # Should be 720
    
    # 3. Precompute overlap matrix
    # overlap[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    # k can be 0 to 5.
    overlap = [[0] * num_blocks for _ in range(num_blocks)]
    
    for i in range(num_blocks):
        bi = blocks[i]
        for j in range(num_blocks):
            if i == j:
                overlap[i][j] = 6
                continue
            bj = blocks[j]
            # Check k from 5 down to 0
            # Max overlap is 5 because if 6, they are identical permutations
            for k in range(5, -1, -1):
                if k == 0:
                    break
                if bi[len(bi) - k:] == bj[:k]:
                    overlap[i][j] = k
                    break
    
    # 4. Define cost function
    # We want to maximize sum of overlaps.
    # Let W[i][j] = -overlap[i][j]
    # Minimize sum W[order[i]][order[i+1]]
    
    # Greedy Nearest Neighbor Initialization
    def greedy_nn():
        visited = [False] * num_blocks
        order = []
        # Start with a random node or node 0
        current = random.randint(0, num_blocks - 1)
        visited[current] = True
        order.append(current)
        
        for _ in range(num_blocks - 1):
            best_next = -1
            best_cost = float('inf')
            for j in range(num_blocks):
                if not visited[j]:
                    cost = -overlap[current][j]
                    if cost < best_cost:
                        best_cost = cost
                        best_next = j
            order.append(best_next)
            visited[best_next] = True
            current = best_next
        return order

    # 2-Opt Improvement
    def two_opt(order):
        improved = True
        while improved:
            improved = False
            for i in range(num_blocks - 1):
                if time.time() - start_time > time_limit:
                    return order
                for j in range(i + 1, num_blocks):
                    # Calculate delta cost for reversing [i, j]
                    # Old edges: (i-1, i), (i, i+1)...(j-1, j), (j, j+1)
                    # New edges: (i-1, j), (j, j-1)...(i+1, i), (i, j+1)
                    
                    # To speed up, we can break early if possible, but delta calc is O(1) if we precompute prefix sums?
                    # No, W is not additive in a simple prefix way for arbitrary reversal.
                    # However, for 2-opt, the change is:
                    # Remove: W[i-1][i] + W[j][j+1] (if exist)
                    # Add: W[i-1][j] + W[i][j+1] (if exist)
                    # Internal edges are reversed.
                    # Sum(W[order[m]][order[m+1]] for m in i..j-1)
                    # vs
                    # Sum(W[order[j-(m-i)]][order[j-(m-i)-1]] for m in i..j-1)
                    
                    # This O(j-i) calculation is expensive if done for all i,j.
                    # N=720. N^2/2 = 259k pairs.
                    # If avg j-i is N/3 = 240.
                    # 259k * 240 = 62M operations per pass.
                    # In Python, 62M ops might take ~5-10 seconds.
                    # We can afford ~10-20 passes.
                    
                    delta = 0
                    if i > 0:
                        delta -= overlap[order[i-1]][order[i]]
                        delta += overlap[order[i-1]][order[j]]
                    if j < num_blocks - 1:
                        delta -= overlap[order[j]][order[j+1]]
                        delta += overlap[order[i]][order[j+1]]
                    
                    # Internal
                    old_sum = 0
                    new_sum = 0
                    for m in range(i, j):
                        old_sum += overlap[order[m]][order[m+1]]
                        idx1 = order[j - (m - i)]
                        idx2 = order[j - (m - i) - 1]
                        new_sum += overlap[idx1][idx2]
                    
                    delta += new_sum - old_sum
                    
                    if delta > 0: # We want to increase overlap, so if delta > 0, it's an improvement
                        order[i:j+1] = order[i:j+1][::-1]
                        improved = True
        
        return order

    # Or-Opt: Move a segment of length 1 or 2 to a new position
    def or_opt(order):
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > time_limit:
                return order
            
            # Try moving segment of length 1
            for i in range(num_blocks):
                if time.time() - start_time > time_limit:
                    return order
                node = order[i]
                # Remove node
                # Calculate cost change
                # This is complex. Let's stick to 2-opt for now as it's simpler and effective.
                # If 2-opt is too slow, we might need to optimize.
                
            # If 2-opt is the only move, we might get stuck.
            # Let's add a simple random restart if no improvement in 2-opt for a while?
            # Or just run multiple 2-opt passes.
            
        return order

    # Run search
    best_order = None
    best_overlap_sum = -1
    
    # Multiple restarts with GNN
    restarts = 0
    while time.time() - start_time < time_limit:
        restarts += 1
        current_order = greedy_nn()
        
        # Calculate initial overlap sum
        curr_sum = 0
        for i in range(num_blocks - 1):
            curr_sum += overlap[current_order[i]][current_order[i+1]]
        
        # Apply 2-Opt
        current_order = two_opt(current_order)
        
        # Recalculate sum
        final_sum = 0
        for i in range(num_blocks - 1):
            final_sum += overlap[current_order[i]][current_order[i+1]]
        
        if final_sum > best_overlap_sum:
            best_overlap_sum = final_sum
            best_order = current_order[:]
            
        # If we've done a few restarts, we might be converging
        if restarts >= 10:
            break

    # Reconstruct word from best_order
    word = []
    for idx in best_order:
        block = list(blocks[idx]) + [n] + list(blocks[idx])
        max_k = min(len(word), len(block))
        overlap_val = 0
        for k in range(max_k, -1, -1):
            if k == 0:
                overlap_val = 0
                break
            if word[len(word) - k:] == block[:k]:
                overlap_val = k
                break
        word += block[overlap_val:]
    
    return "".join(map(str, word))
