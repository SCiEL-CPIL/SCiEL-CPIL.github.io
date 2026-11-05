import time
import random
import math

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "121321"
    
    # Recursively get the superpermutation for n-1
    prev_word_str = construct(n-1)
    prev_list = [int(c) for c in prev_word_str]
    
    # Extract the sequence of distinct (n-1)-permutations (blocks)
    # In the standard construction, these appear in a specific order.
    # We identify them by scanning for windows of size n-1 with distinct elements.
    seen = set()
    blocks = []
    block_indices = [] # Map from block tuple to index in blocks list
    for i in range(len(prev_list) - (n-2)):
        window = tuple(prev_list[i:i+n-1])
        if len(set(window)) == n-1:
            if window not in seen:
                seen.add(window)
                block_indices[window] = len(blocks)
                blocks.append(window)
                
    M = len(blocks)
    L = n - 1
    
    # Precompute overlap matrix
    # overlap[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    # Note: k can be 0 to L.
    # We only care about maximizing the sum of overlaps.
    # overlap[i][j] is independent of the order, it's a property of the pair.
    
    # Optimization: Store blocks as tuples for fast slicing? 
    # Python tuple slicing is fast.
    
    overlap = [[0]*M for _ in range(M)]
    for i in range(M):
        bi = blocks[i]
        for j in range(M):
            bj = blocks[j]
            # Max possible overlap is L
            # Check k from L down to 1
            # Since L is small (max 6 for n=7), this is fast.
            # Actually, for n=7, L=6.
            for k in range(L, 0, -1):
                if bi[-k:] == bj[:k]:
                    overlap[i][j] = k
                    break
    
    # Initial order: the order in which blocks appeared in prev_word
    # This corresponds to the standard construction order.
    # We need to map the sequence of blocks in prev_word to indices.
    # But prev_word was built from blocks. 
    # The order of blocks in the standard superpermutation of n-1 is the order they appear.
    # We extracted them in order of first appearance.
    # So initial_order is just [0, 1, ..., M-1]?
    # Let's verify. The standard construction for n-1 is built from n-2 blocks.
    # The blocks of n-1 are the permutations of n-1.
    # Yes, the order in which they appear in the standard S_{n-1} is the natural order.
    # However, the "standard" order might not be 0..M-1 if we extracted them differently.
    # We extracted them by scanning S_{n-1}. So block 0 is the first one found, etc.
    # So initial_order = list(range(M)) is correct for the standard construction.
    
    initial_order = list(range(M))
    
    def get_length(order):
        # Length = 2*L + 1 + sum(2*L + 1 - overlap[order[i]][order[i+1]]) for i=0..M-2
        # Base: 2*L + 1 (for the first block B0: B0 n B0)
        # Transitions: for each i, add (2*L + 1 - overlap[order[i]][order[i+1]])
        # Wait, the transition adds: suffix of B_{i+1} not overlapping + n + B_{i+1}
        # Added length = (L - k) + 1 + L = 2L + 1 - k.
        # So Total = (2L + 1) + sum_{i=0}^{M-2} (2L + 1 - overlap[order[i]][order[i+1]])
        
        total = 2 * L + 1
        for i in range(M - 1):
            total += (2 * L + 1 - overlap[order[i]][order[i+1]])
        return total
    
    best_order = initial_order[:]
    best_len = get_length(best_order)
    
    start_time = time.time()
    time_limit = 100 # Leave 20s for construction and overhead
    
    # VNS
    # Neighborhoods:
    # 1. 2-opt: Reverse segment [i, j]
    # 2. Or-opt: Move segment [i, j] to position k
    # 3. Swap: Swap i and j
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # Neighborhood 1: 2-opt
        for i in range(M - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, M):
                # Create new order by reversing i..j
                # Instead of creating new list, we can calculate delta or just swap
                # For simplicity and safety, create new list
                new_order = best_order[:i] + best_order[i:j+1][::-1] + best_order[j+1:]
                new_len = get_length(new_order)
                if new_len < best_len:
                    best_len = new_len
                    best_order = new_order
                    improved = True
                    # Restart search? VNS usually moves to new neighborhood or restarts.
                    # For simplicity, we accept and continue scanning or restart.
                    # Let's restart the neighborhood loop to exploit the new position.
                    break
            if improved:
                break
        
        if time.time() - start_time > time_limit:
            break
            
        if not improved:
            # Neighborhood 2: Or-opt (move segment)
            for i in range(M - 1):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i, M):
                    if time.time() - start_time > time_limit:
                        break
                    # Move segment i..j to position k
                    # Try moving to start and end first, then random?
                    # To keep it deterministic and fast, try a few positions.
                    # Or-opt is expensive to evaluate all k.
                    # Let's try swapping adjacent (Swap) as N3 if 2-opt fails.
                    pass # Skip complex Or-opt for now, 2-opt is powerful.
                    
            # Neighborhood 3: Swap adjacent
            for i in range(M - 1):
                if time.time() - start_time > time_limit:
                    break
                # Swap i and i+1
                new_order = best_order[:]
                new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
                new_len = get_length(new_order)
                if new_len < best_len:
                    best_len = new_len
                    best_order = new_order
                    improved = True
                    break
            
            if improved:
                continue
            
            # If no improvement in 2-opt or Swap, try a random perturbation (Simulated Annealing style)
            # or stop. Given time constraints, stopping is safer.
            # Let's try a few random 2-opts if stuck.
            random_improved = False
            for _ in range(10):
                if time.time() - start_time > time_limit:
                    break
                i = random.randint(0, M - 2)
                j = random.randint(i + 1, M - 1)
                new_order = best_order[:i] + best_order[i:j+1][::-1] + best_order[j+1:]
                new_len = get_length(new_order)
                if new_len < best_len:
                    best_len = new_len
                    best_order = new_order
                    random_improved = True
                    break
            
            if random_improved:
                improved = True
            else:
                improved = False

    # Reconstruct the word
    # Start with first block: B0 n B0
    # Then for each next block B_i, append (B_i[overlap:] + n + B_i)
    # Wait, the overlap is between previous block B_{i-1} and current B_i.
    # prev_end = B_{i-1}
    # curr_start = B_i
    # overlap k = overlap[prev_idx][curr_idx]
    # We have already written up to B_{i-1}.
    # We need to append the rest of B_i that doesn't overlap, then n, then B_i.
    # Appended: B_i[k:] + str(n) + B_i
    
    word = []
    first_block = best_order[0]
    word.extend(blocks[first_block])
    word.append(n)
    word.extend(blocks[first_block])
    
    for i in range(1, M):
        prev_idx = best_order[i-1]
        curr_idx = best_order[i]
        k = overlap[prev_idx][curr_idx]
        curr_block = blocks[curr_idx]
        # Append suffix of curr_block starting from k
        word.extend(curr_block[k:])
        word.append(n)
        word.extend(curr_block)
        
    return "".join(map(str, word))
