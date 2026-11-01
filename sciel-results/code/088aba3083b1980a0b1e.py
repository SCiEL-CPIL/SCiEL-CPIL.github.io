import time
import random

def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412314231243121342132413214321"
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Generate standard superpermutation for n-1
    word = "1"
    for m in range(2, n):
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = word[i:i + m - 1]
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = ""
        for p in perms:
            block = p + str(m) + p
            max_overlap = 0
            limit = min(len(new_word), len(block))
            for o in range(limit, 0, -1):
                if new_word[-o:] == block[:o]:
                    max_overlap = o
                    break
            new_word += block[max_overlap:]
        word = new_word
        
    # Now word is superpermutation for n-1
    # Extract permutations of 1..n-1
    perms = []
    seen = set()
    last_m = n - 1
    for i in range(len(word) - last_m + 1):
        window = word[i:i + last_m]
        if len(set(window)) == last_m:
            if window not in seen:
                seen.add(window)
                perms.append(window)
                
    # perms is the list of blocks for n
    # Each block is p + str(n) + p
    # We want to order these blocks to maximize overlap
    
    num_blocks = len(perms)
    
    # Precompute overlap matrix
    # overlap[i][j] is the max overlap between block_i and block_j
    # block_i = perms[i] + str(n) + perms[i]
    
    # To speed up, we can just compute overlaps on the fly or use a cache
    # But 720x720 is small enough to precompute if we are careful.
    
    # Let's create the full block strings first
    blocks = []
    for p in perms:
        blocks.append(p + str(n) + p)
        
    # Precompute overlaps
    # overlap[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    # Max possible overlap is len(blocks[j]) which is 2*n
    # But usually small.
    
    overlap = [[0] * num_blocks for _ in range(num_blocks)]
    
    for i in range(num_blocks):
        bi = blocks[i]
        for j in range(num_blocks):
            bj = blocks[j]
            # Try overlaps from max possible down to 0
            # Max overlap is min(len(bi), len(bj))
            # But we can limit to len(bi)
            max_o = min(len(bi), len(bj))
            # Optimization: we only need to check up to a certain limit?
            # Actually, for superpermutations, overlaps are usually <= n
            # Let's check up to 2*n just in case
            limit = min(max_o, 2 * n)
            for o in range(limit, 0, -1):
                if bi[-o:] == bj[:o]:
                    overlap[i][j] = o
                    break

    # Initial order is the order in perms
    current_order = list(range(num_blocks))
    
    # Calculate initial total overlap
    def get_total_overlap(order):
        total = 0
        for k in range(len(order) - 1):
            total += overlap[order[k]][order[k+1]]
        return total
        
    current_score = get_total_overlap(current_order)
    
    # Local Search
    # Try 2-opt moves: reverse a segment
    # Or swap adjacent elements
    
    # Simulated Annealing or Hill Climbing
    # Given time, Hill Climbing with random restarts or 2-opt is good.
    
    improved = True
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline and improved:
        improved = False
        # Try all 2-opt moves? 720^2 / 2 ~ 250k moves.
        # Each move check is O(1) if we maintain prefix/suffix sums?
        # Or just calculate delta.
        # Delta for reversing segment [i, j]:
        # Old edges: (i-1, i), (j, j+1)
        # New edges: (i-1, j), (i, j+1)
        # Internal edges are reversed, but overlap is not symmetric.
        # So we must recalculate the sum for the segment?
        # That is O(L) for segment length L.
        # Average L is small?
        
        # Simpler: Try swapping adjacent elements (1-opt)
        # Or random 2-opt segments.
        
        # Let's do random 2-opt moves
        for _ in range(1000): # 1000 moves per iteration
            if time.time() > deadline:
                break
                
            i = random.randint(0, num_blocks - 2)
            j = random.randint(i + 1, num_blocks - 1)
            
            # Calculate delta
            # Remove edges (i-1, i) and (j, j+1)
            # Add edges (i-1, j) and (i, j+1)
            # And reverse internal edges
            
            # This is complex to do in O(1).
            # Let's just try adjacent swaps (1-opt) first, it's O(1) delta.
            # If stuck, do random segment reversals with O(L) cost.
            
            # Adjacent swap i, i+1
            if i == num_blocks - 1:
                continue
                
            # Old: ... a, b, c ...
            # New: ... a, c, b ...
            # Old edges: (a,b), (b,c)
            # New edges: (a,c), (c,b)
            
            idx1 = current_order[i-1] if i > 0 else -1
            idx2 = current_order[i]
            idx3 = current_order[i+1]
            idx4 = current_order[i+2] if i+2 < num_blocks else -1
            
            # We need to be careful with boundaries
            # Let's implement a generic delta calculation for swapping i and i+1
            
            # Get neighbors
            # Left neighbor of i is current_order[i-1]
            # Right neighbor of i+1 is current_order[i+2]
            
            # If i=0:
            # Old: (0,1)
            # New: (1,0)
            # Delta = overlap[1][0] - overlap[0][1]
            
            # If i>0 and i+1 < N-1:
            # Old: (i-1, i), (i, i+1), (i+1, i+2)
            # New: (i-1, i+1), (i+1, i), (i, i+2)
            # Wait, swapping i and i+1 means order becomes ..., i-1, i+1, i, i+2, ...
            # Edges removed: (i-1, i), (i, i+1), (i+1, i+2)
            # Edges added: (i-1, i+1), (i+1, i), (i, i+2)
            
            delta = 0
            if i > 0:
                delta -= overlap[current_order[i-1]][current_order[i]]
                delta += overlap[current_order[i-1]][current_order[i+1]]
            
            delta -= overlap[current_order[i]][current_order[i+1]]
            delta += overlap[current_order[i+1]][current_order[i]]
            
            if i + 2 < num_blocks:
                delta -= overlap[current_order[i+1]][current_order[i+2]]
                delta += overlap[current_order[i]][current_order[i+2]]
            
            if delta > 0:
                # Perform swap
                current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
                current_score += delta
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
                
        if no_improve_count > max_no_improve:
            # Try a random restart or more aggressive move?
            # For now, break if no improvement in many tries
            # Or do a random perturbation
            pass

    # Reconstruct word
    # Start with first block
    final_word = blocks[current_order[0]]
    for k in range(1, num_blocks):
        prev_block = blocks[current_order[k-1]]
        curr_block = blocks[current_order[k]]
        o = overlap[current_order[k-1]][current_order[k]]
        final_word += curr_block[o:]
        
    return final_word
