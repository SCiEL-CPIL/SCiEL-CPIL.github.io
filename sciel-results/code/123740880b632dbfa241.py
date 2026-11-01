import time
import sys

def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # Build S_n iteratively, keeping track of blocks for the last step to allow local search
    # For n < 7, we can just build the string.
    # For n == 7, we build the list of blocks and then optimize.
    
    word = "1"
    blocks = None
    current_m = 1
    
    for m in range(2, n + 1):
        perms = []
        seen = set()
        for i in range(len(word) - m + 2):
            window = word[i:i + m - 1]
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_blocks = []
        for p in perms:
            block = p + str(m) + p
            new_blocks.append(block)
            
        if m == n:
            blocks = new_blocks
            # Perform local search on blocks
            # Deadline check
            start_time = time.time()
            deadline = start_time + 100 # Leave 50s margin
            
            # Simple adjacent swap optimization
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(len(blocks) - 1):
                    # Calculate current overlap
                    b1 = blocks[i]
                    b2 = blocks[i+1]
                    curr_overlap = 0
                    limit = min(len(b1), len(b2))
                    for o in range(limit, 0, -1):
                        if b1.endswith(b2[:o]):
                            curr_overlap = o
                            break
                    
                    # Try swapping
                    # New overlap: overlap(blocks[i-1], b2) + overlap(b2, b1) + overlap(b1, blocks[i+2])
                    # vs overlap(blocks[i-1], b1) + overlap(b1, b2) + overlap(b2, blocks[i+2])
                    # We only need to check the change.
                    # Change = (overlap(i-1, b2) + overlap(b2, b1) + overlap(b1, i+2)) - (overlap(i-1, b1) + overlap(b1, b2) + overlap(b2, i+2))
                    
                    # Get neighbors
                    b_prev = blocks[i-1] if i > 0 else ""
                    b_next = blocks[i+2] if i + 2 < len(blocks) else ""
                    
                    # Current cost (length added)
                    # Cost = len(b1) + len(b2) - overlap(b1, b2) - overlap(prev, b1) - overlap(b2, next) ?
                    # Actually, total length = sum(len(b)) - sum(overlaps).
                    # Minimizing length = Maximizing overlaps.
                    
                    def get_overlap(a, b):
                        if not a or not b: return 0
                        limit = min(len(a), len(b))
                        for o in range(limit, 0, -1):
                            if a.endswith(b[:o]):
                                return o
                        return 0

                    curr_overlap_total = get_overlap(b_prev, b1) + get_overlap(b1, b2) + get_overlap(b2, b_next)
                    
                    # Swap b1, b2
                    new_overlap_total = get_overlap(b_prev, b2) + get_overlap(b2, b1) + get_overlap(b1, b_next)
                    
                    if new_overlap_total > curr_overlap_total:
                        blocks[i], blocks[i+1] = blocks[i+1], blocks[i]
                        improved = True
                        if time.time() > deadline:
                            break
            
            # Concatenate
            new_word = blocks[0]
            for i in range(1, len(blocks)):
                b_prev = new_word
                b_curr = blocks[i]
                overlap = 0
                limit = min(len(b_prev), len(b_curr))
                for o in range(limit, 0, -1):
                    if b_prev.endswith(b_curr[:o]):
                        overlap = o
                        break
                new_word += b_curr[overlap:]
            word = new_word
        else:
            # For m < n, just concatenate normally (or keep blocks if we want, but string is fine for small m)
            new_word = new_blocks[0]
            for i in range(1, len(new_blocks)):
                b_prev = new_word
                b_curr = new_blocks[i]
                overlap = 0
                limit = min(len(b_prev), len(b_curr))
                for o in range(limit, 0, -1):
                    if b_prev.endswith(b_curr[:o]):
                        overlap = o
                        break
                new_word += b_curr[overlap:]
            word = new_word
            
    return word
