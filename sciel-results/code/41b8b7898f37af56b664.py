import time
import random
import math

def construct(n):
    if n <= 1:
        return "1"
    
    # Generate the standard superpermutation for n-1
    prev_word = construct(n-1)
    prev_list = [int(c) for c in prev_word]
    
    # Extract the sequence of (n-1)-permutations from prev_word
    seen = set()
    blocks = []
    for i in range(len(prev_list) - (n-2)):
        window = tuple(prev_list[i:i+n-1])
        if len(set(window)) == n-1:
            if window not in seen:
                seen.add(window)
                blocks.append(window)
    
    # Precompute overlap matrix between all pairs of blocks
    # overlap[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    num_blocks = len(blocks)
    block_len = n - 1
    overlap = [[0] * num_blocks for _ in range(num_blocks)]
    
    for i in range(num_blocks):
        for j in range(num_blocks):
            if i == j:
                overlap[i][j] = 0
                continue
            bi = blocks[i]
            bj = blocks[j]
            max_k = min(block_len, block_len)
            for k in range(max_k, 0, -1):
                if bi[-k:] == bj[:k]:
                    overlap[i][j] = k
                    break
    
    # Calculate total length for a given ordering
    def calc_length(order):
        if not order:
            return 0
        # First block contributes: len(block) + 1 (for n) + len(block)
        total = block_len + 1 + block_len
        for idx in range(len(order) - 1):
            curr_idx = order[idx]
            next_idx = order[idx + 1]
            ov = overlap[curr_idx][next_idx]
            # Added: 1 (for n) + len(next_block) - ov (overlap with current block's end)
            total += 1 + block_len - ov
        return total
    
    # Initialize with standard order
    best_order = list(range(num_blocks))
    best_len = calc_length(best_order)
    
    start_time = time.time()
    time_limit = 105  # Leave 15s for overhead and reconstruction
    
    # Variable Neighborhood Search
    # Neighborhood 1: 2-opt (reverse contiguous segment)
    # Neighborhood 2: Or-opt (move a contiguous segment of length 1, 2, or 3)
    
    def try_2opt(order, best_len_val):
        """Try 2-opt moves. Return improved order and length if found."""
        n_blocks = len(order)
        for i in range(n_blocks - 1):
            for j in range(i + 1, n_blocks):
                # Calculate new length efficiently
                # Only edges (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1) change
                # Old edges: (i-1, i) if i>0, (j, j+1) if j<n_blocks-1
                # New edges: (i-1, j) if i>0, (j, i) if j<n_blocks-1
                old_len = best_len_val
                
                # Remove old edges
                if i > 0:
                    old_len -= 1 + block_len - overlap[order[i-1]][order[i]]
                if j < n_blocks - 1:
                    old_len -= 1 + block_len - overlap[order[j]][order[j+1]]
                
                # Add new edges
                if i > 0:
                    old_len += 1 + block_len - overlap[order[i-1]][order[j]]
                if j < n_blocks - 1:
                    old_len += 1 + block_len - overlap[order[j]][order[i]]
                
                # Internal edges within reversed segment don't change in count, but their values change
                # Actually, the internal edges are reversed, so we need to account for that
                # Let me recalculate more carefully
                
                # Better approach: compute delta
                delta = 0
                # Edge before i: was (i-1, i), now (i-1, j)
                if i > 0:
                    delta += (1 + block_len - overlap[order[i-1]][order[j]]) - (1 + block_len - overlap[order[i-1]][order[i]])
                # Edge after j: was (j, j+1), now (j, i)  
                if j < n_blocks - 1:
                    delta += (1 + block_len - overlap[order[j]][order[i]]) - (1 + block_len - overlap[order[j]][order[j+1]])
                # Internal edges: for k from i to j-1, edge was (k, k+1), now (j-(k-i), j-(k-i)+1)
                # i.e., edge (k, k+1) becomes (j - (k - i), j - (k - i) + 1) = (j - k + i, j - k + i + 1)
                for k in range(i, j):
                    old_edge = (order[k], order[k+1])
                    new_idx = j - (k - i)
                    new_edge = (order[new_idx], order[new_idx + 1])
                    # Wait, after reversing i..j, position k gets element order[j-(k-i)]
                    # Position k+1 gets element order[j-(k-i)-1]
                    # So edge at position k becomes (order[j-(k-i)], order[j-(k-i)-1])
                    # The contribution was 1 + block_len - overlap[order[k]][order[k+1]]
                    # New contribution is 1 + block_len - overlap[order[j-(k-i)]][order[j-(k-i)-1]]
                    old_contrib = 1 + block_len - overlap[order[k]][order[k+1]]
                    new_contrib = 1 + block_len - overlap[order[j-(k-i)]][order[j-(k-i)-1]]
                    delta += new_contrib - old_contrib
                
                new_len = best_len_val + delta
                if new_len < best_len_val:
                    new_order = order[:i] + order[i:j+1][::-1] + order[j+1:]
                    return new_order, new_len
        return None, best_len_val
    
    def try_oropt(order, best_len_val, seg_len):
        """Try Or-opt moves: move a segment of length seg_len to another position."""
        n_blocks = len(order)
        for i in range(n_blocks - seg_len):
            seg = order[i:i+seg_len]
            for j in range(n_blocks - seg_len + 1):
                if j == i or j == i + seg_len:
                    continue
                # Remove segment from i, insert at j
                # This changes multiple edges, so compute delta
                delta = 0
                
                # Determine the actual positions after removal and insertion
                # Case 1: j > i + seg_len (insert after the segment)
                # Case 2: j < i (insert before the segment)
                
                if j < i:
                    # Insert at j, so segment moves from i to j
                    # Edges affected:
                    # Before i: (i-1, i) if i>0
                    # After i+seg_len-1: (i+seg_len-1, i+seg_len) if i+seg_len<n_blocks
                    # Before j: (j-1, j) if j>0
                    # After j+seg_len-1: (j+seg_len-1, j+seg_len) if j+seg_len<n_blocks (but this is after the moved segment)
                    
                    # Let's think of it as: remove segment at i, then insert at j
                    # After removal, indices shift
                    # This is complex, so let's just compute full length for small segments
                    
                    # For efficiency, compute delta by tracking all changed edges
                    # Original edges:
                    # (i-1, i) if i>0
                    # (i+seg_len-1, i+seg_len) if i+seg_len < n_blocks
                    # (j-1, j) if j>0
                    # (j+seg_len-1, j+seg_len) if j+seg_len < n_blocks
                    
                    # New edges:
                    # (i-1, i+seg_len) if i>0 and i+seg_len < n_blocks
                    # (j-1, seg[0]) if j>0
                    # (seg[-1], j+seg_len) if j+seg_len < n_blocks
                    # (i-1, i+seg_len) if i>0 -- wait, need to be careful
                    
                    # Let me use a different approach: build new order and compute delta
                    new_order = order[:j] + seg + order[j:i] + order[i+seg_len:]
                    
                    # Compute delta by comparing edge contributions
                    # This is O(n) per move, which is acceptable for small seg_len
                    old_contrib = 0
                    new_contrib = 0
                    for idx in range(n_blocks - 1):
                        if idx < i or idx >= i + seg_len - 1:
                            pass  # These edges might be affected
                    
                    # Simpler: just compute the difference directly
                    # Edges in old order that are removed:
                    removed_edges = set()
                    added_edges = set()
                    
                    # Old edge (i-1, i) if i > 0
                    if i > 0:
                        removed_edges.add((order[i-1], order[i]))
                    # Old edge (i+seg_len-1, i+seg_len) if i+seg_len < n_blocks
                    if i + seg_len < n_blocks:
                        removed_edges.add((order[i+seg_len-1], order[i+seg_len]))
                    # Old edge (j-1, j) if j > 0
                    if j > 0:
                        removed_edges.add((order[j-1], order[j]))
                    # Old edge (j+seg_len-1, j+seg_len) if j+seg_len < n_blocks
                    if j + seg_len < n_blocks:
                        removed_edges.add((order[j+seg_len-1], order[j+seg_len]))
                    
                    # New edge (i-1, i+seg_len) if i > 0 and i+seg_len < n_blocks
                    if i > 0 and i + seg_len < n_blocks:
                        added_edges.add((order[i-1], order[i+seg_len]))
                    # New edge (j-1, seg[0]) if j > 0
                    if j > 0:
                        added_edges.add((order[j-1], seg[0]))
                    # New edge (seg[-1], j+seg_len) if j+seg_len < n_blocks
                    if j + seg_len < n_blocks:
                        added_edges.add((seg[-1], order[j+seg_len]))
                    
                    for edge in removed_edges:
                        delta -= 1 + block_len - overlap[edge[0]][edge[1]]
                    for edge in added_edges:
                        delta += 1 + block_len - overlap[edge[0]][edge[1]]
                    
                    new_len = best_len_val + delta
                    if new_len < best_len_val:
                        return new_order, new_len
                else:
                    # j > i + seg_len
                    # Similar logic
                    new_order = order[:i] + order[i+seg_len:j] + seg + order[j:]
                    
                    delta = 0
                    removed_edges = set()
                    added_edges = set()
                    
                    # Old edge (i-1, i) if i > 0
                    if i > 0:
                        removed_edges.add((order[i-1], order[i]))
                    # Old edge (i+seg_len-1, i+seg_len) if i+seg_len < n_blocks
                    if i + seg_len < n_blocks:
                        removed_edges.add((order[i+seg_len-1], order[i+seg_len]))
                    # Old edge (j-1, j) if j > 0
                    if j > 0:
                        removed_edges.add((order[j-1], order[j]))
                    # Old edge (j+seg_len-1, j+seg_len) if j+seg_len < n_blocks
                    if j + seg_len < n_blocks:
                        removed_edges.add((order[j+seg_len-1], order[j+seg_len]))
                    
                    # New edge (i-1, i+seg_len) if i > 0 and i+seg_len < n_blocks
                    if i > 0 and i + seg_len < n_blocks:
                        added_edges.add((order[i-1], order[i+seg_len]))
                    # New edge (j-1, seg[0]) if j > 0
                    if j > 0:
                        added_edges.add((order[j-1], seg[0]))
                    # New edge (seg[-1], j+seg_len) if j+seg_len < n_blocks
                    if j + seg_len < n_blocks:
                        added_edges.add((seg[-1], order[j+seg_len]))
                    
                    for edge in removed_edges:
                        delta -= 1 + block_len - overlap[edge[0]][edge[1]]
                    for edge in added_edges:
                        delta += 1 + block_len - overlap[edge[0]][edge[1]]
                    
                    new_len = best_len_val + delta
                    if new_len < best_len_val:
                        return new_order, new_len
        
        return None, best_len_val
    
    # VNS loop
    improved = True
    neighborhood_idx = 0
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        if neighborhood_idx == 0:
            # 2-opt
            new_order, new_len = try_2opt(best_order, best_len)
            if new_order is not None:
                best_order = new_order
                best_len = new_len
                improved = True
        elif neighborhood_idx == 1:
            # Or-opt with segment length 1
            new_order, new_len = try_oropt(best_order, best_len, 1)
            if new_order is not None:
                best_order = new_order
                best_len = new_len
                improved = True
        elif neighborhood_idx == 2:
            # Or-opt with segment length 2
            new_order, new_len = try_oropt(best_order, best_len, 2)
            if new_order is not None:
                best_order = new_order
                best_len = new_len
                improved = True
        elif neighborhood_idx == 3:
            # Or-opt with segment length 3
            new_order, new_len = try_oropt(best_order, best_len, 3)
            if new_order is not None:
                best_order = new_order
                best_len = new_len
                improved = True
        
        if not improved:
            neighborhood_idx = (neighborhood_idx + 1) % 4
        else:
            neighborhood_idx = 0  # Reset to smallest neighborhood on improvement
    
    # Reconstruct the word from best_order
    # The word is: for each block in order, append [n] + block, with overlap
    word = []
    for idx, block_idx in enumerate(best_order):
        block = blocks[block_idx]
        segment = [n] + list(block)
        
        if idx == 0:
            word.extend(segment)
        else:
            # Find max overlap between current word's end and segment's start
            max_k = min(len(word), len(segment))
            ov = 0
            for k in range(max_k, 0, -1):
                if word[-k:] == segment[:k]:
                    ov = k
                    break
            word.extend(segment[ov:])
    
    return "".join(map(str, word))
