import time
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin for 120s limit
    
    points = range(12)
    all_5subsets = list(combinations(points, 5))
    num_5subsets = len(all_5subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_5subsets)}
    
    all_6subsets = list(combinations(points, 6))
    num_blocks = len(all_6subsets)
    
    # Pre-compute block contents
    block_subs = []
    candidates = [[] for _ in range(num_5subsets)]
    
    for b_idx, b in enumerate(all_6subsets):
        subs = []
        for s in combinations(b, 5):
            s_idx = subset_to_idx[s]
            subs.append(s_idx)
            candidates[s_idx].append(b_idx)
        block_subs.append(subs)
        
    # Pre-compute conflicts
    conflicts = [set() for _ in range(num_blocks)]
    for b_idx, subs in enumerate(block_subs):
        for s_idx in subs:
            for other_b in candidates[s_idx]:
                if other_b != b_idx:
                    conflicts[b_idx].add(other_b)
    conflicts = [list(s) for s in conflicts]
    
    valid = [True] * num_blocks
    best_solution = []
    
    # Symmetry Breaking: Fix the first block to be (0, 1, 2, 3, 4, 5)
    # This is valid because the group acting on the design is transitive on blocks.
    first_block_idx = None
    for i, b in enumerate(all_6subsets):
        if b == (0, 1, 2, 3, 4, 5):
            first_block_idx = i
            break
            
    if first_block_idx is None:
        # Fallback if not found (should not happen)
        first_block_idx = 0
        # Adjust logic if needed, but (0..5) is in combinations(0..11, 6)
        
    # Apply first block
    current_blocks = [first_block_idx]
    valid[first_block_idx] = False
    initial_uncovered = set(range(num_5subsets))
    for s_idx in block_subs[first_block_idx]:
        initial_uncovered.discard(s_idx)
        
    # Invalidate conflicting blocks
    for c in conflicts[first_block_idx]:
        valid[c] = False
        
    def backtrack(uncovered, current_blocks):
        nonlocal best_solution
        if time.time() - start_time > timeout:
            return False
            
        if not uncovered:
            best_solution = current_blocks[:]
            return True
            
        # Heuristic: Min remaining values
        min_subset = -1
        min_count = float('inf')
        
        # Iterate uncovered subsets
        # Optimization: If we have many uncovered, sampling or limiting checks might help,
        # but 792 is small enough to scan fully if the inner loop is fast.
        for s_idx in uncovered:
            count = 0
            for b_idx in candidates[s_idx]:
                if valid[b_idx]:
                    count += 1
            if count < min_count:
                min_count = count
                min_subset = s_idx
            if min_count == 0:
                return False
            if min_count == 1:
                break
                
        if min_subset == -1:
            return False # Should not happen if uncovered is not empty
            
        # Try blocks
        # Sort candidates by some heuristic? e.g. degree?
        # For now, just iterate.
        for b_idx in candidates[min_subset]:
            if not valid[b_idx]:
                continue
                
            # Apply choice
            undo_stack = [b_idx]
            valid[b_idx] = False
            
            for c in conflicts[b_idx]:
                if valid[c]:
                    valid[c] = False
                    undo_stack.append(c)
                    
            new_uncovered = set(uncovered)
            for s_idx in block_subs[b_idx]:
                new_uncovered.discard(s_idx)
                
            if backtrack(new_uncovered, current_blocks + [b_idx]):
                # Undo not strictly needed if returning True immediately and exiting,
                # but good practice.
                for idx in reversed(undo_stack):
                    valid[idx] = True
                return True
                
            # Undo
            for idx in reversed(undo_stack):
                valid[idx] = True
                
        return False
        
    if backtrack(initial_uncovered, current_blocks):
        return [list(all_6subsets[i]) for i in best_solution]
    else:
        # Fallback: Greedy
        covered = set()
        blocks = []
        # Try to add blocks greedily to maximize count
        # Sort blocks by size of intersection with covered? No, just random or index order.
        # Better: Sort by number of newly covered subsets.
        remaining_blocks = [i for i in range(num_blocks) if i != first_block_idx and valid[i] is not None] 
        # Note: valid array was modified. We should reconstruct a fresh greedy if backtrack fails.
        
        # Re-init for greedy
        valid_greedy = [True] * num_blocks
        valid_greedy[first_block_idx] = False
        for c in conflicts[first_block_idx]:
            valid_greedy[c] = False
            
        uncovered_greedy = set(range(num_5subsets))
        for s_idx in block_subs[first_block_idx]:
            uncovered_greedy.discard(s_idx)
            
        greedy_blocks = [first_block_idx]
        
        while True:
            best_b = -1
            max_new = 0
            for b_idx in range(num_blocks):
                if not valid_greedy[b_idx]:
                    continue
                new_count = 0
                for s_idx in block_subs[b_idx]:
                    if s_idx in uncovered_greedy:
                        new_count += 1
                if new_count > max_new:
                    max_new = new_count
                    best_b = b_idx
            if best_b == -1 or max_new == 0:
                break
            greedy_blocks.append(best_b)
            valid_greedy[best_b] = False
            for c in conflicts[best_b]:
                valid_greedy[c] = False
            for s_idx in block_subs[best_b]:
                uncovered_greedy.discard(s_idx)
                
        return [list(all_6subsets[i]) for i in greedy_blocks]
