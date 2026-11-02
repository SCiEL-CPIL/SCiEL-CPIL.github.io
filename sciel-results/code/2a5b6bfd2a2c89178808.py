import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Precompute all t-subsets
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    # Precompute all k-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, precompute which triples it covers (as a set of indices)
    block_triples = []
    for b in all_blocks:
        tris = [triple_to_idx[tri] for tri in itertools.combinations(b, t)]
        block_triples.append(frozenset(tris))
    
    # For each triple, precompute which blocks cover it
    triple_blocks = [[] for _ in range(n_triples)]
    for i, b in enumerate(all_blocks):
        for tri_idx in block_triples[i]:
            triple_blocks[tri_idx].append(i)
    
    # Greedy with randomization
    def greedy_random():
        uncovered = set(range(n_triples))
        blocks = []
        
        while uncovered:
            # Pick random pivot triple
            pivot = random.choice(list(uncovered))
            pivot_triple = all_triples[pivot]
            pivot_points = set(pivot_triple)
            other_points = [p for p in range(v) if p not in pivot_points]
            
            best_block_idx = -1
            best_gain = -1
            
            # Find best block containing pivot points
            for b_idx, b in enumerate(all_blocks):
                if not set(b) >= pivot_points:
                    continue
                gain = len(block_triples[b_idx] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx >= 0:
                blocks.append(best_block_idx)
                uncovered -= block_triples[best_block_idx]
            
            # If no block found (shouldn't happen), pick any
            if best_block_idx < 0 and uncovered:
                pivot = min(uncovered)
                pivot_triple = all_triples[pivot]
                pivot_points = set(pivot_triple)
                for b_idx, b in enumerate(all_blocks):
                    if set(b) >= pivot_points:
                        blocks.append(b_idx)
                        uncovered -= block_triples[b_idx]
                        break
        
        return blocks
    
    # Check if a set of block indices covers all triples
    def is_complete(block_indices):
        if not block_indices:
            return n_triples == 0
        covered = set()
        for b_idx in block_indices:
            covered |= block_triples[b_idx]
        return len(covered) == n_triples
    
    # Remove redundant blocks
    def remove_redundant(block_indices):
        block_indices = list(block_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(block_indices)):
                remaining = block_indices[:i] + block_indices[i+1:]
                if is_complete(remaining):
                    block_indices.pop(i)
                    changed = True
                    break
        return block_indices
    
    # Local search: try to swap blocks to reduce count
    def local_search(block_indices, max_time=20):
        block_indices = list(block_indices)
        improved = True
        while improved and time.time() < start_time + max_time:
            improved = False
            n = len(block_indices)
            
            # Try removing each block
            for i in range(n):
                if time.time() > start_time + max_time:
                    return block_indices
                remaining = block_indices[:i] + block_indices[i+1:]
                if is_complete(remaining):
                    block_indices = remaining
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping one block for another
            for i in range(len(block_indices)):
                if time.time() > start_time + max_time:
                    return block_indices
                
                old_idx = block_indices[i]
                
                # Find triples only covered by this block
                only_triples = set()
                for tri_idx in block_triples[old_idx]:
                    covered_by_other = False
                    for j, b_idx in enumerate(block_indices):
                        if j != i and tri_idx in block_triples[b_idx]:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        only_triples.add(tri_idx)
                
                if not only_triples:
                    # Redundant, remove it
                    block_indices.pop(i)
                    improved = True
                    break
                
                # Find a replacement block that covers all only_triples
                # and has good coverage of other triples
                best_candidate = -1
                best_score = -1
                
                for new_idx in range(n_blocks):
                    if new_idx == old_idx or new_idx in block_indices:
                        continue
                    
                    # Must cover all only_triples
                    if not only_triples <= block_triples[new_idx]:
                        continue
                    
                    # Score: number of triples covered by new block that aren't covered by others
                    others_covered = set()
                    for j, b_idx in enumerate(block_indices):
                        if j != i:
                            others_covered |= block_triples[b_idx]
                    
                    gain = len(block_triples[new_idx] - others_covered)
                    if gain > best_score:
                        best_score = gain
                        best_candidate = new_idx
                
                if best_candidate >= 0:
                    block_indices[i] = best_candidate
                    improved = True
                    break
        
        return remove_redundant(block_indices)
    
    best_blocks = None
    best_len = float('inf')
    
    iterations = 0
    while time.time() < deadline - 5:
        blocks = greedy_random()
        blocks = remove_redundant(blocks)
        
        # Apply local search if we have time
        if time.time() < deadline - 25:
            blocks = local_search(blocks, max_time=20)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
            
        iterations += 1
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered = set(range(n_triples))
        blocks = []
        while uncovered:
            pivot = min(uncovered)
            pivot_triple = all_triples[pivot]
            pivot_points = set(pivot_triple)
            
            best_block_idx = -1
            best_gain = -1
            for b_idx, b in enumerate(all_blocks):
                if not set(b) >= pivot_points:
                    continue
                gain = len(block_triples[b_idx] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx >= 0:
                blocks.append(best_block_idx)
                uncovered -= block_triples[best_block_idx]
            else:
                break
        
        best_blocks = remove_redundant(blocks)
    
    # Convert block indices to actual blocks
    return [list(all_blocks[i]) for i in best_blocks]
