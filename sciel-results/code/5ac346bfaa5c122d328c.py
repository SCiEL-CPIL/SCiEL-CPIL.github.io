import itertools
import random
import time

# Global cache for triple indices of blocks
_block_triple_cache = {}

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Precompute all triples
    points = range(v)
    all_triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    def get_block_triples(block):
        """Return list of triple indices covered by the block, using cache."""
        if block in _block_triple_cache:
            return _block_triple_cache[block]
        res = [triple_to_idx[triple] for triple in itertools.combinations(block, t)]
        _block_triple_cache[block] = res
        return res

    def greedy_cover():
        """Run a single greedy construction."""
        uncovered = [True] * num_triples
        blocks = []
        
        while any(uncovered):
            # Pick a random uncovered triple to start the block
            uncov_indices = [i for i, u in enumerate(uncovered) if u]
            if not uncov_indices:
                break
            start_triple_idx = random.choice(uncov_indices)
            start_triple = all_triples[start_triple_idx]
            
            current_block = list(start_triple)
            
            # Expand block to size k
            while len(current_block) < k:
                best_gain = -1
                best_points = []
                
                for p in points:
                    if p in current_block:
                        continue
                    
                    # Calculate gain
                    new_block = tuple(sorted(current_block + [p]))
                    gain = 0
                    # Only check triples in new_block
                    for triple in itertools.combinations(new_block, t):
                        idx = triple_to_idx[triple]
                        if uncovered[idx]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [p]
                    elif gain == best_gain:
                        best_points.append(p)
                
                # Pick a point from best_points randomly
                if best_points:
                    current_block.append(random.choice(best_points))
                else:
                    break
            
            # Add block
            block_tuple = tuple(sorted(current_block))
            blocks.append(block_tuple)
            
            # Update uncovered
            for idx in get_block_triples(block_tuple):
                uncovered[idx] = False
                
        return blocks

    def local_search(blocks):
        """Perform local search to reduce block count."""
        blocks = list(blocks)
        
        # Initialize cover counts
        cover_count = [0] * num_triples
        for b in blocks:
            for idx in get_block_triples(b):
                cover_count[idx] += 1
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            for i in range(len(blocks)):
                if time.time() > deadline:
                    break
                
                b = blocks[i]
                b_triples = get_block_triples(b)
                
                # Tentatively remove b
                for idx in b_triples:
                    cover_count[idx] -= 1
                
                # Check if covering is still valid
                essential_triples = []
                valid = True
                for idx in range(num_triples):
                    if cover_count[idx] == 0:
                        essential_triples.append(idx)
                        valid = False
                
                if valid:
                    # Success! Remove b
                    blocks.pop(i)
                    improved = True
                    break
                else:
                    # Try to find a replacement block b_new
                    essential_points_set = set()
                    for idx in essential_triples:
                        essential_points_set.update(all_triples[idx])
                    
                    essential_points = list(essential_points_set)
                    
                    if len(essential_points) > k:
                        # Cannot cover with one block
                        for idx in b_triples:
                            cover_count[idx] += 1
                        continue
                    
                    # Generate candidate blocks containing essential_points
                    remaining_points = [p for p in points if p not in essential_points_set]
                    need = k - len(essential_points)
                    
                    found_swap = False
                    for combo in itertools.combinations(remaining_points, need):
                        candidate = tuple(sorted(essential_points + list(combo)))
                        
                        # Check if candidate is in current blocks (excluding b)
                        # Note: b is still in blocks list at index i
                        # If candidate == b, skip.
                        # If candidate is another block, skip.
                        if candidate in blocks:
                            continue
                        
                        # Valid swap
                        for idx in get_block_triples(candidate):
                            cover_count[idx] += 1
                        
                        blocks[i] = candidate
                        improved = True
                        found_swap = True
                        break
                    
                    if not found_swap:
                        # Restore b
                        for idx in b_triples:
                            cover_count[idx] += 1
        
        return blocks

    # Run multiple greedy restarts
    best_blocks = None
    best_size = float('inf')
    
    # Time budget for greedy: 20s
    greedy_deadline = start_time + 20.0
    
    while time.time() < greedy_deadline:
        blocks = greedy_cover()
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            
    # Run local search on the best greedy result
    if best_blocks is not None:
        best_blocks = local_search(best_blocks)
        
    return [list(b) for b in best_blocks]
