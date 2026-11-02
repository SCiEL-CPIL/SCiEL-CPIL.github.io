import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_blocks = None
    best_len = float('inf')
    
    def is_valid(blocks):
        covered = set()
        for b in blocks:
            covered.update(itertools.combinations(b, t))
        return len(covered) == len(list(itertools.combinations(range(v), t)))
        
    def remove_redundant(blocks):
        blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                remaining = blocks[:i] + blocks[i+1:]
                covered = set()
                for b in remaining:
                    covered.update(itertools.combinations(b, t))
                if len(covered) == len(list(itertools.combinations(range(v), t))):
                    blocks.pop(i)
                    changed = True
                    break
        return blocks

    def greedy_random_pivot():
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        
        while uncovered:
            pivot_triple = random.choice(list(uncovered))
            pivot_points = set(pivot_triple)
            other_points = [p for p in range(v) if p not in pivot_points]
            
            best_block = None
            best_gain = -1
            
            for combo in itertools.combinations(other_points, k - t):
                block = tuple(sorted(pivot_points | set(combo)))
                gain = 0
                for tri in itertools.combinations(block, t):
                    if tri in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is not None:
                blocks.append(list(best_block))
                for tri in itertools.combinations(best_block, t):
                    if tri in uncovered:
                        uncovered.remove(tri)
                        
        return blocks

    def local_search(blocks, max_time=30):
        """Attempt to reduce block count via swaps and removals."""
        if time.time() > start_time + max_time:
            return blocks
        
        blocks = list(blocks)
        n = len(blocks)
        improved = True
        while improved and time.time() < start_time + max_time:
            improved = False
            # Try removing each block
            for i in range(len(blocks)):
                if time.time() > start_time + max_time:
                    return blocks
                remaining = blocks[:i] + blocks[i+1:]
                if is_valid(remaining):
                    blocks = remaining
                    improved = True
                    break
            
            if not improved:
                # Try swapping a block with a better one
                for i in range(len(blocks)):
                    if time.time() > start_time + max_time:
                        return blocks
                    # Generate candidate blocks that cover the same triples as block i
                    # or better ones
                    old_block = set(blocks[i])
                    # Find triples that are only covered by this block
                    only_triples = set()
                    for tri in itertools.combinations(blocks[i], t):
                        count = 0
                        for j, b in enumerate(blocks):
                            if j != i and tri in itertools.combinations(b, t):
                                count += 1
                        if count == 0:
                            only_triples.add(tri)
                    
                    if not only_triples:
                        # This block is redundant, remove it
                        blocks.pop(i)
                        improved = True
                        break
                    
                    # Try to find a replacement block that covers all only_triples
                    # and has higher gain
                    all_points = set(range(v))
                    must_include = set()
                    for tri in only_triples:
                        must_include.update(tri)
                    
                    if len(must_include) > k:
                        continue
                    
                    # Generate candidates containing must_include
                    remaining_points = list(all_points - must_include)
                    need = k - len(must_include)
                    best_candidate = None
                    best_score = -1
                    
                    for combo in itertools.combinations(remaining_points, need):
                        candidate = tuple(sorted(must_include | set(combo)))
                        # Score: number of uncovered triples this would cover if we remove old block
                        gain = 0
                        for tri in itertools.combinations(candidate, t):
                            # Check if this triple is covered by any other block
                            covered = False
                            for j, b in enumerate(blocks):
                                if j != i and tri in itertools.combinations(b, t):
                                    covered = True
                                    break
                            if not covered:
                                gain += 1
                        if gain > best_score:
                            best_score = gain
                            best_candidate = candidate
                    
                    if best_candidate is not None and best_score >= len(only_triples):
                        blocks[i] = list(best_candidate)
                        improved = True
                        break
        
        return remove_redundant(blocks)

    iterations = 0
    while time.time() < deadline:
        blocks = greedy_random_pivot()
        blocks = remove_redundant(blocks)
        
        # Apply local search if we have time
        if time.time() < deadline - 20:
            blocks = local_search(blocks, max_time=20)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
            
        iterations += 1
        
    if best_blocks is None:
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            pivot_triple = min(uncovered)
            pivot_points = set(pivot_triple)
            other_points = [p for p in range(v) if p not in pivot_points]
            best_block = None
            best_gain = -1
            for combo in itertools.combinations(other_points, k - t):
                block = tuple(sorted(pivot_points | set(combo)))
                gain = sum(1 for tri in itertools.combinations(block, t) if tri in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            blocks.append(list(best_block))
            for tri in itertools.combinations(best_block, t):
                if tri in uncovered:
                    uncovered.remove(tri)
        best_blocks = remove_redundant(blocks)

    return best_blocks
