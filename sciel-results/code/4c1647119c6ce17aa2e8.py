import time
import random
import itertools
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    # Precompute all t-subsets and map to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)

    def greedy_construct():
        """Standard greedy: pick block covering most uncovered triples."""
        uncovered = set(range(num_tsubs))
        blocks = []
        # For speed, maintain count of how many uncovered triples each potential block covers
        # We'll use a simpler approach: for each candidate block, count covered uncovered triples
        
        while uncovered:
            best_block = None
            best_count = -1
            # Sample candidate blocks from uncovered triples
            candidates = set()
            for idx in list(uncovered)[:50]:
                triple = all_tsubs[idx]
                # Generate blocks containing this triple
                remaining = [x for x in range(v) if x not in triple]
                for combo in itertools.combinations(remaining, k - t):
                    block = tuple(sorted(triple + combo))
                    candidates.add(block)
            
            if not candidates:
                break
                
            for block in candidates:
                count = 0
                for triple in itertools.combinations(block, t):
                    idx = tsub_to_idx[triple]
                    if idx in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_block = block
            
            if best_block is None:
                break
                
            blocks.append(list(best_block))
            for triple in itertools.combinations(best_block, t):
                idx = tsub_to_idx[triple]
                uncovered.discard(idx)
            
            if time.time() > deadline - 5:
                break
        
        return blocks

    def local_search(blocks, max_iter=500):
        """Local search: try to remove redundant blocks, then try swaps."""
        blocks = [list(b) for b in blocks]
        
        # Helper: check if a block is redundant
        def is_redundant(block_idx):
            block = blocks[block_idx]
            # For each triple in this block, check if covered by another block
            for triple in itertools.combinations(block, t):
                idx = tsub_to_idx[triple]
                covered_by_other = False
                for j, other_block in enumerate(blocks):
                    if j == block_idx:
                        continue
                    if set(triple).issubset(set(other_block)):
                        covered_by_other = True
                        break
                if not covered_by_other:
                    return False
            return True
        
        # Remove redundant blocks
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if is_redundant(i):
                    blocks.pop(i)
                    changed = True
        
        # Try to replace blocks with better ones
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            # Pick a random block and try to find a replacement that covers more unique triples
            i = random.randint(0, len(blocks) - 1)
            old_block = blocks[i]
            old_triples = set(itertools.combinations(old_block, t))
            
            # Find triples only covered by this block
            unique_triples = set()
            for triple in old_triples:
                idx = tsub_to_idx[triple]
                covered_by_other = False
                for j, other_block in enumerate(blocks):
                    if j == i:
                        continue
                    if set(triple).issubset(set(other_block)):
                        covered_by_other = True
                        break
                if not covered_by_other:
                    unique_triples.add(triple)
            
            if not unique_triples:
                continue  # Block is redundant, will be removed
            
            # Try to find a new block that covers all unique_triples and maximizes additional coverage
            best_new_block = None
            best_score = -1
            
            # Sample candidate blocks
            for _ in range(20):
                # Pick a random triple from unique_triples
                triple = random.choice(list(unique_triples))
                remaining = [x for x in range(v) if x not in triple]
                combo = random.sample(remaining, k - t)
                candidate = tuple(sorted(triple + combo))
                
                # Check if it covers all unique_triples
                if not all(set(u).issubset(set(candidate)) for u in unique_triples):
                    continue
                
                # Score: number of additional triples covered
                score = 0
                for triple2 in itertools.combinations(candidate, t):
                    if triple2 not in old_triples:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_new_block = candidate
            
            if best_new_block and best_score > 0:
                blocks[i] = list(best_new_block)
        
        return blocks

    # Phase 1: Greedy construction
    blocks = greedy_construct()
    
    # Phase 2: Local search refinement
    if time.time() < deadline - 10:
        blocks = local_search(blocks)
    
    # Phase 3: More local search passes if time permits
    while time.time() < deadline - 2:
        new_blocks = local_search(blocks, max_iter=200)
        if len(new_blocks) < len(blocks):
            blocks = new_blocks
        else:
            break
    
    return blocks
