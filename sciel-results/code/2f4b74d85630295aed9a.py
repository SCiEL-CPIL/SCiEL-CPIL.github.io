import time
import random
from itertools import combinations

def construct():
    points = range(12)
    all_blocks = list(combinations(points, 6))
    n_blocks = len(all_blocks)
    
    # Map block tuple to index
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Complementary pairs
    # A block b and its complement c = tuple(x for x in points if x not in b)
    # We want to pick 66 pairs.
    # To avoid double counting, only consider pairs where b < c lexicographically?
    # Or just ensure we don't pick the same pair twice.
    
    pairs = []
    seen = set()
    for b in all_blocks:
        c = tuple(x for x in points if x not in b)
        pair = (b, c) if b < c else (c, b)
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
            
    # Sort pairs lexicographically
    pairs.sort()
    
    # Greedy selection of pairs
    used_5subs = set()
    selected_blocks = []
    
    # Precompute 5-subsets for each block for speed
    block_5subs = {}
    for b in all_blocks:
        block_5subs[b] = list(combinations(b, 5))
        
    for b1, b2 in pairs:
        subs1 = block_5subs[b1]
        subs2 = block_5subs[b2]
        
        # Check conflict
        conflict = False
        for s in subs1:
            if s in used_5subs:
                conflict = True
                break
        if not conflict:
            for s in subs2:
                if s in used_5subs:
                    conflict = True
                    break
                    
        if not conflict:
            # Select both
            selected_blocks.append(b1)
            selected_blocks.append(b2)
            for s in subs1:
                used_5subs.add(s)
            for s in subs2:
                used_5subs.add(s)
                
            if len(selected_blocks) == 132:
                return [list(b) for b in selected_blocks]
                
    # If greedy failed (unlikely for this specific design with lex order),
    # fallback to the previous random method or a backtracking search.
    # For safety, implement a quick backtracking if greedy fails.
    
    # Backtracking search
    # State: index in pairs, used_5subs, selected_blocks
    # This might be slow if greedy fails, but greedy should work.
    # Let's just return what we have if greedy fails, or use the random method as fallback.
    
    # Fallback to random greedy (from current program) if deterministic fails
    start_time = time.time()
    deadline = start_time + 110.0
    best_count = len(selected_blocks)
    best_blocks = selected_blocks[:]
    
    rng = random.Random(42)
    all_indices = list(range(n_blocks))
    
    # Precompute neighbors for random fallback
    block_sets = [set(b) for b in all_blocks]
    neighbors = [[] for _ in range(n_blocks)]
    for i in range(n_blocks):
        for j in range(i + 1, n_blocks):
            if len(block_sets[i] & block_sets[j]) == 5:
                neighbors[i].append(j)
                neighbors[j].append(i)

    while time.time() < deadline:
        avail = all_indices.copy()
        current_blocks = []
        
        while avail:
            idx = rng.randrange(len(avail))
            block_idx = avail.pop(idx)
            current_blocks.append(all_blocks[block_idx])
            
            for nb in neighbors[block_idx]:
                if nb in avail:
                    avail.remove(nb)
                    
            if len(current_blocks) == 132:
                return [list(b) for b in current_blocks]
                
        if len(current_blocks) > best_count:
            best_count = len(current_blocks)
            best_blocks = [list(b) for b in current_blocks]
            
    return best_blocks
