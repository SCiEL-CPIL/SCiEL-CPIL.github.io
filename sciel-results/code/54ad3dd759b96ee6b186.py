import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer for safety
    
    # Pre-compute all t-subsets
    all_t_subsets = set(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_score = float('inf')
    
    for run in range(100):
        if time.time() > deadline:
            break
            
        uncovered = set(all_t_subsets)
        blocks = []
        
        # Greedy construction
        while uncovered:
            if time.time() > deadline:
                break
            
            target = random.choice(list(uncovered))
            remaining_points = [p for p in range(v) if p not in target]
            
            best_block = None
            best_gain = -1
            
            for combo in itertools.combinations(remaining_points, k - t):
                candidate_block = tuple(sorted(target + combo))
                
                gain = 0
                for sub in itertools.combinations(candidate_block, t):
                    if sub in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = candidate_block
            
            if best_block is None:
                break
            
            blocks.append(list(best_block))
            
            for sub in itertools.combinations(best_block, t):
                uncovered.discard(sub)
        
        if not uncovered:
            # Post-processing: Remove redundant blocks
            coverage_count = {}
            for b in blocks:
                for sub in itertools.combinations(b, t):
                    coverage_count[sub] = coverage_count.get(sub, 0) + 1
            
            changed = True
            while changed:
                changed = False
                new_blocks = []
                for b in blocks:
                    is_redundant = True
                    for sub in itertools.combinations(b, t):
                        if coverage_count[sub] == 1:
                            is_redundant = False
                            break
                    
                    if is_redundant:
                        changed = True
                        for sub in itertools.combinations(b, t):
                            coverage_count[sub] -= 1
                    else:
                        new_blocks.append(b)
                
                blocks = new_blocks
                
                if time.time() > deadline:
                    break
            
            # Local search: try to replace one block with another that covers more uncovered subsets
            # and allows removal of another block
            local_search_rounds = 20
            for _ in range(local_search_rounds):
                if time.time() > deadline:
                    break
                
                # Rebuild coverage count
                coverage_count = {}
                for b in blocks:
                    for sub in itertools.combinations(b, t):
                        coverage_count[sub] = coverage_count.get(sub, 0) + 1
                
                # Try random block removals and replacements
                improved = False
                for i in range(len(blocks)):
                    if time.time() > deadline:
                        break
                    
                    # Try removing block i
                    candidate_blocks = blocks[:i] + blocks[i+1:]
                    
                    # Check if still valid
                    valid = True
                    for sub in all_t_subsets:
                        if coverage_count[sub] <= 1:
                            # This subset is only covered by block i, so removing it breaks coverage
                            # Check if any other block covers it
                            found = False
                            for b in candidate_blocks:
                                if sub in itertools.combinations(b, t):
                                    found = True
                                    break
                            if not found:
                                valid = False
                                break
                    
                    if valid:
                        blocks = candidate_blocks
                        improved = True
                        break
                    
                    # Try swapping block i with a new block
                    # Find an uncovered subset (relative to blocks without i)
                    temp_coverage = {}
                    for j, b in enumerate(blocks):
                        if j != i:
                            for sub in itertools.combinations(b, t):
                                temp_coverage[sub] = temp_coverage.get(sub, 0) + 1
                    
                    uncovered_subs = [sub for sub in all_t_subsets if temp_coverage.get(sub, 0) == 0]
                    
                    if uncovered_subs:
                        # Try to find a block that covers many uncovered subsets
                        best_new_block = None
                        best_new_gain = 0
                        
                        # Sample some candidate blocks
                        for _ in range(50):
                            target = random.choice(uncovered_subs)
                            remaining_points = [p for p in range(v) if p not in target]
                            
                            for combo in itertools.combinations(remaining_points, k - t):
                                candidate_block = tuple(sorted(target + combo))
                                
                                gain = 0
                                for sub in itertools.combinations(candidate_block, t):
                                    if sub in uncovered_subs:
                                        gain += 1
                                
                                if gain > best_new_gain:
                                    best_new_gain = gain
                                    best_new_block = candidate_block
                        
                        if best_new_block and best_new_gain > 0:
                            # Check if adding this block makes the covering valid
                            new_coverage = dict(temp_coverage)
                            for sub in itertools.combinations(best_new_block, t):
                                new_coverage[sub] = new_coverage.get(sub, 0) + 1
                            
                            valid = all(new_coverage.get(sub, 0) > 0 for sub in all_t_subsets)
                            
                            if valid:
                                blocks = blocks[:i] + [list(best_new_block)] + blocks[i+1:]
                                improved = True
                                break
                
                if not improved:
                    break
            
            if len(blocks) < best_score:
                best_score = len(blocks)
                best_blocks = blocks.copy()
                
    if best_blocks is None:
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            target = min(uncovered)
            remaining_points = [p for p in range(v) if p not in target]
            best_block = None
            best_gain = -1
            for combo in itertools.combinations(remaining_points, k - t):
                candidate_block = tuple(sorted(target + combo))
                gain = sum(1 for sub in itertools.combinations(candidate_block, t) if sub in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = candidate_block
            blocks.append(list(best_block))
            for sub in itertools.combinations(best_block, t):
                uncovered.discard(sub)
        best_blocks = blocks

    return best_blocks
