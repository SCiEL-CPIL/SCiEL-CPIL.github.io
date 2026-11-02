"""Covering design C(24,7,3) via potential-based greedy with aggressive local search."""
import itertools
import time
import random
from collections import defaultdict

def _prune_redundant_blocks(blocks, v, t):
    """
    Removes blocks that do not cover any unique t-subset.
    Returns the pruned list of blocks.
    """
    if not blocks:
        return blocks
    
    # Precompute t-subsets for each block
    block_tsets = []
    for b in blocks:
        tsets = [frozenset(c) for c in itertools.combinations(b, t)]
        block_tsets.append(tsets)
        
    # Count how many blocks cover each t-set
    tset_to_blocks = defaultdict(list)
    for i, tsets in enumerate(block_tsets):
        for ts in tsets:
            tset_to_blocks[ts].append(i)
            
    to_remove = set()
    for i, tsets in enumerate(block_tsets):
        is_redundant = True
        for ts in tsets:
            if len(tset_to_blocks[ts]) == 1:
                is_redundant = False
                break
        if is_redundant:
            to_remove.add(i)
            
    if to_remove:
        new_blocks = [b for i, b in enumerate(blocks) if i not in to_remove]
        return new_blocks
    else:
        return blocks

def _local_search(blocks, v, k, t, max_iter=50, deadline=None):
    """
    Try to improve the covering by replacing blocks or removing redundant ones.
    """
    if not blocks:
        return blocks
    
    best_blocks = list(blocks)
    best_count = len(best_blocks)
    
    def get_tsets(block):
        return [frozenset(c) for c in itertools.combinations(block, t)]
    
    def build_coverage(block_list):
        cov = defaultdict(int)
        for b in block_list:
            for ts in get_tsets(b):
                cov[ts] += 1
        return cov
    
    for iteration in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        cov = build_coverage(best_blocks)
        
        for i in range(len(best_blocks)):
            if deadline and time.time() > deadline:
                break
            
            candidate = best_blocks[i]
            candidate_tsets = get_tsets(candidate)
            
            uncovered_by_removal = []
            for ts in candidate_tsets:
                if cov[ts] == 1:
                    uncovered_by_removal.append(ts)
            
            if not uncovered_by_removal:
                best_blocks = [b for j, b in enumerate(best_blocks) if j != i]
                best_count = len(best_blocks)
                break
            
            best_replacement = None
            best_gain = -1
            
            for _ in range(30):
                if deadline and time.time() > deadline:
                    break
                seed_ts = random.choice(uncovered_by_removal)
                block = list(seed_ts)
                while len(block) < k:
                    x = random.randint(0, v - 1)
                    if x not in block:
                        block.append(x)
                block.sort()
                
                block_tsets = set(get_tsets(block))
                covers_all = all(ts in block_tsets for ts in uncovered_by_removal)
                
                if covers_all:
                    additional = len(block_tsets - set(candidate_tsets))
                    if additional > best_gain:
                        best_gain = additional
                        best_replacement = block
            
            if best_replacement is not None:
                best_blocks[i] = best_replacement
        else:
            break
    
    return best_blocks

def _potential_greedy(v, k, t, deadline):
    """
    Greedy construction using a potential score that penalizes redundancy.
    Score for adding x to current partial block B:
      sum over (t-1)-subsets R of B of 1/(cov({x}∪R))^2
    This is a smooth surrogate that drives coverage toward uniformity.
    """
    all_tsets = set(frozenset(c) for c in itertools.combinations(range(v), t))
    uncovered = set(all_tsets)
    coverage_count = defaultdict(int)  # tset -> number of blocks covering it
    blocks = []
    
    while uncovered and time.time() < deadline - 20:
        # Pick a seed t-set from uncovered, weighted toward those with lowest coverage
        # (all uncovered have coverage 0, so just pick randomly)
        seed_tset = random.choice(list(uncovered))
        block = list(seed_tset)
        
        while len(block) < k and time.time() < deadline - 20:
            block_set = set(block)
            best_x = -1
            best_score = -1.0
            best_candidates = []
            
            for x in range(v):
                if x in block_set:
                    continue
                # Compute potential score: sum of 1/cov^2 for each new t-set formed
                score = 0.0
                for R in itertools.combinations(block, t - 1):
                    ts = frozenset(list(R) + [x])
                    if ts in uncovered:
                        # cov is 0 for uncovered, so 1/0^2 is infinity -> use large value
                        score += 1.0  # uncovered gets full weight
                    else:
                        c = coverage_count.get(ts, 0)
                        if c > 0:
                            score += 1.0 / (c * c)
                        else:
                            score += 1.0
                
                if score > best_score:
                    best_score = score
                    best_candidates = [x]
                elif score == best_score:
                    best_candidates.append(x)
            
            if not best_candidates:
                remaining = [x for x in range(v) if x not in block_set]
                if remaining:
                    best_x = random.choice(remaining)
                else:
                    break
            else:
                best_x = random.choice(best_candidates)
            
            block.append(best_x)
        
        if len(block) == k:
            block.sort()
            # Update coverage
            for c in itertools.combinations(block, t):
                ts = frozenset(c)
                coverage_count[ts] += 1
                uncovered.discard(ts)
            blocks.append(block)
    
    # Fill remaining uncovered t-sets
    while uncovered and time.time() < deadline - 5:
        ts = random.choice(list(uncovered))
        block = list(ts)
        while len(block) < k:
            x = random.randint(0, v - 1)
            if x not in block:
                block.append(x)
        block.sort()
        if block not in blocks:
            blocks.append(block)
            for c in itertools.combinations(block, t):
                ts2 = frozenset(c)
                coverage_count[ts2] += 1
                uncovered.discard(ts2)
    
    return blocks if not uncovered else None

def _standard_greedy(v, k, t, deadline):
    """
    Standard greedy: pick x that maximizes number of newly covered t-sets.
    """
    all_tsets = set(frozenset(c) for c in itertools.combinations(range(v), t))
    uncovered = set(all_tsets)
    blocks = []
    
    while uncovered and time.time() < deadline - 20:
        seed_tset = random.choice(list(uncovered))
        block = list(seed_tset)
        
        while len(block) < k and time.time() < deadline - 20:
            block_set = set(block)
            best_x = -1
            best_gain = -1
            best_candidates = []
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for R in itertools.combinations(block, t - 1):
                    ts = frozenset(list(R) + [x])
                    if ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [x]
                elif gain == best_gain:
                    best_candidates.append(x)
            
            if not best_candidates:
                remaining = [x for x in range(v) if x not in block_set]
                if remaining:
                    best_x = random.choice(remaining)
                else:
                    break
            else:
                best_x = random.choice(best_candidates)
            
            block.append(best_x)
        
        if len(block) == k:
            block.sort()
            for c in itertools.combinations(block, t):
                uncovered.discard(frozenset(c))
            blocks.append(block)
    
    # Fill remaining
    while uncovered and time.time() < deadline - 5:
        ts = random.choice(list(uncovered))
        block = list(ts)
        while len(block) < k:
            x = random.randint(0, v - 1)
            if x not in block:
                block.append(x)
        block.sort()
        if block not in blocks:
            blocks.append(block)
            for c in itertools.combinations(block, t):
                uncovered.discard(frozenset(c))
    
    return blocks if not uncovered else None

def construct(v, k, t):
    deadline = time.time() + 115
    best_blocks = None
    best_count = float('inf')
    
    # Multi-start with both potential-based and standard greedy
    while time.time() < deadline - 30:
        # Alternate between strategies
        if random.random() < 0.5:
            blocks = _potential_greedy(v, k, t, deadline)
        else:
            blocks = _standard_greedy(v, k, t, deadline)
        
        if blocks is None:
            continue
        
        # Prune redundant blocks
        pruned = _prune_redundant_blocks(blocks, v, t)
        prev_len = len(pruned)
        while len(pruned) < prev_len and time.time() < deadline - 15:
            prev_len = len(pruned)
            pruned = _prune_redundant_blocks(pruned, v, t)
        
        # Local search
        if time.time() < deadline - 15:
            refined = _local_search(pruned, v, k, t, max_iter=40, deadline=deadline - 5)
            if len(refined) < len(pruned):
                pruned = refined
        
        if len(pruned) < best_count:
            best_count = len(pruned)
            best_blocks = pruned
            
        if time.time() > deadline - 30:
            break
    
    if best_blocks is None:
        # Fallback
        blocks = _standard_greedy(v, k, t, deadline)
        if blocks is not None:
            best_blocks = blocks
        else:
            # Last resort: cover each t-set with a random block
            all_tsets = list(itertools.combinations(range(v), t))
            best_blocks = []
            for ts in all_tsets:
                block = list(ts)
                while len(block) < k:
                    x = random.randint(0, v - 1)
                    if x not in block:
                        block.append(x)
                block.sort()
                if block not in best_blocks:
                    best_blocks.append(block)
    
    return best_blocks
