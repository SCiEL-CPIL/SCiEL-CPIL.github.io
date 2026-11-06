import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    
    # Generate candidates: all vectors with weight <= 5
    # Weight 0: 1
    # Weight 1: 18
    # Weight 2: 153
    # Weight 3: 816
    # Weight 4: 3060
    # Weight 5: 8568
    # Total: 12616
    
    candidates = []
    
    # Helper to generate combinations
    def gen_combinations(n, k):
        # Simple recursive or iterative generation
        # For small n, k, this is fast
        def combine(prefix, start, remaining):
            if remaining == 0:
                candidates.append(prefix)
                return
            for i in range(start, n - remaining + 1):
                combine(prefix | (1 << i), i + 1, remaining - 1)
        
        combine(0, 0, k)

    for w in range(6):
        gen_combinations(N, w)
        
    # Sort by weight
    candidates.sort(key=lambda x: x.bit_count())
    
    best_code = []
    best_size = 0
    
    start_time = time.time()
    deadline = 85.0 # 5s margin
    
    # Pre-compute weight of each candidate to avoid re-calculation in shuffle logic?
    # We can just store (weight, candidate) pairs or similar.
    # But sorting is done. We just need to shuffle within groups.
    
    # To speed up shuffling, we can pre-identify the groups
    groups = []
    current_weight = -1
    current_group = []
    for c in candidates:
        w = c.bit_count()
        if w != current_weight:
            if current_weight != -1:
                groups.append(current_group)
            current_weight = w
            current_group = []
        current_group.append(c)
    if current_weight != -1:
        groups.append(current_group)
        
    # groups is a list of lists of candidates, sorted by weight
    
    while time.time() - start_time < deadline:
        # Shuffle each group
        shuffled_candidates = []
        for g in groups:
            # Copy and shuffle to avoid modifying original?
            # random.shuffle modifies in place.
            # We can just shuffle the group list itself, then extend.
            # But we need to restore it? No, we can just shuffle a copy.
            # Or shuffle the group, extend, then... we don't care about the original order for next iteration?
            # Actually, we do want the original group to be unshuffled for the next shuffle?
            # No, shuffling a shuffled list is still a random permutation.
            # So we can just shuffle the groups in place.
            random.shuffle(g)
            shuffled_candidates.extend(g)
            
        # Greedy selection
        code = []
        for c in shuffled_candidates:
            valid = True
            for x in code:
                if (c ^ x).bit_count() < MIN_DIST:
                    valid = False
                    break
            if valid:
                code.append(c)
                
        if len(code) > best_size:
            best_size = len(code)
            best_code = code
            
    return best_code
