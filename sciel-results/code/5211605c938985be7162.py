import random
import time
from itertools import combinations

def is_sidon(s):
    sums = set()
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            val = s[i] + s[j]
            if val in sums:
                return False
            sums.add(val)
    return True

def find_sidon_4_subsets(s):
    sidon_subsets = []
    for subset in combinations(s, 4):
        if is_sidon(list(subset)):
            sidon_subsets.append(subset)
    return sidon_subsets

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    s = random.sample(range(129), 12)
    s.sort()
    
    best_s = s[:]
    best_count = len(find_sidon_4_subsets(s))
    
    if best_count == 0:
        return s

    while time.time() < deadline:
        sidon_subsets = find_sidon_4_subsets(s)
        if not sidon_subsets:
            return s
        
        if len(sidon_subsets) < best_count:
            best_count = len(sidon_subsets)
            best_s = s[:]
            
        subset = random.choice(sidon_subsets)
        a, b, c, d = subset
        
        candidates = set()
        
        # Helper to add candidates for replacing x with x' in {x, y, z, w}
        def add_candidates_for(x, y, z, w):
            # Sums of {y,z,w}: 2y, y+z, y+w, 2z, z+w, 2w
            target_sums = {2*y, y+z, y+w, 2*z, z+w, 2*w}
            
            # x' + x' = target
            for t in target_sums:
                if t % 2 == 0:
                    candidates.add(t // 2)
            
            # x' + y = target
            for t in target_sums:
                candidates.add(t - y)
            
            # x' + z = target
            for t in target_sums:
                candidates.add(t - z)
            
            # x' + w = target
            for t in target_sums:
                candidates.add(t - w)
        
        add_candidates_for(a, b, c, d)
        add_candidates_for(b, a, c, d)
        add_candidates_for(c, a, b, d)
        add_candidates_for(d, a, b, c)
        
        # Filter valid candidates
        valid_candidates = []
        for val in candidates:
            if 0 <= val <= 128 and val not in s:
                valid_candidates.append(val)
        
        if not valid_candidates:
            # Fallback to random
            for _ in range(20):
                val = random.randint(0, 128)
                if val not in s:
                    valid_candidates.append(val)
            if not valid_candidates:
                continue
        
        # Pick a random valid candidate and the element to replace it
        # We need to know which element we are replacing to revert if needed
        # But since we just pick a val, we can pick a random element from the subset to replace
        # Wait, the candidates were generated for specific replacements.
        # We should pick a (element, candidate) pair.
        
        # Let's re-structure to keep track
        pairs = []
        def add_pairs_for(x, y, z, w):
            target_sums = {2*y, y+z, y+w, 2*z, z+w, 2*w}
            
            # x' + x' = target
            for t in target_sums:
                if t % 2 == 0:
                    pairs.append((x, t // 2))
            
            # x' + y = target
            for t in target_sums:
                pairs.append((x, t - y))
            
            # x' + z = target
            for t in target_sums:
                pairs.append((x, t - z))
            
            # x' + w = target
            for t in target_sums:
                pairs.append((x, t - w))
        
        pairs = []
        add_pairs_for(a, b, c, d)
        add_pairs_for(b, a, c, d)
        add_pairs_for(c, a, b, d)
        add_pairs_for(d, a, b, c)
        
        valid_pairs = []
        for old_val, new_val in pairs:
            if 0 <= new_val <= 128 and new_val not in s:
                valid_pairs.append((old_val, new_val))
        
        if not valid_pairs:
             # Random move
             idx = random.randint(0, 11)
             old_val = s[idx]
             new_val = random.randint(0, 128)
             if new_val in s:
                 continue
             valid_pairs = [(old_val, new_val)]
        
        old_val, new_val = random.choice(valid_pairs)
        
        # Apply
        s.remove(old_val)
        s.append(new_val)
        s.sort()
        
        new_count = len(find_sidon_4_subsets(s))
        if new_count >= len(sidon_subsets):
            # Revert
            s.remove(new_val)
            s.append(old_val)
            s.sort()
        else:
            # Accept
            pass
            
    return best_s
