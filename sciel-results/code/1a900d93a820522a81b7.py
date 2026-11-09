import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = 55.0
    
    N = 128
    full_mask = (1 << (N + 1)) - 2  # bits 1..N set
    
    # Precompute LCMs
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    # Find all bad triples
    bad_triple_masks = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm(a, b)
            for c in range(b + 1, N + 1):
                if lab == lcm(a, c) == lcm(b, c):
                    mask = (1 << a) | (1 << b) | (1 << c)
                    bad_triple_masks.append(mask)
    
    # If no bad triples, return full set
    if not bad_triple_masks:
        return [i for i in range(1, N + 1) if full_mask & (1 << i)]
    
    def is_valid(mask):
        for t in bad_triple_masks:
            if (mask & t) == t:
                return False
        return True
    
    def greedy_remove(full_mask, randomize=False):
        mask = full_mask
        # Cache active triples? No, just recompute.
        while True:
            active_triples = [t for t in bad_triple_masks if (mask & t) == t]
            if not active_triples:
                break
            
            counts = [0] * (N + 1)
            for t in active_triples:
                x = t
                while x:
                    lsb = x & -x
                    idx = lsb.bit_length() - 1
                    counts[idx] += 1
                    x ^= lsb
            
            max_count = 0
            candidates = []
            for i in range(1, N + 1):
                if mask & (1 << i):
                    if counts[i] > max_count:
                        max_count = counts[i]
                        candidates = [i]
                    elif counts[i] == max_count:
                        candidates.append(i)
            
            if not candidates:
                break
                
            if randomize:
                u = random.choice(candidates)
            else:
                # Tie-break: remove the one with smallest index? Or largest?
                # Smallest index is arbitrary. Let's pick max index to remove "large" numbers?
                # Or just first.
                u = candidates[0]
            
            mask &= ~(1 << u)
            
            if time.time() - start_time > deadline - 5:
                break
        
        return mask
    
    def add_back(mask):
        improved = True
        while improved:
            improved = False
            for i in range(1, N + 1):
                if not (mask & (1 << i)):
                    new_mask = mask | (1 << i)
                    if is_valid(new_mask):
                        mask = new_mask
                        improved = True
        return mask
    
    def swap_improve(mask):
        # 1 out, 2 in
        improved = True
        while improved:
            improved = False
            in_mask = [i for i in range(1, N + 1) if mask & (1 << i)]
            not_in_mask = [i for i in range(1, N + 1) if not (mask & (1 << i))]
            
            if not not_in_mask:
                break
                
            # Sort not_in_mask to make it deterministic?
            # Try all pairs
            for i in range(len(not_in_mask)):
                v = not_in_mask[i]
                for j in range(i + 1, len(not_in_mask)):
                    w = not_in_mask[j]
                    # Try to remove any u in in_mask
                    # Optimization: If adding v,w creates a conflict, it must be with some u.
                    # We can check if (mask | (1<<v) | (1<<w)) is valid?
                    # No, we must remove u first.
                    # If (mask | (1<<v) | (1<<w)) is invalid, removing u might fix it.
                    # But we want to find a u such that (mask ^ (1<<u) | (1<<v) | (1<<w)) is valid.
                    
                    # Efficient check:
                    # Let M' = mask | (1<<v) | (1<<w)
                    # We need to remove u from in_mask such that no bad triple is fully in M' \ {u}.
                    # This is equivalent to: For all bad triples T, if T subset of M', then u must be in T.
                    # So u must be in the intersection of all bad triples that are subset of M'.
                    # If there is a bad triple T subset of M' that does NOT contain u, then removing u is not enough.
                    
                    M_prime = mask | (1 << v) | (1 << w)
                    # Find all bad triples contained in M_prime
                    blocking_triples = [t for t in bad_triple_masks if (M_prime & t) == t]
                    
                    if not blocking_triples:
                        # If no blocking triples, we can add v,w without removing anything?
                        # But we are in swap_improve, so we assume we can't just add.
                        # Actually, if no blocking triples, we can just add v and w?
                        # Wait, add_back should have handled that.
                        # If add_back didn't add them, it means they couldn't be added individually.
                        # But together they might be valid? No, if {v} invalid and {w} invalid, {v,w} might be valid?
                        # Yes.
                        # So if no blocking triples in M_prime, we can just set mask = M_prime?
                        # But we are in "1 out, 2 in".
                        # If we can add 2 without removing 1, that's even better.
                        # Let's handle that in add_back? No, add_back adds 1 at a time.
                        # If add_back stuck, it means no single element can be added.
                        # But two elements together might not form a bad triple with the existing set.
                        # So we should check if M_prime is valid.
                        if is_valid(M_prime):
                            mask = M_prime
                            improved = True
                            break
                        # If M_prime is invalid, we MUST remove an element.
                    
                    # We need to find a u in in_mask such that removing u fixes all blocking_triples.
                    # This means u must be in ALL blocking_triples.
                    # Intersection of all blocking_triples.
                    common_intersection = None
                    for t in blocking_triples:
                        if common_intersection is None:
                            common_intersection = t
                        else:
                            common_intersection &= t
                    
                    if common_intersection is None:
                        continue
                        
                    # u must be in common_intersection AND in in_mask
                    # And u must be such that removing it doesn't break anything else?
                    # Removing u only reduces the set, so it can't create new bad triples.
                    # So if u is in the intersection of all blocking triples, removing u will eliminate all blocking triples.
                    # So the resulting set will be valid.
                    
                    # Find a candidate u
                    candidate_u = None
                    x = common_intersection
                    while x:
                        lsb = x & -x
                        idx = lsb.bit_length() - 1
                        if mask & (1 << idx):
                            candidate_u = idx
                            break
                        x ^= lsb
                    
                    if candidate_u is not None:
                        # Remove candidate_u, add v, w
                        mask = (mask & ~(1 << candidate_u)) | (1 << v) | (1 << w)
                        improved = True
                        break
            
            if improved:
                continue
                
        return mask
    
    best_mask = 0
    best_size = 0
    
    # Deterministic run
    m = greedy_remove(full_mask, randomize=False)
    m = add_back(m)
    m = swap_improve(m)
    size = m.bit_count()
    if size > best_size:
        best_size = size
        best_mask = m
        
    # Randomized runs
    random.seed(42)
    for _ in range(50):
        if time.time() - start_time > deadline - 5:
            break
        m = greedy_remove(full_mask, randomize=True)
        m = add_back(m)
        m = swap_improve(m)
        size = m.bit_count()
        if size > best_size:
            best_size = size
            best_mask = m
            
    # Convert mask to list
    result = [i for i in range(1, N + 1) if best_mask & (1 << i)]
    return result
