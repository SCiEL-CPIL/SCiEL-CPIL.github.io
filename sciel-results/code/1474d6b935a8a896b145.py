import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Tighter safety margin
    
    n = 128
    
    # Precompute LCM table for O(1) lookups
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a, n + 1):
            lcm_table[a][b] = a // gcd(a, b) * b
            lcm_table[b][a] = lcm_table[a][b]
    
    # Find all bad triples and build conflict structure
    # conflict_mask[x]: bitmask of elements that form a bad triple with x and some other element
    # For element x, we need to track which pairs (u,v) conflict with it
    
    # Better approach: For each element x, precompute which elements cannot be in the set
    # along with x. But the constraint is 3-uniform, so we need to track pairs.
    
    # Let's use a different representation:
    # For each element x, conflict_with[x] = set of (u, v) pairs such that {x, u, v} is a bad triple
    # When checking if we can add x to a set S, we check if any (u, v) in conflict_with[x] has both u,v in S
    
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L_ab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == L_ab and lcm_table[b][c] == L_ab:
                    bad_triples.append((a, b, c))
                    
    # Build conflict structure: for each x, store pairs that conflict with x
    conflict_pairs = [set() for _ in range(n + 1)]
    for a, b, c in bad_triples:
        conflict_pairs[a].add((b, c))
        conflict_pairs[b].add((a, c))
        conflict_pairs[c].add((a, b))
    
    # Precompute degrees
    degrees = [len(conflict_pairs[i]) for i in range(n + 1)]
    
    # Bitmask helpers
    def popcount(mask):
        return bin(mask).count('1')
    
    def can_add(x, mask):
        """Check if element x can be added to the set represented by mask."""
        for u, v in conflict_pairs[x]:
            if (mask >> (u - 1)) & 1 and (mask >> (v - 1)) & 1:
                return False
        return True
    
    # Branch and bound for maximum independent set
    # State: current_mask, remaining candidates, current size
    best_size = 0
    best_mask = 0
    
    def branch_and_bound(mask, candidates, current_size, depth=0):
        nonlocal best_size, best_mask
        
        if time.time() > deadline - 3:
            return
        
        # Upper bound: current_size + len(candidates)
        if current_size + len(candidates) <= best_size:
            return
        
        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_mask = mask
            return
        
        # Choose next element to branch on
        # Use the element with highest degree in the remaining subgraph
        # For simplicity, just pick the first candidate
        x = candidates[0]
        remaining = candidates[1:]
        
        # Option 1: Don't include x
        branch_and_bound(mask, remaining, current_size, depth + 1)
        
        # Option 2: Include x (if possible)
        if can_add(x, mask):
            new_mask = mask | (1 << (x - 1))
            # Filter candidates that can still be added
            new_candidates = [y for y in remaining if can_add(y, new_mask)]
            branch_and_bound(new_mask, new_candidates, current_size + 1, depth + 1)
    
    # Multi-start greedy to get good initial solutions
    def greedy(order, mask=0):
        selected = mask
        for x in order:
            if not (selected >> (x - 1)) & 1:
                if can_add(x, selected):
                    selected |= (1 << (x - 1))
        return selected
    
    def local_search(mask):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Greedy add pass
            for x in range(1, n + 1):
                if not (mask >> (x - 1)) & 1:
                    if can_add(x, mask):
                        mask |= (1 << (x - 1))
                        improved = True
            
            if not improved:
                # 2. Perturbation: Remove one, fill greedily
                s_list = [i for i in range(1, n + 1) if (mask >> (i - 1)) & 1]
                if not s_list:
                    break
                
                for _ in range(20):
                    if time.time() > deadline - 5:
                        break
                    x = random.choice(s_list)
                    mask_no_x = mask & ~(1 << (x - 1))
                    new_mask = mask_no_x
                    for y in range(1, n + 1):
                        if not (new_mask >> (y - 1)) & 1:
                            if can_add(y, new_mask):
                                new_mask |= (1 << (y - 1))
                    
                    if popcount(new_mask) > popcount(mask):
                        mask = new_mask
                        improved = True
                        break
                        
        return mask
    
    # Phase 1: Multi-start greedy + local search to get good initial solution
    num_restarts = 0
    while time.time() < deadline - 10:
        num_restarts += 1
        
        r = random.random()
        if r < 0.4:
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif r < 0.7:
            order = sorted(range(1, n + 1), key=lambda x: degrees[x])
        else:
            order = sorted(range(1, n + 1), key=lambda x: -degrees[x])
            
        mask = greedy(order)
        mask = local_search(mask)
        
        size = popcount(mask)
        if size > best_size:
            best_size = size
            best_mask = mask
    
    # Phase 2: Try branch-and-bound from the best solution found
    # This is expensive, so only do it if we have time
    if time.time() < deadline - 15:
        # Start from the best mask and try to improve via B&B
        # We'll use a limited B&B that explores additions
        initial_candidates = [x for x in range(1, n + 1) if not (best_mask >> (x - 1)) & 1]
        
        # Sort candidates by degree (lower degree first - easier to add)
        initial_candidates.sort(key=lambda x: degrees[x])
        
        # Limited B&B: only explore up to depth 3 additions
        def limited_bb(mask, candidates, current_size, max_depth=3):
            nonlocal best_size, best_mask
            
            if time.time() > deadline - 5:
                return
            
            if current_size + len(candidates) <= best_size:
                return
            
            if not candidates or max_depth <= 0:
                if current_size > best_size:
                    best_size = current_size
                    best_mask = mask
                return
            
            # Try adding each candidate
            for i, x in enumerate(candidates):
                if time.time() > deadline - 5:
                    return
                
                if can_add(x, mask):
                    new_mask = mask | (1 << (x - 1))
                    new_candidates = [y for y in candidates[i+1:] if can_add(y, new_mask)]
                    limited_bb(new_mask, new_candidates, current_size + 1, max_depth - 1)
        
        limited_bb(best_mask, initial_candidates, best_size, max_depth=4)
    
    # Final polish
    if time.time() < deadline - 2:
        best_mask = local_search(best_mask)
    
    result = [i for i in range(1, n + 1) if (best_mask >> (i - 1)) & 1]
    
    if not result:
        result = [1]
        
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Result size: {len(res)}")
    print(res)
