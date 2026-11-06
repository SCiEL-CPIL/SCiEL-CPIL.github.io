"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85  # 85s safety margin for 90s limit
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute LCM for all pairs
    lcm_table = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_table[(i, j)] = i * j // gcd(i, j)
    
    # Precompute all bad triples: (a, b, c) where lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[(a, b)]
            for c in range(b + 1, N + 1):
                if lcm_table[(a, c)] == lab and lcm_table[(b, c)] == lab:
                    bad_triples.add((a, b, c))
    
    # Precompute: for each x, the list of (y, z) pairs such that (x,y,z) is a bad triple
    conflict_pairs = {x: [] for x in range(1, N + 1)}
    for (a, b, c) in bad_triples:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))
    
    # Precompute bitmask for each x: bitmask[y | z] for each conflict pair
    # This allows O(1) validity check: (S_mask & conflict_mask[x]) == 0
    # But we need to check that BOTH y AND z are in S, so we need a different approach.
    # Instead, precompute for each x the set of conflict pairs, and use a different check.
    
    # For fast can_add: for each x, store list of (y, z). Check if any (y,z) both in S.
    # Use a bitset representation for S for faster membership.
    
    def can_add(x, S_mask):
        """Check if x can be added given S_mask (bitset as int)."""
        for y, z in conflict_pairs[x]:
            if (S_mask >> (y - 1)) & 1 and (S_mask >> (z - 1)) & 1:
                return False
        return True
    
    def mask_to_set(mask):
        return {i + 1 for i in range(N) if (mask >> i) & 1}
    
    def set_to_mask(S):
        mask = 0
        for x in S:
            mask |= (1 << (x - 1))
        return mask
    
    def greedy_fill(S_mask):
        """Try to add any remaining elements that can be added. Returns new mask."""
        improved = True
        while improved:
            improved = False
            for x in elements:
                if not (S_mask >> (x - 1)) & 1 and can_add(x, S_mask):
                    S_mask |= (1 << (x - 1))
                    improved = True
        return S_mask
    
    def simulated_annealing(start_mask, iterations, start_temp, end_temp):
        """Simulated annealing to maximize set size."""
        current_mask = start_mask
        current_size = bin(current_mask).count('1')
        best_mask = current_mask
        best_size = current_size
        
        for i in range(iterations):
            if time.time() > deadline - 5:
                break
            
            temp = start_temp * (end_temp / start_temp) ** (i / max(iterations, 1))
            
            # Random move: add or remove a random element
            x = random.choice(elements)
            bit = 1 << (x - 1)
            
            if current_mask & bit:
                # Remove x
                new_mask = current_mask & ~bit
                new_size = current_size - 1
            else:
                # Try to add x
                if not can_add(x, current_mask):
                    continue
                new_mask = current_mask | bit
                new_size = current_size + 1
            
            delta = new_size - current_size
            if delta >= 0 or random.random() < pow(2.718281828, delta / max(temp, 1e-10)):
                current_mask = new_mask
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
        
        return best_mask, best_size
    
    best = set()
    best_size = 0
    best_mask = 0
    
    # Start with the odd numbers as a good base (they tend to avoid lcm conflicts)
    odd_base = {x for x in elements if x % 2 == 1}
    base_mask = set_to_mask(odd_base)
    base_mask = greedy_fill(base_mask)
    base_size = bin(base_mask).count('1')
    
    if base_size > best_size:
        best_size = base_size
        best_mask = base_mask
    
    # Multiple restarts with different strategies
    num_starts = 1500
    for start in range(num_starts):
        if time.time() > deadline - 10:
            break
        
        # Choose initial strategy
        strategy = random.random()
        
        if strategy < 0.3:
            # Start from odd base
            current_mask = base_mask
        elif strategy < 0.6:
            # Random greedy
            order = elements[:]
            random.shuffle(order)
            current_mask = 0
            for x in order:
                if can_add(x, current_mask):
                    current_mask |= (1 << (x - 1))
        else:
            # Random subset of size ~60-70
            k = random.randint(60, 70)
            subset = random.sample(elements, k)
            current_mask = 0
            for x in subset:
                if can_add(x, current_mask):
                    current_mask |= (1 << (x - 1))
        
        # Greedy fill
        current_mask = greedy_fill(current_mask)
        
        # Simulated annealing
        sa_iters = 500
        start_temp = 5.0
        end_temp = 0.1
        sa_mask, sa_size = simulated_annealing(current_mask, sa_iters, start_temp, end_temp)
        
        if sa_size > best_size:
            best_size = sa_size
            best_mask = sa_mask
    
    # Final greedy fill on best
    if time.time() < deadline - 2:
        best_mask = greedy_fill(best_mask)
        best_size = bin(best_mask).count('1')
    
    best = mask_to_set(best_mask)
    
    if not best:
        best = {1}
    
    return list(best)
# EVOLVE-BLOCK-END
