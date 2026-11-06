import time
import math
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # 1. Precompute GCD matrix
    gcd_mat = [[1] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # 2. Identify Bad Triples and Build Conflict Masks
    # conflict_mask[u][v] is a bitmask of w such that {u,v,w} is a bad triple
    # We use 0-indexed bits for numbers 1..128, so bit (x-1) corresponds to number x.
    # Bitmask size 128 bits. Python ints handle this easily.
    
    conflict_mask = [[0] * (N + 1) for _ in range(N + 1)]
    bad_triples_count = 0
    
    # Iterate all triples
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_mat[a][b]
            for c in range(b + 1, N + 1):
                if gcd_mat[a][c] == gab and gcd_mat[b][c] == gab:
                    # {a, b, c} is a bad triple
                    bad_triples_count += 1
                    # Update conflict masks
                    # If we have a and b, we cannot have c.
                    # If we have a and c, we cannot have b.
                    # If we have b and c, we cannot have a.
                    conflict_mask[a][b] |= (1 << (c - 1))
                    conflict_mask[b][a] |= (1 << (c - 1))
                    
                    conflict_mask[a][c] |= (1 << (b - 1))
                    conflict_mask[c][a] |= (1 << (b - 1))
                    
                    conflict_mask[b][c] |= (1 << (a - 1))
                    conflict_mask[c][b] |= (1 << (a - 1))
                    
    # 3. Precompute "Total Conflict Degree" for ordering
    # A number x is "conflict-heavy" if it participates in many bad triples.
    # We can estimate this by summing popcount of conflict_mask[x][y] for all y?
    # Or just count how many bad triples contain x.
    
    triple_count_per_num = [0] * (N + 1)
    # Re-iterate or derive from masks? 
    # Easier: just count during triple iteration? We already did.
    # Let's just use a simple heuristic: numbers with many small factors are likely more constrained?
    # Actually, let's just sort by index or random for now, or by "potential".
    
    # 4. Branch and Bound Search
    
    # State:
    # available: int bitmask of numbers that can still be added
    # current_set: list of numbers in the current solution
    # best_size: global variable for best solution found
    
    best_size = 0
    best_set = []
    
    # Initial available: all numbers
    initial_available = (1 << N) - 1
    
    # Ordering: Sort numbers by some heuristic.
    # Let's try sorting by "number of conflicts" descending.
    # conflict_degree[x] = sum of popcount(conflict_mask[x][y]) for all y? No, that's not right.
    # Let's just use a static order: 1..128 or random.
    # Better: Order by "constraint density".
    # For each x, let deg(x) be the number of y such that conflict_mask[x][y] != 0.
    deg = [0] * (N + 1)
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            if conflict_mask[x][y]:
                deg[x] += 1
                
    # Sort numbers: high degree first?
    # If a number is in many bad triples, deciding it early might prune more.
    search_order = sorted(numbers, key=lambda x: deg[x], reverse=True)
    
    # Map number to index in search_order for quick lookup?
    # We need to pick the "next" number to branch on from `available`.
    # We can precompute the list of numbers in search_order.
    # At each step, find the first number in search_order that is in `available`.
    
    def get_next_candidate(available):
        for x in search_order:
            if available & (1 << (x - 1)):
                return x
        return None

    def search(available, current_set, current_size):
        nonlocal best_size, best_set
        
        # Time check
        if time.time() > deadline:
            return
        
        # Pruning
        # Max possible size = current_size + popcount(available)
        if current_size + bin(available).count('1') <= best_size:
            return
            
        # Base case: no more candidates
        if available == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return
            
        # Heuristic: Try to add as many as possible greedily first to get a good lower bound?
        # Or just branch.
        
        x = get_next_candidate(available)
        if x is None:
            return
            
        # Branch 1: Exclude x
        new_available_excl = available & ~(1 << (x - 1))
        search(new_available_excl, current_set, current_size)
        
        # Branch 2: Include x
        # If we include x, we must remove from available any y that forms a bad triple with x and some z in current_set.
        # Also, we must add x to current_set.
        
        # Calculate forbidden mask
        forbidden = 0
        for y in current_set:
            forbidden |= conflict_mask[x][y]
            
        # Also, x itself is removed from available
        new_available_incl = available & ~(1 << (x - 1)) & ~forbidden
        
        # Additional check:
        # Are there any pairs (y, z) in current_set such that x, y, z is a bad triple?
        # If so, we cannot include x.
        # Wait, if x, y, z is a bad triple, then z is in conflict_mask[x][y].
        # So if current_set contains y and z, then z is in forbidden?
        # No, forbidden is calculated based on current_set.
        # If y is in current_set, and z is in current_set, and {x,y,z} is bad,
        # then z is in conflict_mask[x][y].
        # So z is in `forbidden`.
        # But z is ALREADY in current_set.
        # We are checking if we can ADD x.
        # If adding x creates a bad triple with existing elements, it's invalid.
        # The condition "z is in conflict_mask[x][y]" means "if x and y are chosen, z cannot be chosen".
        # If z is already chosen, then we CANNOT choose x (if y is also chosen).
        # So we need to check: for any y in current_set, if (forbidden & (1 << (y-1)))? No.
        # We need to check if there exist y, z in current_set such that {x,y,z} is bad.
        # This is equivalent to: Is there any y in current_set such that conflict_mask[x][y] intersects current_set_mask?
        
        current_set_mask = 0
        for y in current_set:
            current_set_mask |= (1 << (y - 1))
            
        # Check validity of adding x
        valid = True
        for y in current_set:
            # If any element in current_set is in conflict_mask[x][y], it's a conflict.
            # Wait, conflict_mask[x][y] contains z such that {x,y,z} is bad.
            # If z is in current_set, then we have a bad triple {x,y,z}.
            if conflict_mask[x][y] & current_set_mask:
                valid = False
                break
                
        if valid:
            new_set = current_set + [x]
            search(new_available_incl, new_set, current_size + 1)

    # Start search
    search(initial_available, [], 0)
    
    # If the search is too slow or doesn't find a good solution, fallback to heuristic
    if best_size < 50: # Threshold
        # Simple greedy fallback
        S = set()
        S_bool = [False] * (N + 1)
        for x in numbers:
            # Check if x can be added
            ok = True
            mask_x = 1 << (x - 1)
            for y in S:
                if conflict_mask[x][y] & mask_x: # This check is wrong for greedy, need to check against all pairs in S
                    pass
            # Re-implement simple check
            can_add = True
            for y in S:
                for z in S:
                    if y == z: continue
                    if conflict_mask[x][y] & (1 << (z - 1)):
                        can_add = False
                        break
                if not can_add: break
            if can_add:
                S.add(x)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)

    return best_set

# EVOLVE-BLOCK-END
