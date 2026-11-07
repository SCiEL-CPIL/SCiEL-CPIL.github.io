# EVOLVE-BLOCK-START
import itertools
import time

N = 128

def construct():
    # 1. Precompute forbidden sets
    # A forbidden set F is a subset of {1..N} such that if all elements of F are in S,
    # there exist distinct a,b,c in S such that a+b, a+c, b+c are also in S.
    # Specifically, for any distinct a,b,c, if a+b, a+c, b+c are all <= N,
    # then F = {a,b,c, a+b, a+c, b+c} is a forbidden set.
    
    forbidden_sets = []
    # To avoid duplicates, we can just generate all unique sets.
    # Since N is small, we can use a set of frozensets or bitmasks.
    forbidden_masks = set()
    
    # a < b < c
    # a+b <= N, a+c <= N, b+c <= N
    # Since b+c is the largest sum, b+c <= N is the main constraint.
    for c in range(1, N + 1):
        for b in range(1, c):
            if b + c > N:
                break
            for a in range(1, b):
                if a + c > N:
                    break
                # a+b <= N is implied by a+c <= N since b < c
                f = {a, b, c, a + b, a + c, b + c}
                mask = 0
                for v in f:
                    mask |= (1 << v)
                forbidden_masks.add(mask)
                
    forbidden_list = list(forbidden_masks)
    
    # 2. Precompute for each vertex x, the list of masks M such that
    # M is a forbidden set containing x, and M \ {x} is the rest.
    # If we have S and want to add x, we check if any (M \ {x}) is subset of S.
    # If yes, adding x completes M, which is forbidden.
    
    vertex_constraints = [[] for _ in range(N + 1)]
    for mask in forbidden_list:
        # Find bits set in mask
        # We need to identify which x is the "new" one.
        # Actually, for a given mask, any x in the mask could be the one added last.
        # So for each x in mask, we add (mask without x) to vertex_constraints[x].
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            x = lsb.bit_length() - 1
            rest = mask ^ lsb
            vertex_constraints[x].append(rest)
            m ^= lsb
            
    # 3. Branch and Bound
    # State: 
    #   selected_mask: int
    #   available_mask: int (bits that can still be considered)
    #   current_size: int
    
    best_size = 0
    best_mask = 0
    
    # Greedy initialization to get a good lower bound
    # Try to build a set greedily
    def greedy_init():
        S = 0
        # Try adding numbers in descending order? Or random?
        # Descending often works well for sum constraints because large numbers
        # have fewer partners that sum to <= N.
        for x in range(N, 0, -1):
            can_add = True
            for rest in vertex_constraints[x]:
                if (S & rest) == rest:
                    can_add = False
                    break
            if can_add:
                S |= (1 << x)
        return S

    init_mask = greedy_init()
    init_size = init_mask.bit_count()
    best_size = init_size
    best_mask = init_mask
    
    # B&B Search
    # We search for sets larger than best_size.
    # Use recursion.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def search(selected_mask, available_mask, current_size, depth):
        nonlocal best_size, best_mask
        
        # Pruning: If we can't beat best_size, return
        # Upper bound: current_size + popcount(available_mask)
        if current_size + available_mask.bit_count() <= best_size:
            return
            
        # If no candidates, update best
        if available_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = selected_mask
            return
            
        # Choose a candidate
        # Heuristic: Pick a vertex with high degree in remaining constraints?
        # Or just pick the lowest/highest available?
        # Let's pick the lowest available to keep it simple, or use a heuristic.
        # A good heuristic is to pick a vertex that is involved in many "tight" constraints.
        # For now, let's pick the highest available number, as large numbers
        # constrain the space less (fewer sums <= N).
        # Actually, picking a number that is "hard" to place might prune faster.
        # Let's try picking the lowest available.
        
        # Get lowest set bit
        lsb = available_mask & -available_mask
        x = lsb.bit_length() - 1
        
        # Branch 1: Exclude x
        search(selected_mask, available_mask ^ lsb, current_size, depth + 1)
        
        # Branch 2: Include x
        # Check if x can be added
        can_add = True
        for rest in vertex_constraints[x]:
            if (selected_mask & rest) == rest:
                can_add = False
                break
                
        if can_add:
            new_selected = selected_mask | lsb
            # Update available_mask
            # When we add x, do we remove other candidates?
            # No, adding x doesn't immediately invalidate other candidates y,
            # unless y completes a forbidden set with x and others already in S.
            # But we check validity when we *try* to add y later.
            # So we just remove x from available.
            new_available = available_mask ^ lsb
            search(new_selected, new_available, current_size + 1, depth + 1)
            
    # Start search from empty set?
    # To speed up, we can start from the greedy solution and try to add more?
    # Or just run B&B from scratch with the lower bound.
    # B&B from scratch explores 2^N. We need strong pruning.
    # The lower bound from greedy helps.
    
    # To make it faster, we can order the vertices.
    # Let's run the search.
    # Time limit check: The task says 60s.
    # N=128 is large for pure Python B&B without C extensions.
    # However, the constraint structure is sparse.
    # Most numbers > 64 cannot be in any forbidden set as a "sum" component?
    # If x > 64, x cannot be a sum of two distinct positive integers <= 128?
    # Wait, 1+64=65. So 65 can be a sum.
    # Numbers > 64 can be sums.
    # But numbers > 64 cannot be part of a triple a,b,c where b+c <= 128?
    # If c > 64, then b+c > 64+b. If b>=1, b+c >= 65.
    # If c=65, b=1, sum=66.
    # So large numbers are involved in few constraints.
    
    # Let's try to optimize the search by fixing the order of vertices.
    # We can permute the vertices.
    # Let's just run the standard B&B. If it's too slow, we might need to stop early.
    # But we must return a valid result.
    
    # To ensure we don't time out, we can add a time check.
    start_time = time.time()
    
    # We'll do a limited depth or time-bound search if needed.
    # But let's hope the pruning is strong enough.
    
    # One optimization: If we have a large connected component of constraints,
    # we can solve it separately? No, the graph is likely connected.
    
    # Let's run the search.
    # To avoid stack overflow or excessive time, we can use an iterative approach or
    # limit the recursion depth if it's too deep, but 128 is fine for depth.
    
    # A better heuristic for vertex selection:
    # Pick the vertex with the most constraints among available?
    # This might prune faster.
    
    def get_vertex_with_most_constraints(available_mask):
        max_count = -1
        best_x = -1
        m = available_mask
        while m:
            lsb = m & -m
            x = lsb.bit_length() - 1
            # Count constraints that are "tight" (i.e., rest is subset of selected_mask? No, rest is fixed)
            # Actually, just count len(vertex_constraints[x])
            # This is static.
            # Better: Count how many of its constraints are "almost satisfied"?
            # Let's stick to static degree for now.
            deg = len(vertex_constraints[x])
            if deg > max_count:
                max_count = deg
                best_x = x
            m ^= lsb
        return best_x

    # Let's modify search to use this heuristic
    def search_heuristic(selected_mask, available_mask, current_size, depth):
        nonlocal best_size, best_mask
        
        if current_size + available_mask.bit_count() <= best_size:
            return
            
        if available_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = selected_mask
            return
            
        if time.time() - start_time > 55:
            return
            
        x = get_vertex_with_most_constraints(available_mask)
        if x == -1:
            return
            
        lsb = 1 << x
        
        # Exclude x
        search_heuristic(selected_mask, available_mask ^ lsb, current_size, depth + 1)
        
        # Include x
        can_add = True
        for rest in vertex_constraints[x]:
            if (selected_mask & rest) == rest:
                can_add = False
                break
                
        if can_add:
            new_selected = selected_mask | lsb
            new_available = available_mask ^ lsb
            search_heuristic(new_selected, new_available, current_size + 1, depth + 1)
            
    # Initialize with greedy
    # Also try a few random greedy inits to get a better lower bound
    import random
    random.seed(42)
    
    for _ in range(10):
        S = 0
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        for x in candidates:
            can_add = True
            for rest in vertex_constraints[x]:
                if (S & rest) == rest:
                    can_add = False
                    break
            if can_add:
                S |= (1 << x)
        sz = S.bit_count()
        if sz > best_size:
            best_size = sz
            best_mask = S
            
    search_heuristic(0, (1 << (N + 1)) - 2, 0, 0) # Bit 0 is unused, so mask is (1<<(N+1))-2
    
    # Convert best_mask to list of ints
    result = []
    for i in range(1, N + 1):
        if best_mask & (1 << i):
            result.append(i)
            
    return result

# EVOLVE-BLOCK-END
