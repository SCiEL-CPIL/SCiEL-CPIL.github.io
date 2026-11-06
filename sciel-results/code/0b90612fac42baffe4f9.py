"""Backtracking search for small inclusion-maximal Sidon sets in [1,256]."""
import time

N = 256
DEADLINE = time.time() + 55  # safety margin before 60s kill

# EVOLVE-BLOCK-START
def construct():
    # Precompute for each x in [1,N] the set of sums it would create
    # with itself and with each other element. We'll track forbidden sums
    # and forbidden elements via bitmasks.
    
    # A Sidon set S: all sums a+b (a<=b, a,b in S) are distinct.
    # An element x is "forbidden" (can't be added) if for some a in S,
    # x+a is already a sum in S, OR 2x is already a sum in S.
    
    # We build S incrementally. At each step, we pick an element to add.
    # To be inclusion-maximal, every element NOT in S must be forbidden.
    # We want to minimize |S|.
    
    # Strategy: backtracking with greedy heuristic.
    # - Maintain current set S, set of used sums, and set of forbidden elements.
    # - An element x is "available" if it's not in S and not yet forbidden.
    # - At each step, we can either:
    #   a) Add an available element to S (which may forbid more elements)
    #   b) Stop and check if S is maximal (all non-S elements are forbidden)
    # - To minimize |S|, we prefer to stop as soon as S becomes maximal.
    # - But we also need to ensure S is maximal, so we may need to add elements
    #   that forbid the remaining available elements.
    
    # Better strategy: 
    # - We want S to be Sidon AND maximal.
    # - Maximal means: for every x not in S, x is "blocked" by S.
    # - x is blocked by S if adding x to S would violate Sidon property.
    # - This happens if for some a in S, x+a is in sumset(S), or 2x is in sumset(S).
    
    # Let's use a different approach: 
    # - Pick elements greedily, each time choosing the element that blocks
    #   the most other elements (while maintaining Sidon property).
    # - Continue until no more elements can be added (all remaining are blocked).
    # - This gives a maximal Sidon set.
    # - To minimize size, we want each pick to block as many as possible.
    
    # Actually, let me think about this more carefully.
    # 
    # A Sidon set S in [1,N]. S is maximal if for every x in [1,N]\S,
    # S ∪ {x} is NOT a Sidon set.
    # S ∪ {x} fails to be Sidon iff there exist a,b,c,d in S∪{x} with
    # a+b = c+d, where {a,b} ≠ {c,d}.
    # Since S is Sidon, the only new sums involving x are x+x, x+s for s in S.
    # So S∪{x} fails Sidon iff:
    #   - 2x = a+b for some a,b in S (a<=b), OR
    #   - x+s = a+b for some s in S, a,b in S (a<=b), s≠a,b or appropriate
    # 
    # More precisely, x is "blocked" by S iff:
    #   - 2x is in the sumset of S (i.e., 2x = a+b for some a,b in S, a<=b), OR
    #   - For some s in S, x+s is in the sumset of S, i.e., x+s = a+b for some a,b in S.
    #     Note: if a=x or b=x, that's not possible since x not in S.
    #     So we need a,b in S with a+b = x+s.
    
    # Let's implement a greedy approach with backtracking for the first few picks.
    
    def is_blocked(x, S, sumset):
        """Check if x is blocked by Sidon set S with sumset."""
        if 2 * x in sumset:
            return True
        for s in S:
            if x + s in sumset:
                return True
        return False
    
    def compute_sumset(S):
        """Compute all sums a+b for a<=b in S."""
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        return sums
    
    # Greedy: at each step, among available (not blocked) elements,
    # pick the one that, when added, blocks the most other available elements.
    
    # But this might not minimize |S|. Let's try a smarter approach.
    
    # Alternative: we want to find a small set S such that every x not in S is blocked.
    # This is like a hitting set / domination problem.
    
    # Let me try: randomly initialize with a few elements, then greedily complete.
    
    import random
    random.seed(42)
    
    best_size = N + 1
    best_set = None
    start_time = time.time()
    
    # Try multiple random starting seeds
    for attempt in range(200):
        if time.time() > DEADLINE:
            break
        
        # Start with a random small seed
        seed_size = random.randint(1, 4)
        S = random.sample(range(1, N + 1), seed_size)
        S.sort()
        
        # Check if seed is Sidon
        sumset = compute_sumset(S)
        if len(sumset) != sum(1 for i in range(len(S)) for j in range(i, len(S))):
            continue
        
        # Greedily add elements to make it maximal
        while True:
            if time.time() > DEADLINE:
                break
            
            # Find all unblocked elements
            unblocked = []
            for x in range(1, N + 1):
                if x not in S:
                    if not is_blocked(x, S, sumset):
                        unblocked.append(x)
            
            if not unblocked:
                # S is maximal
                if len(S) < best_size:
                    best_size = len(S)
                    best_set = list(S)
                break
            
            # Pick the element that blocks the most other unblocked elements
            best_x = None
            best_block_count = -1
            for x in unblocked:
                # Simulate adding x
                new_sumset = sumset | {2 * x} | {x + s for s in S}
                # Count how many unblocked elements get blocked
                block_count = 0
                for y in unblocked:
                    if y != x:
                        if is_blocked(y, S + [x], new_sumset):
                            block_count += 1
                if block_count > best_block_count:
                    best_block_count = block_count
                    best_x = x
            
            if best_x is None:
                break
            
            S.append(best_x)
            S.sort()
            sumset = compute_sumset(S)
        
        if time.time() > DEADLINE:
            break
    
    if best_set is not None:
        return best_set
    
    # Fallback: greedy from scratch
    S = []
    sumset = set()
    for x in range(1, N + 1):
        new_sums = {2 * x} | {x + s for s in S}
        if len(new_sums) == len(new_sums | sumset) and not (new_sums & sumset):
            S.append(x)
            sumset |= new_sums
    
    return S
# EVOLVE-BLOCK-END
