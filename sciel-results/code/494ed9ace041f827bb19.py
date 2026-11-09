import time
import sys
from math import isqrt

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin
    
    # Range of integers
    N = 256
    vertices = list(range(2, N + 1))
    
    # Helper to check if a is prime
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, isqrt(n) + 1):
            if n % i == 0:
                return False
        return True

    # Initial solution: All primes
    primes = [x for x in vertices if is_prime(x)]
    best_set = set(primes)
    best_size = len(best_set)
    
    # Precompute divisibility? 
    # For v in 2..256, and b in 2..256, c in 2..256
    # We need to check v | b*c
    
    def get_conflicts(v, S):
        """
        Returns the set of b in S such that there exists c in S \ {b, v} with v | b*c.
        These b's must be removed if v is added.
        """
        to_remove = set()
        S_list = list(S)
        # Optimization: if v is large, b*c must be multiple of v.
        # If v > max(S)^2, no conflicts. But v <= 256, max(S) <= 256.
        for b in S_list:
            if b == v:
                continue
            found = False
            for c in S_list:
                if c == b or c == v:
                    continue
                if (b * c) % v == 0:
                    found = True
                    break
            if found:
                to_remove.add(b)
        return to_remove

    # Branch and Bound
    # State: current_set (set), unvisited (list)
    # We start with S = primes, U = composites
    # But wait, if we start with S=primes, we assume primes are in the set.
    # We are looking for a set S' that might include composites and exclude some primes.
    # So we should start with S = empty, U = all vertices?
    # No, that's 2^255.
    # We must use the fact that the solution is likely close to primes.
    # Strategy:
    # 1. Start with S = primes.
    # 2. U = composites.
    # 3. Try to add composites to S, removing conflicting primes.
    # 4. If we get a larger set, update best.
    # 5. Also, we might need to remove primes without adding composites? No, that reduces size.
    # 6. What if we need to remove two primes to add one composite? Net loss.
    # 7. What if we remove one prime to add two composites? Net gain.
    #    Example: Remove p. Add c1, c2.
    #    This is handled by the search if we allow adding c1, then c2.
    
    # However, the search space "Start with primes, add composites" assumes primes are the base.
    # Is it possible the optimal set has NO primes? Unlikely.
    # Is it possible the optimal set has FEWER primes but MORE composites?
    # E.g., replace 5 primes with 6 composites?
    # The search should explore this.
    
    # To be safe, we can also try a greedy construction from scratch.
    # But primes are a very strong baseline.
    
    # Let's refine the search:
    # We maintain S (current set) and U (candidates).
    # Initially S = primes, U = composites.
    # We iterate through U.
    # For each v in U:
    #   Option 1: Add v.
    #     Remove conflicts from S.
    #     Recurse.
    #   Option 2: Skip v.
    #     Recurse.
    
    # This assumes we never remove a prime without adding a composite.
    # If we skip all composites, we end up with primes.
    # If we add composites, we remove primes.
    # This covers the space of sets that are "subset of primes union composites" where we start with primes.
    # It does NOT cover sets that exclude a prime but don't replace it immediately, or exclude a prime that doesn't conflict with any added composite (just to allow another composite later?).
    # Actually, if a prime p doesn't conflict with any added composite, there's no reason to remove it.
    # So any optimal set can be viewed as: Start with all primes. For each composite c, decide to add it. If added, remove conflicting primes.
    # Is this sufficient?
    # What if adding c1 requires removing p1. Adding c2 requires removing p2.
    # But c1 and c2 conflict with each other?
    # If c1 and c2 are in S, they must not conflict with each other or remaining S.
    # The `get_conflicts` function handles conflicts with the CURRENT S.
    # So if we add c1, remove p1. S = Primes \ {p1} + {c1}.
    # Then add c2. Check conflicts with new S.
    # This seems to cover the "swap" logic.
    
    # One caveat: What if we need to remove a prime p that does NOT conflict with any single composite, but conflicts with a PAIR of composites?
    # The condition is: a | bc.
    # If p is in S, and we add c1, c2.
    # If p | c1*c2, then p conflicts with the pair {c1, c2}.
    # My `get_conflicts(v, S)` checks if v | b*c for b,c in S.
    # So if we add c1, S has c1.
    # If we then add c2, `get_conflicts(c2, S)` will check if c2 | p*c1 (if p is still in S) AND if p | c2*c1.
    # Wait, `get_conflicts(v, S)` finds b in S such that v | b*c.
    # So if we add c2, it checks if c2 | b*c for b,c in S.
    # It does NOT check if some b in S divides c2*c.
    # The condition is symmetric in the sense that the set must be valid.
    # If S was valid before adding c2, the only new invalid triples involve c2.
    # So we only need to check triples {c2, b, c} where b, c are in S.
    # The condition is: c2 | b*c OR b | c2*c OR c | c2*b.
    # My `get_conflicts` only checks `v | b*c`.
    # It misses `b | v*c` and `c | v*b`.
    # 
    # Correction:
    # When adding v to S, we must ensure:
    # 1. v | b*c is false for all distinct b,c in S.
    # 2. b | v*c is false for all distinct c in S.
    # 3. c | v*b is false for all distinct b in S.
    # 
    # If condition 2 fails for some b, we must remove b.
    # If condition 3 fails for some c, we must remove c.
    # 
    # So `get_conflicts(v, S)` should return the set of b in S such that:
    # (Exists c in S \ {b, v} such that v | b*c) OR
    # (Exists c in S \ {b, v} such that b | v*c) OR
    # (Exists c in S \ {b, v} such that c | v*b) -- wait, this is covered by the second condition if we swap b and c?
    # No, b and c are specific elements.
    # If we remove b, the triple is gone.
    # So we need to find all b in S that are part of ANY invalid triple involving v.
    # Invalid triple {v, b, c} if v|bc or b|vc or c|vb.
    # So b is "conflicting" if:
    # - There exists c in S \ {b, v} such that v | b*c
    # - There exists c in S \ {b, v} such that b | v*c
    # - There exists c in S \ {b, v} such that c | v*b (this implies b is involved in a conflict where c is the "divisor". If we remove b, the triple is gone. So b must be removed if ANY c in S makes {v,b,c} invalid).
    
    # So for a fixed v and b in S:
    # b is conflicting if:
    # 1. Any c in S \ {b, v} satisfies v | b*c
    # 2. Any c in S \ {b, v} satisfies b | v*c
    # 3. Any c in S \ {b, v} satisfies c | v*b
    
    # Let's implement this correctly.

    def get_conflicts_full(v, S):
        to_remove = set()
        S_list = list(S)
        for b in S_list:
            if b == v:
                continue
            conflict = False
            for c in S_list:
                if c == b or c == v:
                    continue
                if (b * c) % v == 0: # v | bc
                    conflict = True
                    break
                if (v * c) % b == 0: # b | vc
                    conflict = True
                    break
                if (v * b) % c == 0: # c | vb
                    conflict = True
                    break
            if conflict:
                to_remove.add(b)
        return to_remove

    # Branch and Bound DFS
    # To avoid recursion depth issues and allow time checking, use iterative or careful recursion.
    # 255 depth is fine for Python if we don't have huge branching.
    # But branching factor 2^255 is impossible. Pruning is key.
    
    # Heuristic: Order U by "cost" (number of conflicts).
    # High cost vertices are harder to add.
    # Maybe try adding low cost vertices first?
    # Or high cost to prune?
    # Let's sort U dynamically? That's expensive.
    # Static sort: Composites sorted by value?
    # Let's try sorting composites by "divisibility degree" (how many multiples they have in range).
    
    composites = [x for x in vertices if not is_prime(x)]
    
    # Precompute a simple score for composites to order them
    # Score: number of x in 2..256 such that x % c == 0 or c % x == 0?
    # Let's just sort by value for now.
    
    best_set_global = set(primes)
    
    def dfs(S, U):
        nonlocal best_set_global
        if time.time() > deadline:
            return
        
        current_size = len(S)
        if current_size + len(U) <= len(best_set_global):
            return
            
        if not U:
            if current_size > len(best_set_global):
                best_set_global = set(S)
            return
            
        # Pick a vertex
        # Heuristic: Pick the one with highest conflict count?
        # Let's pick the first one for simplicity, but maybe sort U initially.
        # We can't sort U dynamically in every node (expensive).
        # Let's assume the order in U is fixed.
        
        v = U[0]
        U_rest = U[1:]
        
        # Option 1: Include v
        conflicts = get_conflicts_full(v, S)
        S_new = (S - conflicts) | {v}
        # Note: conflicts are removed from S. They are not in U (they were in S).
        # They are effectively excluded.
        dfs(S_new, U_rest)
        
        # Option 2: Exclude v
        dfs(S, U_rest)
        
    # Initialize
    S_init = set(primes)
    U_init = composites
    
    # To improve search, we can try multiple orderings of composites.
    # But let's start with one.
    
    # Also, we should consider that the optimal set might not contain ALL primes.
    # Our search starts with all primes and only removes them if a composite is added.
    # It does not explore removing a prime without adding a composite.
    # This is safe because removing a prime without adding anything reduces size.
    # It does not explore removing a prime p1 to add composite c1, and removing p2 to add c2, where p1 and p2 are distinct.
    # This IS explored: Add c1 (removes p1), then add c2 (removes p2 from the new S).
    
    # One risk: What if adding c1 removes p1, but p1 was needed to "block" c2?
    # No, p1 being in S doesn't block c2 unless they conflict.
    # If they conflict, p1 would be removed when c2 is added?
    # No, if c2 is added, we check conflicts with current S.
    # If p1 is in S, and c2 conflicts with p1, p1 is removed.
    # So the logic holds.
    
    dfs(S_init, U_init)
    
    # Return as list of ints
    return list(best_set_global)

# EVOLVE-BLOCK-END
