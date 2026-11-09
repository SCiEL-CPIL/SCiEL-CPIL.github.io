import random
from math import gcd
import time

N = 128

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # Precompute all bad triples
    # A triple (a, b, c) with a < b < c is bad if gcd(a,b) == gcd(b,c) == gcd(c,a)
    bad_triples = []
    # Map each vertex to a list of indices of bad triples it participates in
    # Or better, map each vertex to a set of "partner pairs" that form a bad triple with it.
    # For fast checking: can_add(x, S) is False if there exists u, v in S such that (x, u, v) is a bad triple.
    
    # Let's precompute for each x, the set of pairs (u, v) such that {x, u, v} is a bad triple.
    # We can store this as a dictionary: conflict_pairs[x] = set of frozenset({u, v})
    # But checking "is there u, v in S" is O(|S|^2) or O(|conflict_pairs[x]| * |S|).
    
    # Alternative: For each x, store the list of bad triples involving x.
    # triples_with[x] = list of (u, v) such that {x, u, v} is bad.
    triples_with = {i: [] for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd(a, b)
            for c in range(b + 1, N + 1):
                if gcd(a, c) == g_ab and gcd(b, c) == g_ab:
                    # Triple (a, b, c) is bad
                    triples_with[a].append((b, c))
                    triples_with[b].append((a, c))
                    triples_with[c].append((a, b))
    
    # Function to check if x can be added to set S
    # S is a set of integers
    def can_add(x, S):
        # Check if any pair in triples_with[x] is fully contained in S
        # triples_with[x] contains tuples (u, v)
        for u, v in triples_with[x]:
            if u in S and v in S:
                return False
        return True

    def greedy_build(order):
        """
        Build a set by adding elements in the given order if possible.
        """
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S

    def local_search(S):
        """
        Improve set S by adding elements or swapping.
        Returns the improved set.
        """
        improved = True
        # Limit iterations to prevent infinite loops in case of oscillation
        max_iters = 100
        iter_count = 0
        
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # 1. Try to add any element
            candidates = [x for x in range(1, N + 1) if x not in S]
            random.shuffle(candidates)
            for x in candidates:
                if can_add(x, S):
                    S.add(x)
                    improved = True
                    break
            
            if improved:
                continue

            # 2. If no single addition possible, try 1-remove, 1-add (swap) or 1-remove, 2-add
            # We want to increase size. So 1-remove, 1-add keeps size same.
            # 1-remove, 2-add increases size by 1.
            # Let's try to remove one element u, and then greedily add as many as possible from complement.
            
            list_S = list(S)
            random.shuffle(list_S)
            
            best_S = S
            best_size = len(S)
            
            for u in list_S:
                S_temp = S - {u}
                # Try to add elements to S_temp
                # We can just run a mini greedy add loop
                added = 0
                # To be efficient, just try to add any available element repeatedly?
                # Or just one pass?
                # Let's do a pass of adding any possible element.
                # This is O(N * |triples_with[x]|)
                
                # We'll try to add elements one by one.
                # Since we removed u, some elements might become addable.
                
                # Optimization: Only check elements that were blocked by u?
                # An element x is blocked by S if there exists (u, v) in triples_with[x] with u, v in S.
                # If we remove u, x might be unblocked if all its blocking pairs involved u.
                
                # Simpler: Just try to add any element from complement to S_temp.
                # We can reuse the 'can_add' logic.
                
                # To avoid O(N^2) overhead in local search, we'll just try a few random additions.
                # Actually, let's just try to add ALL possible elements to S_temp greedily.
                # This is effectively re-running the greedy fill on S_temp.
                
                # Let's just try to add one element x. If we can add x, size is same.
                # If we can add x, and then y, size is +1.
                
                # Let's try to add as many as possible in one shot.
                # We'll iterate through complement and add if possible.
                # This is O(N * cost(can_add)).
                
                new_S = set(S_temp)
                # We need to check complement of new_S
                # To speed up, we can just iterate 1..N
                for x in range(1, N + 1):
                    if x in new_S:
                        continue
                    if can_add(x, new_S):
                        new_S.add(x)
                
                if len(new_S) > best_size:
                    best_size = len(new_S)
                    best_S = new_S
                    improved = True
                    # Don't break, keep looking for even better? 
                    # Or break to apply this improvement and restart loop?
                    # Breaking and restarting the main while loop is safer.
                    break
            
            if improved:
                S = best_S
            else:
                # No improvement found with 1-remove
                # Try 2-remove? Too expensive.
                # Accept local minimum.
                break
                
        return S

    best_set = set()
    
    # Run multiple random greedy builds
    # Time budget allocation:
    # 1. Greedy builds
    # 2. Local search on the best few
    
    while time.time() < deadline:
        # Random order
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S = greedy_build(order)
        
        # Improve with local search
        S = local_search(S)
        
        if len(S) > len(best_set):
            best_set = S
            # If we find a very good set, we might want to spend more time polishing it?
            # For now, just keep track.
        
        # If we found a set of size 100+, we might be close to optimal.
        # The max possible is likely around 100-110.
        # Let's check current best.
        
        # To ensure we use time effectively, if we have a good set, 
        # we can try to perturb it.
        # But for now, random restarts are a good structural change from pure local search.
        
        # Add a small sleep or check time to avoid tight loop if greedy is too fast
        if time.time() > deadline - 5.0:
            break

    return list(best_set)

# Test locally
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    from itertools import combinations
    def is_valid(S):
        for a, b, c in combinations(S, 3):
            if gcd(a, b) == gcd(b, c) == gcd(c, a):
                return False
        return True
    print(f"Valid: {is_valid(res)}")
