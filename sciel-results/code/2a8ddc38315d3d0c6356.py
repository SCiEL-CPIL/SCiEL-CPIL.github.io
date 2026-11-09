"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Return a nonempty subset of [1,255] containing 255, with gcd(a,b)>1 for all distinct pairs.
    Maximize cardinality.
    """
    start_time = time.time()
    deadline = 85

    # Build candidate set: numbers sharing a factor with 255 = 3*5*17
    candidates = [i for i in range(1, 256) if i % 3 == 0 or i % 5 == 0 or i % 17 == 0]
    n = len(candidates)
    
    # Precompute adjacency as sets for fast membership
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    idx_255 = candidates.index(255)
    
    def is_clique(S):
        """Check if S is a clique."""
        for a in S:
            for b in S:
                if a != b and b not in adj[a]:
                    return False
        return True
    
    def can_add(S, x):
        """Check if x can be added to S to maintain clique property."""
        if x in S:
            return False
        for y in S:
            if x not in adj[y] and x != y:
                return False
        return True
    
    def greedy_expand(S):
        """Greedily add compatible elements to S."""
        S = set(S)
        changed = True
        while changed and (time.time() - start_time < deadline - 3):
            changed = False
            for add in range(n):
                if add in S:
                    continue
                if can_add(S, add):
                    S.add(add)
                    changed = True
                    break
        return S
    
    def local_search(S, max_iters=50):
        """Local search with remove-one-add-many and 2-swap moves."""
        S = set(S)
        best = S
        best_size = len(S)
        
        for _ in range(max_iters):
            if time.time() - start_time > deadline - 3:
                break
            improved = False
            
            # Try remove-one-add-many
            for rem in list(S):
                if rem == idx_255:
                    continue
                S_t = S - {rem}
                # Add all compatible elements
                for add in range(n):
                    if add in S_t:
                        continue
                    if can_add(S_t, add):
                        S_t.add(add)
                if len(S_t) > best_size:
                    best = S_t
                    best_size = len(S_t)
                    S = S_t
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 2-swap: remove two, add two
            for rem1, rem2 in combinations(list(S), 2):
                if rem1 == idx_255 or rem2 == idx_255:
                    continue
                S_t = S - {rem1, rem2}
                # Find best additions
                additions = []
                for add in range(n):
                    if add in S_t:
                        continue
                    if can_add(S_t, add):
                        additions.append(add)
                # Try adding up to 2 new elements
                if len(additions) >= 2:
                    # Try all pairs of additions
                    for a1, a2 in combinations(additions[:10], 2):
                        S_try = S_t | {a1, a2}
                        if len(S_try) > best_size and is_clique(S_try):
                            best = S_try
                            best_size = len(S_try)
                            S = S_try
                            improved = True
                            break
                elif len(additions) == 1:
                    S_try = S_t | {additions[0]}
                    if len(S_try) > best_size and is_clique(S_try):
                        best = S_try
                        best_size = len(S_try)
                        S = S_try
                        improved = True
                
                if improved:
                    break
            
            if not improved:
                break
        
        return best
    
    # Start with the known large clique: all multiples of 3 (size 85)
    best_set = set(i for i in range(n) if candidates[i] % 3 == 0)
    best_size = len(best_set)
    
    # Run local search from initial clique
    if time.time() - start_time < deadline - 10:
        result = local_search(best_set, max_iters=100)
        if len(result) > best_size:
            best_set = result
            best_size = len(result)
    
    # Try starting from other large cliques
    if time.time() - start_time < deadline - 15:
        for prime in [5, 17]:
            if time.time() - start_time > deadline - 5:
                break
            S = set(i for i in range(n) if candidates[i] % prime == 0)
            if idx_255 not in S:
                continue
            # Expand
            S = greedy_expand(S)
            # Local search
            result = local_search(S, max_iters=50)
            if len(result) > best_size:
                best_set = result
                best_size = len(result)
    
    # Try mixed initializations: multiples of 3 plus some multiples of 5 or 17
    if time.time() - start_time < deadline - 20:
        # Try: all multiples of 3, plus multiples of 15 (already included), plus multiples of 51 (already included)
        # Try: numbers divisible by at least two of {3, 5, 17}
        S2 = set(i for i in range(n) if (candidates[i] % 3 == 0 and candidates[i] % 5 == 0) or 
                 (candidates[i] % 3 == 0 and candidates[i] % 17 == 0) or
                 (candidates[i] % 5 == 0 and candidates[i] % 17 == 0))
        if idx_255 in S2:
            S2 = greedy_expand(S2)
            result = local_search(S2, max_iters=50)
            if len(result) > best_size:
                best_set = result
                best_size = len(result)
    
    # Final polish: try to add any remaining compatible elements
    if time.time() - start_time < deadline - 3:
        changed = True
        while changed and (time.time() - start_time < deadline - 3):
            changed = False
            for add in range(n):
                if add in best_set:
                    continue
                if can_add(best_set, add):
                    best_set.add(add)
                    best_size = len(best_set)
                    changed = True
                    break
    
    return sorted(candidates[i] for i in best_set)

# EVOLVE-BLOCK-END
