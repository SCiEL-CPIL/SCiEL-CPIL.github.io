import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    N = 180
    
    # Precompute all forbidden triples: (a, b, c) with 1/a = 1/b + 1/c, all distinct
    # 1/a = 1/b + 1/c => c = ab/(b-a), requiring b > a, (b-a) | ab
    # Also need a < c (since 1/a > 1/c means a < c)
    # And b != c (distinct)
    
    triples = []
    # For each a, b with b > a:
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    # Sort the triple for consistent storage
                    t = tuple(sorted([a, b, c]))
                    triples.append(t)
    
    # Remove duplicates
    triples = list(set(triples))
    
    # Build optimized conflict structures
    # For each element x, store the set of pairs (u, v) such that {x, u, v} is a triple
    # This allows fast checking: can we add x to S? 
    # Yes iff no pair in triples_by_elem[x] is a subset of S.
    triples_by_elem = defaultdict(set)
    for (a, b, c) in triples:
        triples_by_elem[a].add(frozenset((b, c)))
        triples_by_elem[b].add(frozenset((a, c)))
        triples_by_elem[c].add(frozenset((a, b)))
    
    # Also keep the list of triples for counting or other uses if needed
    # But for validity check, the above is faster.
    
    def can_add(x, S):
        """Check if x can be added to S without forming a triple."""
        for pair in triples_by_elem[x]:
            if pair.issubset(S):
                return False
        return True

    def is_valid(S):
        """Check if set S has no forbidden triple."""
        # This is slower, only use for final verification or if structure is lost
        S_set = set(S)
        for (a, b, c) in triples:
            if a in S_set and b in S_set and c in S_set:
                return False
        return True

    # Strategy 1: Greedy removal (fast baseline)
    def greedy_remove():
        S = set(range(1, N + 1))
        # Count how many triples each element is in
        elem_triple_count = defaultdict(int)
        for (a, b, c) in triples:
            elem_triple_count[a] += 1
            elem_triple_count[b] += 1
            elem_triple_count[c] += 1
        
        while True:
            if time.time() > deadline:
                break
            # Find element with max triple count
            if not S:
                break
            max_count = -1
            max_elem = None
            for e in S:
                if elem_triple_count[e] > max_count:
                    max_count = elem_triple_count[e]
                    max_elem = e
            if max_count == 0:
                break
            # Remove max_elem
            S.remove(max_elem)
            # Update counts for all triples containing max_elem
            for pair in triples_by_elem[max_elem]:
                # pair is (u, v)
                u, v = tuple(pair)
                # The triple is (max_elem, u, v)
                # Decrease count for u and v if they are in S
                if u in S:
                    elem_triple_count[u] -= 1
                if v in S:
                    elem_triple_count[v] -= 1
            elem_triple_count[max_elem] = 0
        
        return S

    # Strategy 2: Optimized Local Search with Kicks
    def local_search(S, max_iters=20000):
        S = set(S)
        best = set(S)
        best_size = len(S)
        
        # Precompute list of elements for faster iteration
        all_elems = list(range(1, N + 1))
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # 1. Try to add elements
            # Randomize order to escape biases? Or just iterate.
            # Iterating all is O(N * deg). N=180, deg is small. Very fast.
            for x in all_elems:
                if x not in S:
                    if can_add(x, S):
                        S.add(x)
                        improved = True
                        if len(S) > best_size:
                            best = set(S)
                            best_size = len(S)
            
            # 2. If no improvement, try swaps
            if not improved:
                in_S = list(S)
                not_in_S = [x for x in all_elems if x not in S]
                
                if in_S and not_in_S:
                    # Try random swaps
                    # We want to remove an element that is "expensive" (in many triples with S)
                    # and add one that is "cheap".
                    # Approximation: remove random, add random if valid.
                    # To be smarter, pick x in S with highest degree in induced subgraph?
                    # Let's just do random swaps first, it's fast.
                    swapped = False
                    for _ in range(50): # Try up to 50 random swaps
                        if time.time() > deadline:
                            break
                        if not in_S or not not_in_S:
                            break
                        x = random.choice(in_S)
                        y = random.choice(not_in_S)
                        
                        # Check if S - {x} + {y} is valid
                        S_temp = S - {x}
                        if can_add(y, S_temp):
                            S = S_temp | {y}
                            in_S = list(S)
                            not_in_S = [z for z in all_elems if z not in S]
                            if len(S) > best_size:
                                best = set(S)
                                best_size = len(S)
                            improved = True
                            swapped = True
                            break # Restart loop with new state? Or continue?
                            # Usually better to restart the "add" phase after a successful swap
                            # But for speed, let's just break to outer loop
                            break
                    
                    if swapped:
                        continue # Go back to top of loop, try adding again
            
            # 3. If still no improvement, perform a "kick"
            # Remove a few random elements to break local optimum
            if not improved and len(S) > 10:
                kick_size = random.randint(2, 5)
                for _ in range(kick_size):
                    if in_S:
                        x = random.choice(in_S)
                        S.remove(x)
                        in_S = list(S)
                        not_in_S = [z for z in all_elems if z not in S]
                # Now try to refill
                for x in all_elems:
                    if x not in S:
                        if can_add(x, S):
                            S.add(x)
                            if len(S) > best_size:
                                best = set(S)
                                best_size = len(S)
        
        return best

    # Run greedy removal first
    result = greedy_remove()
    best_result = set(result)
    best_size = len(result)
    
    # Try local search from this result
    if time.time() < deadline - 10:
        ls_result = local_search(result, max_iters=50000)
        if len(ls_result) > best_size:
            best_result = ls_result
            best_size = len(ls_result)
    
    # Try multiple random restarts of greedy build + local search
    # Greedy build: add elements in random order if valid
    def greedy_build_random():
        S = set()
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S

    if time.time() < deadline - 5:
        for _ in range(10): # Try a few random restarts
            if time.time() > deadline - 5:
                break
            gb_result = greedy_build_random()
            # Polish with local search
            ls_res = local_search(gb_result, max_iters=10000)
            if len(ls_res) > best_size:
                best_result = ls_res
                best_size = len(ls_res)
                
    # Final local search on best result
    if time.time() < deadline - 2:
        final_result = local_search(best_result, max_iters=10000)
        if len(final_result) > best_size:
            best_result = final_result
            best_size = len(final_result)
    
    # Final verification (optional, but good for safety)
    # is_valid(best_result) is expensive, but N=180 is small.
    # The can_add logic is sound, so it should be valid.
    
    return list(best_result)
