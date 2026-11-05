import time
import math
import random
import sys
from itertools import combinations

N = 128
TIME_LIMIT = 85.0

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Precompute all forbidden triples
    # A triple (a, b, c) with a < b < c is forbidden if lcm(a,b) == lcm(a,c) == lcm(b,c)
    # We can optimize: let L = lcm(a,b). Then c must divide L, and lcm(a,c)=L and lcm(b,c)=L.
    # Actually, if L = lcm(a,b), then lcm(a,c) = L implies c divides L.
    # And lcm(b,c) = L implies c divides L.
    # So c must be a divisor of L.
    # Also, we need distinct a, b, c.
    
    # Let's just brute force all triples since N=128 is small.
    # C(128, 3) is about 340,000. This is fast.
    forbidden_pairs = [[False] * (N + 1) for _ in range(N + 1)]
    # Actually, we want to check if adding x to a set S creates a forbidden triple.
    # This happens if there exist u, v in S such that {x, u, v} is a forbidden triple.
    # So for each x, we can precompute a list of pairs (u, v) such that {x, u, v} is forbidden.
    # Let conflicts[x] be a list of (u, v).
    
    conflicts = [[] for _ in range(N + 1)]
    
    # To speed up, we can iterate a < b, compute L = lcm(a,b), then iterate c such that c < a (or c > b? no, just distinct)
    # But we need to be careful with ordering.
    # Let's just do the triple loop.
    
    # Optimization: 
    # lcm(a,b) = L. 
    # lcm(a,c) = L => c | L.
    # lcm(b,c) = L => c | L.
    # So c is a divisor of L.
    # We can precompute divisors for all numbers up to L_max.
    # L_max can be as large as lcm(127, 128) which is 127*128 = 16256.
    # Divisors of 16256 are not too many.
    
    # Let's stick to the simple loop first, it's only 340k iterations.
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            L_ab = math.lcm(a, b)
            # c must be distinct from a and b.
            # Also lcm(a,c) == L_ab and lcm(b,c) == L_ab.
            # This implies c divides L_ab.
            # So we only need to check divisors of L_ab.
            # But we also need c in [1, N].
            
            # Let's iterate c from 1 to N.
            # If L_ab > N, c must still be a divisor of L_ab and <= N.
            # Actually, if L_ab > N, c can still be <= N.
            # Example: a=3, b=5, L=15. c=15 (if N>=15).
            # If N=128, L can be large.
            
            # Let's just iterate c in 1..N.
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if math.lcm(a, c) == L_ab and math.lcm(b, c) == L_ab:
                    # Found forbidden triple {a, b, c}
                    # Add to conflicts for each element
                    # We store pairs.
                    conflicts[a].append((b, c))
                    conflicts[b].append((a, c))
                    conflicts[c].append((a, b))
    
    # Now we have conflicts.
    # We want to find a large subset S of [1, N] such that no {x, u, v} in S is a forbidden triple.
    
    # Strategy:
    # 1. Greedy construction with multiple orderings.
    # 2. Local search (swap/add/remove) to improve.
    
    def can_add(x, current_set):
        """Check if x can be added to current_set."""
        # current_set is a set of ints.
        # For each (u, v) in conflicts[x], if u in current_set and v in current_set, return False.
        for u, v in conflicts[x]:
            if u in current_set and v in current_set:
                return False
        return True
    
    def greedy(order):
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S
    
    def local_search(S, time_limit):
        """Improve S by swapping elements."""
        S = set(S)
        improved = True
        while improved and (time.time() < time_limit):
            improved = False
            # Try to add an element not in S
            # Or swap an element in S with one not in S
            
            # To make this efficient, we can try to find an element x not in S
            # such that removing some y in S allows adding x.
            # Or just try to add x directly if possible (though greedy should have done that).
            # Actually, greedy might have missed opportunities due to ordering.
            
            # Let's try: for each x not in S, see if we can add it.
            # If yes, add it.
            # If no, see if removing some y in S allows adding x.
            
            # This is O(|S| * |NotS| * cost(can_add)).
            # |S| ~ 50, |NotS| ~ 80. 4000 checks.
            # cost(can_add) is O(len(conflicts[x])).
            
            # Let's try a simpler local search:
            # 1. Try to add any x not in S.
            for x in range(1, N + 1):
                if x not in S and can_add(x, S):
                    S.add(x)
                    improved = True
                    break
            
            if improved:
                continue
                
            # 2. Try to swap y in S with x not in S.
            # For each y in S, for each x not in S:
            #   If can_add(x, S - {y}), then swap.
            
            S_list = list(S)
            NotS_list = [x for x in range(1, N + 1) if x not in S]
            
            for y in S_list:
                S_minus_y = S - {y}
                for x in NotS_list:
                    if can_add(x, S_minus_y):
                        # Check if this swap actually improves things?
                        # Since we maintain size, it's neutral in size.
                        # But it might allow further additions.
                        # Let's accept it if it doesn't decrease size (it doesn't).
                        # To avoid cycling, we can use a tabu list or just randomize.
                        # For now, let's just do it.
                        S.remove(y)
                        S.add(x)
                        improved = True
                        break
                if improved:
                    break
            
            # 3. Try to remove y from S if it allows adding two elements?
            # This is more complex.
            
        return S
    
    # Initial solutions
    best_S = set()
    best_size = 0
    
    # 1. Greedy with different orderings
    orders = []
    
    # Natural order
    orders.append(list(range(1, N + 1)))
    # Reverse order
    orders.append(list(range(N, 0, -1)))
    
    # Random orders
    num_random = 50
    for _ in range(num_random):
        o = list(range(1, N + 1))
        random.shuffle(o)
        orders.append(o)
    
    # Degree-based orders
    # Degree of x is number of pairs in conflicts[x]
    degrees = [len(conflicts[x]) for x in range(N + 1)]
    orders.append(sorted(range(1, N + 1), key=lambda x: degrees[x]))
    orders.append(sorted(range(1, N + 1), key=lambda x: -degrees[x]))
    
    for o in orders:
        if time.time() > deadline:
            break
        S = greedy(o)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
    
    # 2. Local search on the best solution
    if time.time() < deadline:
        best_S = local_search(best_S, deadline)
        best_size = len(best_S)
        
    # 3. Try local search on other good solutions?
    # If we have time, run local search on the top few greedy solutions.
    
    # Let's collect top 5 greedy solutions
    greedy_solutions = []
    for o in orders:
        if time.time() > deadline:
            break
        S = greedy(o)
        greedy_solutions.append((len(S), S))
        
    greedy_solutions.sort(key=lambda x: -x[0])
    
    # Run local search on the top ones if they are distinct
    seen = set()
    for size, S in greedy_solutions:
        if time.time() > deadline:
            break
        # Check if this S is already processed
        key = frozenset(S)
        if key in seen:
            continue
        seen.add(key)
        
        S_opt = local_search(S, deadline)
        if len(S_opt) > best_size:
            best_size = len(S_opt)
            best_S = S_opt.copy()
            
    return list(best_S)

if __name__ == "__main__":
    res = construct()
    print(len(res))
