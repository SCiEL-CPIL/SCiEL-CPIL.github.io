import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 85.0
    
    N = 180
    
    # Precompute all conflict triples: (a, b, c) where a < b < c and 1/a = 1/b + 1/c
    # Equivalent: bc = a(b+c)
    # For given a, b: c = a*b / (b - a), must be integer, c > b, c <= N, c != a, c != b
    conflict_pairs = {}  # x -> set of y such that x,y are in a conflicting triple together
    triples = []
    # For each a, store list of (b, c) pairs
    triples_by_a = {}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            den = b - a
            num = a * b
            if num % den != 0:
                continue
            c = num // den
            if c > b and c <= N and c != a and c != b:
                triples.append((a, b, c))
                triples_by_a.setdefault(a, []).append((b, c))
                # a conflicts with b, a conflicts with c, b conflicts with c
                conflict_pairs.setdefault(a, set()).add(b)
                conflict_pairs.setdefault(a, set()).add(c)
                conflict_pairs.setdefault(b, set()).add(a)
                conflict_pairs.setdefault(b, set()).add(c)
                conflict_pairs.setdefault(c, set()).add(a)
                conflict_pairs.setdefault(c, set()).add(b)
    
    # Also need to check: what if a is not the smallest? 
    # The equation 1/a = 1/b + 1/c means a is the harmonic mean related.
    # Since 1/a = 1/b + 1/c > 1/c and > 1/b, we have a < b and a < c.
    # So a is always the smallest. Good.
    
    # But wait - we also need to consider that the three elements could be in any order
    # in the equation. Let me re-examine: 1/a = 1/b + 1/c
    # This requires a < b and a < c (since 1/a > 1/b and 1/a > 1/c)
    # So a is the smallest. The equation is symmetric in b and c.
    
    # For checking if a set S is valid: for all triples (a,b,c) in our precomputed list,
    # not all three are in S.
    
    def is_valid(S):
        """Check if set S has no conflicting triple."""
        for (a, b, c) in triples:
            if a in S and b in S and c in S:
                return False
        return True
    
    def count_conflicts(S):
        """Count number of conflicting triples in S."""
        count = 0
        for (a, b, c) in triples:
            if a in S and b in S and c in S:
                count += 1
        return count
    
    def can_add_to_S(S, x):
        """Check if x can be added to S without creating a conflict."""
        # For each a where x is b or c in a triple (a, x, y) or (a, y, x),
        # check if a and the other element are in S
        for (a, b, c) in triples:
            if x == a:
                if b in S and c in S:
                    return False
            elif x == b:
                if a in S and c in S:
                    return False
            elif x == c:
                if a in S and b in S:
                    return False
        return True
    
    def count_conflicts_with_x(S, x):
        """Count how many triples would be violated if x were added to S."""
        count = 0
        for (a, b, c) in triples:
            if x == a:
                if b in S and c in S:
                    count += 1
            elif x == b:
                if a in S and c in S:
                    count += 1
            elif x == c:
                if a in S and b in S:
                    count += 1
        return count
    
    def greedy_remove():
        """Start with all elements, greedily remove elements with most conflicts."""
        S = set(range(1, N + 1))
        while True:
            if time.time() - start_time > deadline - 5:
                break
            # Find element with highest number of conflict partners in S
            best_elem = None
            best_count = 0
            for x in S:
                partners_in_S = len(conflict_pairs.get(x, set()) & S)
                if partners_in_S > best_count:
                    best_count = partners_in_S
                    best_elem = x
            if best_elem is None or best_count == 0:
                break
            S.remove(best_elem)
        return S
    
    def structured_greedy_remove():
        """Greedy removal that prioritizes by a-structure: for each a, if both b and c are present, remove one."""
        S = set(range(1, N + 1))
        # Process a's in order of increasing conflict density
        a_order = sorted(range(1, N + 1), key=lambda a: -len(triples_by_a.get(a, [])))
        
        for a in a_order:
            if time.time() - start_time > deadline - 5:
                break
            if a not in S:
                continue
            # For this a, we need to ensure not both b and c are in S for any (b,c) in triples_by_a[a]
            # This is like a bipartite graph: left = b's, right = c's, edge if (b,c) in triples_by_a[a]
            # We want a maximum independent set in this bipartite graph, but constrained to S
            # For simplicity, greedily remove: for each (b,c), if both in S, remove the one with more conflicts
            for (b, c) in triples_by_a.get(a, []):
                if b in S and c in S:
                    # Remove the one with higher degree in overall conflict graph
                    deg_b = len(conflict_pairs.get(b, set()) & S)
                    deg_c = len(conflict_pairs.get(c, set()) & S)
                    if deg_b >= deg_c:
                        S.remove(b)
                    else:
                        S.remove(c)
        return S
    
    def greedy_add_order(order):
        """Greedy add in given order."""
        S = set()
        for x in order:
            # Check if adding x creates any conflict
            can_add = True
            for (a, b, c) in triples:
                # Check if x is one of a,b,c and the other two are in S
                if x == a and b in S and c in S:
                    can_add = False
                    break
                if x == b and a in S and c in S:
                    can_add = False
                    break
                if x == c and a in S and b in S:
                    can_add = False
                    break
            if can_add:
                S.add(x)
        return S
    
    def local_search(S, max_iter=5000):
        """Local search: try add, remove, and swap moves."""
        S = set(S)
        best = set(S)
        best_size = len(S)
        
        for iteration in range(max_iter):
            if time.time() - start_time > deadline - 3:
                break
            
            improved = False
            
            # Try adding elements
            for x in range(1, N + 1):
                if x in S:
                    continue
                if can_add_to_S(S, x):
                    S.add(x)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best = set(S)
            
            if improved:
                continue
            
            # Try removing one element to enable additions
            for x in list(S):
                S2 = S - {x}
                # Try to add back elements
                for y in range(1, N + 1):
                    if y in S2 or y == x:
                        continue
                    if can_add_to_S(S2, y):
                        S2.add(y)
                if len(S2) > len(S):
                    S = S2
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best = set(S)
                    break
            
            if not improved:
                # Try 2-swap: remove x, add y
                found_swap = False
                for x in list(S):
                    if found_swap:
                        break
                    S2 = S - {x}
                    for y in range(1, N + 1):
                        if y in S2 or y == x:
                            continue
                        if can_add_to_S(S2, y) and len(S2) + 1 > len(S):
                            S2.add(y)
                            S = S2
                            found_swap = True
                            improved = True
                            if len(S) > best_size:
                                best_size = len(S)
                                best = set(S)
                            break
            
            if not improved:
                break
        
        return best
    
    def local_search_fast(S, max_iter=3000):
        """Faster local search using precomputed conflict info."""
        S = set(S)
        best = set(S)
        best_size = len(S)
        
        for iteration in range(max_iter):
            if time.time() - start_time > deadline - 3:
                break
            
            improved = False
            
            # Try adding elements - check conflicts more efficiently
            for x in range(1, N + 1):
                if x in S:
                    continue
                if can_add_to_S(S, x):
                    S.add(x)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best = set(S)
            
            if not improved:
                # Try remove + add
                for x in list(S):
                    S2 = S - {x}
                    for y in range(1, N + 1):
                        if y in S2 or y == x:
                            continue
                        if can_add_to_S(S2, y):
                            S2.add(y)
                    if len(S2) > len(S):
                        S = S2
                        improved = True
                        if len(S) > best_size:
                            best_size = len(S)
                            best = set(S)
                        break
            
            if not improved:
                break
        
        return best
    
    # Strategy 1: Greedy removal
    best = greedy_remove()
    best_size = len(best)
    
    # Strategy 1b: Structured greedy removal using a-decomposition
    S_struct = structured_greedy_remove()
    if len(S_struct) > best_size:
        best = S_struct
        best_size = len(S_struct)
    
    # Strategy 2: Greedy add in different orders
    for order_func in [lambda: list(range(1, N+1)), 
                       lambda: list(range(N, 0, -1)),
                       lambda: sorted(range(1, N+1), key=lambda x: -len(conflict_pairs.get(x, set())))]:
        S2 = greedy_add_order(order_func())
        if len(S2) > best_size:
            best = S2
            best_size = len(S2)
    
    # Strategy 3: Random restarts with local search
    random.seed(42)
    count = 0
    while time.time() - start_time < deadline - 10:
        # Random greedy
        order = list(range(1, N+1))
        random.shuffle(order)
        S_r = greedy_add_order(order)
        S_improved = local_search_fast(S_r, max_iter=2000)
        
        if len(S_improved) > best_size:
            best = S_improved
            best_size = len(S_improved)
        
        count += 1
        if count > 300:
            break
    
    # Final polish
    if time.time() - start_time < deadline - 5:
        best_final = local_search_fast(best, max_iter=5000)
        if len(best_final) > best_size:
            best = best_final
            best_size = len(best_final)
    
    # Verify
    if not is_valid(best):
        # Fallback: greedy removal
        best = greedy_remove()
    
    return list(best)
