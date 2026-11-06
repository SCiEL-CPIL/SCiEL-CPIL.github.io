"""Maximum Weight Independent Set on 3-AP hypergraph via specialized branch-and-bound."""

from fractions import Fraction
import time

N = 256

def construct():
    start_time = time.time()
    
    # 1. Precompute conflict partners
    # partners[v][w] is a list of u such that {u, v, w} is a 3-AP.
    # This helps in computing the "forbidden" set when v is included.
    partners = [[[] for _ in range(N + 1)] for _ in range(N + 1)]
    
    for v in range(1, N + 1):
        for w in range(1, N + 1):
            if v == w:
                continue
            # Case 1: v is smallest, w is middle -> u = 2w - v
            u = 2 * w - v
            if 1 <= u <= N and u != v and u != w:
                partners[v][w].append(u)
            # Case 2: v is smallest, w is largest -> u = (v+w)/2
            if (v + w) % 2 == 0:
                u = (v + w) // 2
                if 1 <= u <= N and u != v and u != w:
                    partners[v][w].append(u)
            # Case 3: v is middle, w is smallest -> u = 2v - w
            u = 2 * v - w
            if 1 <= u <= N and u != v and u != w:
                partners[v][w].append(u)
            # Case 4: v is middle, w is largest -> u = 2v - w (Same as 3? No, w is largest, so v is middle. u is smallest. u = 2v - w)
            # Wait, if v is middle and w is largest, then u = 2v - w.
            # If v is middle and w is smallest, then u = 2v - w.
            # So Case 3 and 4 are the same formula, but different interpretations of relative order.
            # The formula u = 2v - w generates the third point.
            # We just need to ensure u is valid and distinct.
            
            # Case 5: v is largest, w is middle -> u = 2w - v (Same as 1)
            # Case 6: v is largest, w is smallest -> u = (v+w)/2 (Same as 2)
            
            # Let's clean up duplicates in partners[v][w]
            if partners[v][w]:
                partners[v][w] = list(set(partners[v][w]))
                
    # 2. Order vertices by decreasing weight (1/x)
    order = list(range(1, N + 1))
    
    # 3. Precompute weights and suffix sums
    weights = [Fraction(0)] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = Fraction(1, i)
        
    suffix_sum = [Fraction(0)] * (N + 2)
    for i in range(N - 1, -1, -1):
        v = order[i]
        suffix_sum[i] = suffix_sum[i + 1] + weights[v]
    
    # 4. Greedy initialization
    in_set = [False] * (N + 1)
    
    def can_add(v):
        # Check if adding v creates a 3-AP with any two existing elements
        # We need to check if there exist a, b in S such that {a, b, v} is a 3-AP.
        # This is equivalent to checking if there exists w in S such that partners[v][w] contains another element in S?
        # No, partners[v][w] gives u such that {v, w, u} is AP.
        # If we add v, we need to ensure no {v, w, u} with w, u in S.
        # So for each w in S, check if any u in partners[v][w] is also in S.
        
        # Optimized check:
        # Iterate through all w in S? That's O(|S| * deg).
        # Alternatively, iterate through conflict pairs of v?
        # The original code used conflict_pairs[v] which are pairs (a,b) such that {a,b,v} is AP.
        # Let's stick to the original conflict_pairs structure for `can_add` as it's direct.
        pass

    # Let's re-create conflict_pairs for fast can_add
    conflict_pairs = [[] for _ in range(N + 1)]
    for v in range(1, N + 1):
        d = 1
        while v + 2 * d <= N:
            conflict_pairs[v].append((v + d, v + 2 * d))
            d += 1
        d = 1
        while v - d >= 1:
            conflict_pairs[v].append((v - d, v + d))
            d += 1
        d = 1
        while v - 2 * d >= 1:
            conflict_pairs[v].append((v - 2 * d, v - d))
            d += 1
            
    def can_add_fast(v):
        for a, b in conflict_pairs[v]:
            if in_set[a] and in_set[b]:
                return False
        return True

    greedy_set = []
    for v in order:
        if can_add_fast(v):
            greedy_set.append(v)
            in_set[v] = True
            
    best_set = list(greedy_set)
    best_weight = sum(weights[v] for v in greedy_set)
    
    import sys
    sys.setrecursionlimit(10000)
    
    # To speed up the "forbidden" calculation, we maintain the list of included vertices
    # and use the partners table.
    
    def bnb(i, included_list, current_weight):
        nonlocal best_set, best_weight
        
        if time.time() - start_time > 58:
            return
        
        if i == N:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = list(included_list)
            return
        
        v = order[i]
        
        # Upper Bound for Excluding v
        ub_exclude = current_weight + suffix_sum[i + 1]
        
        # Upper Bound for Including v
        ub_include = Fraction(0)
        if can_add_fast(v):
            # Calculate forbidden set
            forbidden = set()
            for w in included_list:
                # Add partners of v with w
                for u in partners[v][w]:
                    if u > v: # Only consider remaining vertices? 
                        # Actually, u must be in the remaining set to be relevant for the bound.
                        # The remaining set is order[i+1:].
                        # Since order is 1..N, remaining are > v (if order is sorted by index).
                        # Wait, order is 1, 2, ..., 256.
                        # So remaining indices are i+1 to N-1, corresponding to values v+1 to N.
                        # So we only care about u > v.
                        forbidden.add(u)
            
            # Sum weights of remaining vertices not in forbidden
            sum_rem = Fraction(0)
            for j in range(i + 1, N):
                u = order[j]
                if u not in forbidden:
                    sum_rem += weights[u]
            
            ub_include = current_weight + weights[v] + sum_rem
        
        if max(ub_exclude, ub_include) <= best_weight:
            return
            
        # Branch Include
        if can_add_fast(v):
            in_set[v] = True
            included_list.append(v)
            bnb(i + 1, included_list, current_weight + weights[v])
            included_list.pop()
            in_set[v] = False
            
        # Branch Exclude
        bnb(i + 1, included_list, current_weight)
        
    bnb(0, [], Fraction(0))
    
    return best_set
