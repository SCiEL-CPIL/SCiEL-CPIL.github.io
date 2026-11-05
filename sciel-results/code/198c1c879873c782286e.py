import time
import random
import sys
from collections import defaultdict

def construct():
    """
    Finds a maximum size Sidon set in [1, 284] containing {1, 2, 5, 11}.
    Uses Maximum Independent Set (MIS) via Bron-Kerbosch with pivot selection
    on a conflict graph where edges indicate Sidon property violations.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s limit
    
    required = {1, 2, 5, 11}
    max_val = 284
    target_size = 20
    
    # Build conflict graph: two numbers conflict if adding both to a set
    # with the required elements would create duplicate sums.
    # More precisely: for the full set S = required ∪ {a, b}, we need all
    # pairwise sums to be distinct.
    
    # Strategy: We must include required elements. The remaining elements
    # must form a Sidon set among themselves AND with the required elements.
    
    # Two candidate elements a, b (both not in required) conflict if:
    # 1. a + b creates a sum that equals some other pair's sum in the final set
    # This is complex to precompute statically, so we use a different approach.
    
    # Alternative: Build the conflict graph where vertices are all numbers in [1,284],
    # and an edge exists between x and y if they "conflict" in the sense that
    # they cannot both be in a Sidon set containing the required elements.
    
    # Two elements x, y conflict (cannot both be in the Sidon set with required)
    # if there exist elements p, q in (required ∪ {x, y}) such that:
    #   x + y = p + q, with {x,y} != {p,q}
    # OR more generally, the Sidon condition is violated.
    
    # Let's think more carefully. A Sidon set has all pairwise sums (with repetition) distinct.
    # For set S, the sums are {a+b : a,b in S, a<=b}. All must be distinct.
    
    # Two elements x and y (neither in required) are incompatible if there exist
    # a, b in required ∪ {x, y} with a <= b, c, d in required ∪ {x, y} with c <= d,
    # {a,b} != {c,d}, and a+b = c+d.
    
    # This is still complex. Let's use a simpler sufficient condition:
    # x and y conflict if x+y equals any sum from required∪{x} or required∪{y} pairs,
    # or if x+y creates a duplicate with another required pair, etc.
    
    # Actually, let's just build the graph properly:
    # For each pair (x, y) where x, y are in [1, 284], x != y,
    # check if {required} ∪ {x, y} is NOT a Sidon set. If not, add edge (x,y).
    # Also, x conflicts with required elements if {required} ∪ {x} is not Sidon.
    
    # But we must include ALL required elements, so we only need to choose from
    # the remaining 280 elements such that the full set is Sidon.
    
    # Let's precompute which elements are individually compatible with required:
    # An element x is compatible with required if required ∪ {x} is a Sidon set.
    
    def is_sidon(elements):
        """Check if a set of elements forms a Sidon set."""
        elems = sorted(elements)
        sums = set()
        for i in range(len(elems)):
            for j in range(i, len(elems)):
                s = elems[i] + elems[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    # Precompute compatibility with required
    required_list = sorted(required)
    required_sums = set()
    for i in range(len(required_list)):
        for j in range(i, len(required_list)):
            required_sums.add(required_list[i] + required_list[j])
    
    # An element x is compatible with required if:
    # - For all y in required: x + y not in required_sums (unless y == x, but x not in required)
    # - 2*x not in required_sums
    # - x + y != x + y' for y != y' in required (always true since y != y')
    # - x + y != 2*x for y in required, i.e., y != x (always true)
    
    compatible = set()
    for x in range(1, max_val + 1):
        if x in required:
            continue
        ok = True
        # Check x + y for y in required doesn't collide with required_sums
        for y in required_list:
            if x + y in required_sums:
                ok = False
                break
        if ok and 2 * x in required_sums:
            ok = False
        if ok:
            compatible.add(x)
    
    # Now build conflict graph among compatible elements
    # Two compatible elements x, y conflict if required ∪ {x, y} is not Sidon.
    # This means there exist a,b,c,d in required ∪ {x,y} with a<=b, c<=d, 
    # {a,b} != {c,d}, and a+b = c+d.
    
    # The new sums introduced by adding x,y to required are:
    # x+y, x+1, x+2, x+5, x+11, y+1, y+2, y+5, y+11, 2x, 2y
    # These must all be distinct from each other and from required_sums.
    
    # x and y conflict if:
    # 1. x + y is in required_sums, or x+y equals another new sum
    # 2. x + y' equals y + x' for x', y' in required (i.e., x - x' = y' - y)
    # 3. 2x equals some other new sum or required sum (already checked for single compat)
    # 4. x + y' = 2x (i.e., y' = x, impossible) or x + y' = 2y (i.e., x = 2y - y')
    # 5. x + y' = y + x'' where x', x'' in required, x' != x''
    
    # Let's just check directly:
    def conflict(x, y):
        """Check if x and y conflict (cannot both be in Sidon set with required)."""
        # All sums in required ∪ {x, y}
        elems = sorted(required_list + [x, y])
        sums = set()
        for i in range(len(elems)):
            for j in range(i, len(elems)):
                s = elems[i] + elems[j]
                if s in sums:
                    return True
                sums.add(s)
        return False
    
    # Build adjacency list using bitmasks for speed
    # Map compatible elements to indices
    comp_list = sorted(compatible)
    comp_index = {x: i for i, x in enumerate(comp_list)}
    n_comp = len(comp_list)
    
    # Build conflict bitmask for each element
    # adj[i] is a bitmask of which other compatible elements conflict with comp_list[i]
    adj = [0] * n_comp
    
    for i in range(n_comp):
        xi = comp_list[i]
        for j in range(i + 1, n_comp):
            xj = comp_list[j]
            if conflict(xi, xj):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Now solve MIS on this graph using Bron-Kerbosch with pivot
    best_mis = set()
    best_size = 0
    node_count = [0]
    
    def bron_kerbosch(P, X, R):
        """Bron-Kerbosch with pivot selection for maximum clique (MIS = max clique in complement).
        P: set of candidate vertices (bitmask)
        X: set of excluded vertices (bitmask) 
        R: set of vertices in current clique
        """
        nonlocal best_mis, best_size, node_count
        
        if time.time() > deadline:
            return True
        
        # Upper bound: |R| + |P|
        p_count = bin(P).count('1')
        if len(R) + p_count <= best_size:
            return False
        
        if P == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_mis = set(R)
            return False
        
        # Pivot selection: choose u from P ∪ X with maximum degree in P
        PX = P | X
        # Find vertex with max neighbors in P
        best_pivot = -1
        max_deg = -1
        temp = PX
        while temp:
            # Get lowest set bit
            lb = temp & (-temp)
            idx = lb.bit_length() - 1
            temp ^= lb
            deg = bin(adj[idx] & P).count('1')
            if deg > max_deg:
                max_deg = deg
                best_pivot = idx
        
        # Iterate over P \ N(pivot)
        candidates = P & ~(adj[best_pivot] if best_pivot >= 0 else 0)
        
        temp = candidates
        while temp:
            lb = temp & (-temp)
            idx = lb.bit_length() - 1
            temp ^= lb
            
            node_count[0] += 1
            if node_count[0] % 100000 == 0 and time.time() > deadline:
                return True
            
            # Recurse
            new_R = R | {idx}
            if bron_kerbosch(P & adj[idx], X, new_R):
                return True
            P ^= lb
            X |= lb
            
        return False
    
    # Run Bron-Kerbosch
    all_vertices_mask = (1 << n_comp) - 1
    timed_out = bron_kerbosch(all_vertices_mask, 0, set())
    
    # Convert best_mis (indices) back to values
    mis_values = set(comp_list[i] for i in best_mis)
    solution = sorted(mis_values | required)
    
    # Verify and return
    def verify_solution(sol):
        if not all(r in sol for r in required):
            return False
        if not all(1 <= x <= max_val for x in sol):
            return False
        if len(set(sol)) != len(sol):
            return False
        sums = set()
        for i in range(len(sol)):
            for j in range(i, len(sol)):
                s = sol[i] + sol[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    if verify_solution(solution):
        return solution
    
    # Fallback: try greedy + local search
    # Greedy: add elements one by one if compatible
    greedy_sol = set(required)
    used_sums = set()
    for i in range(len(required_list)):
        for j in range(i, len(required_list)):
            used_sums.add(required_list[i] + required_list[j])
    
    for x in comp_list:
        can_add = True
        for y in required_list:
            if x + y in used_sums:
                can_add = False
                break
        if can_add and 2 * x in used_sums:
            can_add = False
        if can_add:
            for y in greedy_sol:
                if y not in required and x + y in used_sums:
                    can_add = False
                    break
        if can_add:
            # Check against all elements in greedy_sol
            new_sums = set()
            for y in greedy_sol:
                new_sums.add(x + y)
            new_sums.add(2 * x)
            if new_sums & used_sums:
                can_add = False
        if can_add:
            greedy_sol.add(x)
            for y in list(greedy_sol):
                if y != x:
                    used_sums.add(x + y)
            used_sums.add(2 * x)
    
    solution2 = sorted(greedy_sol)
    if verify_solution(solution2):
        return solution2
    
    # Last resort: return just required
    return sorted(required)
