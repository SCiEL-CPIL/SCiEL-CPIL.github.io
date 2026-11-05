import time
import random
from math import lcm

def construct():
    deadline = time.time() + 85
    
    # LCM of 1..22
    D = 232792560
    vals = {x: D // x for x in range(1, 23)}
    all_elements = list(range(1, 23))
    
    def get_sums(S):
        """Return a list of all subset sums of S."""
        sums = [0]
        for x in S:
            v = vals[x]
            new_sums = []
            for s in sums:
                new_sums.append(s + v)
            sums.extend(new_sums)
        return sums

    def is_valid(S):
        """Check if all subset sums of S are distinct."""
        sums = get_sums(S)
        return len(sums) == len(set(sums))

    def find_collision(S):
        """Find two distinct subsets A, B with same sum. Returns (A, B) or None."""
        sum_to_sub = {}
        k = len(S)
        elem_list = list(S)
        for mask in range(1, 1 << k):
            s = 0
            subset = []
            for i in range(k):
                if mask & (1 << i):
                    s += vals[elem_list[i]]
                    subset.append(elem_list[i])
            if s in sum_to_sub:
                return (sum_to_sub[s], subset)
            else:
                sum_to_sub[s] = subset
        return None

    # Precompute conflict graph: two elements conflict if adding them creates a collision
    # Actually, we need: two elements x, y conflict if {x,y} is NOT valid (has duplicate subset sums)
    # For a set S to be valid, ALL subsets must have distinct sums.
    # Two elements x,y are "compatible" if {x,y} is valid.
    # But we need higher-order compatibility too. For MIS, we model it as:
    # Each element is a vertex. Two vertices are connected if they CANNOT coexist in a valid set.
    # But "cannot coexist" depends on the full set, not just the pair.
    
    # Alternative: Use the fact that a set is dissociated iff no non-trivial linear combination with coefficients in {-1,0,1} gives 0.
    # For reciprocal sums: sum_{i in A} 1/a_i = sum_{i in B} 1/b_i where A != B
    # This is equivalent to: there exist disjoint non-empty A, B subset of S such that sum_A = sum_B
    
    # For the MIS approach, we can use a simpler model:
    # Precompute for each pair (x,y) whether {x,y} is valid.
    # If not, they conflict.
    # This gives a necessary but not sufficient condition.
    # We'll use this as a heuristic upper bound in branch-and-bound.
    
    # Let's precompute pairwise compatibility
    n = 22
    # Element indices 0..21 correspond to values 1..22
    elem_vals = [D // (i+1) for i in range(n)]
    
    # Pairwise conflict: x and y conflict if {x,y} has duplicate subset sums
    # Subset sums of {x,y}: 0, vx, vy, vx+vy
    # They're distinct iff vx != vy and vx != 0 and vy != 0 and vx+vy != 0
    # Since all vals are positive, vx+vy > 0, and vx, vy > 0
    # So conflict iff vx == vy
    # But vx = D/x, vy = D/y, so vx == vy iff x == y
    # So no pairwise conflicts! That means the conflict graph is empty at the pair level.
    
    # The conflicts are higher-order. We need a different approach.
    
    # Let's use a different strategy: branch-and-bound where we build the set incrementally
    # and check validity at each step.
    
    best_set = []
    best_size = 0
    
    # Greedy Removal Strategy
    for restart in range(20):
        if time.time() > deadline:
            break
            
        S = set(all_elements)
        
        heuristic = restart % 3
        
        while time.time() < deadline:
            if is_valid(S):
                changed = True
                while changed and time.time() < deadline:
                    changed = False
                    for y in all_elements:
                        if y not in S:
                            new_S = S | {y}
                            if is_valid(new_S):
                                S = new_S
                                changed = True
                                break
                                                    
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = list(S)
                
                if len(S) == 22:
                    return list(S)
                
                if len(S) < 22:
                    if random.random() < 0.5:
                        to_remove = random.choice(list(S))
                        S.discard(to_remove)
                        changed = True
                        while changed and time.time() < deadline:
                            changed = False
                            for y in all_elements:
                                if y not in S:
                                    new_S = S | {y}
                                    if is_valid(new_S):
                                        S = new_S
                                        changed = True
                                        break
                        if len(S) > best_size:
                            best_size = len(S)
                            best_set = list(S)
                    else:
                        break
            else:
                collision = find_collision(S)
                if collision is None:
                    break
                A, B = collision
                C = (set(A) ^ set(B))
                if not C:
                    break
                
                if heuristic == 0:
                    x_to_remove = max(C, key=lambda x: x)
                elif heuristic == 1:
                    x_to_remove = min(C, key=lambda x: x)
                else:
                    x_to_remove = random.choice(list(C))
                
                S.discard(x_to_remove)
        
    # Now try a more systematic branch-and-bound approach
    if time.time() < deadline - 5:
        # Use a recursive search with pruning
        # State: current set S, remaining candidates R
        # Try to add elements from R to S, checking validity
        
        def search(S, R, depth):
            """Try to find a larger valid set by adding elements from R to S."""
            nonlocal best_set, best_size, deadline
            
            if time.time() > deadline:
                return
            
            if len(S) + len(R) <= best_size:
                return  # Can't beat best
            
            # Try adding each element from R
            for i, y in enumerate(R):
                if time.time() > deadline:
                    return
                
                new_S = S | {y}
                if is_valid(new_S):
                    # Update best
                    if len(new_S) > best_size:
                        best_size = len(new_S)
                        best_set = list(new_S)
                    
                    # Recurse
                    remaining = [z for z in R if z != y]
                    search(new_S, remaining, depth + 1)
            
            # Also consider not adding any more elements
            # (already handled by the base case)
        
        # Start with a small seed set and try to grow
        # Try different starting points
        for start_size in [1, 2, 3]:
            if time.time() > deadline - 5:
                break
            
            # Try all combinations of start_size elements
            from itertools import combinations
            for combo in combinations(all_elements, start_size):
                if time.time() > deadline - 5:
                    break
                
                S = set(combo)
                if is_valid(S):
                    R = [y for y in all_elements if y not in S]
                    # Sort R by some heuristic (e.g., smaller x first = larger value)
                    R.sort(key=lambda x: x)
                    search(S, R, 0)
    
    if not best_set:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
