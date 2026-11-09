import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin

    # Try to find a valid Cayley graph for n=40 down to n=30
    for n in range(40, 29, -1):
        if time.time() > deadline:
            break
        
        # Number of attempts
        # n=40 is hardest, but we try all
        max_attempts = 500 if n >= 38 else 1000
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate a random maximal symmetric sum-free set S in Z_n
            S = _generate_maximal_sum_free(n)
            
            # Check if alpha(G_S) < 10
            # G_S is defined by S. 
            # Independent set in G_S is a set A such that A-A does not intersect S.
            # This is a clique in the complement graph H, where H has edges for differences NOT in S.
            
            if _check_independence_number_less_than(S, n, 10, deadline):
                # Construct edge list
                edge_list = []
                for i in range(n):
                    for j in range(i + 1, n):
                        diff = (j - i) % n
                        if diff in S:
                            edge_list.append([i, j])
                return {'n': n, 'edges': edge_list}
        
        # If n=39 or 40 failed, we might have found a smaller one in previous iterations?
        # No, the loop breaks on return.
        # If we finish the loop for a specific n without returning, we try smaller n.
        
    # Fallback: Andrásfai graph And(13) on n=38
    # And(k) on Z_{3k-1}, S = {1, k+1, 2k+1} mod (3k-1)
    # k=13 -> n=38. S = {1, 14, 27} mod 38.
    # Symmetric? -1=37, -14=24, -27=11.
    # S should be {1, 37, 14, 24, 27, 11}.
    # Let's verify sum-free.
    # 1+1=2 (not in S). 1+11=12. 1+14=15. 1+24=25. 1+27=28. 1+37=0.
    # 11+11=22. 11+14=25. 11+24=35. 11+27=38=0.
    # 14+14=28. 14+24=38=0.
    # 24+24=48=10.
    # 27+27=54=16.
    # 37+37=74=36.
    # It seems sum-free.
    
    n = 38
    S = {1, 11, 14, 24, 27, 37}
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in S:
                edge_list.append([i, j])
                
    return {'n': n, 'edges': edge_list}

def _generate_maximal_sum_free(n):
    """
    Generates a maximal symmetric sum-free set S in Z_n.
    """
    S = set()
    # Possible differences: 1 to n//2
    candidates = list(range(1, n // 2 + 1))
    random.shuffle(candidates)
    
    for d in candidates:
        # d represents the pair {d, n-d}
        # Check if adding d to S keeps it sum-free
        if _can_add_to_sum_free(S, d, n):
            S.add(d)
            if d != n - d:
                S.add(n - d)
            # Note: if d == n-d (only if n even and d=n/2), we add it once.
            # But S is a set of values in 1..n-1.
            # We store both d and n-d in S to make membership check easy.
            
    return S

def _can_add_to_sum_free(S, d, n):
    """
    Check if S U {d, n-d} is sum-free.
    Sum-free means no x, y, z in S' such that x+y=z.
    """
    # New elements to check: d and n-d
    # We need to check if any sum of two elements in S' equals a third element in S'.
    # Since S is sum-free, we only need to check sums involving the new elements.
    
    new_elements = [d, n - d]
    if d == n - d:
        new_elements = [d]
        
    # 1. Check if x + y = new_elem for x, y in S
    # 2. Check if x + new_elem = y for x, y in S
    # 3. Check if new_elem + new_elem = y for y in S
    # 4. Check if new_elem1 + new_elem2 = y for y in S (if len(new_elements)>1)
    
    for x in S:
        for y in S:
            s = (x + y) % n
            if s in new_elements:
                return False
            # Check if x + new_elem = y
            # This is covered by (x + new_elem) % n == y
            # We check this in the next loop or combine.
            
    # Let's be more precise.
    # For each new_elem in new_elements:
    #   If any x, y in S such that (x+y)%n == new_elem: Fail
    #   If any x in S such that (x + new_elem)%n in S: Fail
    
    for ne in new_elements:
        for x in S:
            for y in S:
                if (x + y) % n == ne:
                    return False
            if (x + ne) % n in S:
                return False
                
    # Check sums of new elements with each other
    for i in range(len(new_elements)):
        for j in range(i, len(new_elements)):
            s = (new_elements[i] + new_elements[j]) % n
            if s in S:
                return False
            # If s is another new element?
            # If s is in new_elements, that's fine as long as it's not forming a triangle 0, a, b?
            # Triangle 0, a, b requires a+b=0? No, a+b=c where c is edge.
            # If a+b=c and c is in S, then 0, a, a+b is a triangle.
            # So if s is in S, fail.
            # If s is in new_elements, it means a+b=c where c is also a new edge.
            # Then 0, a, c is a triangle?
            # Edge (0,a) exists. Edge (0,c) exists. Edge (a,c) exists if c-a in S.
            # c-a = b. b is in S. So yes, triangle.
            # So if s is in new_elements, it also fails?
            # Wait. If a+b=c, and a,b,c are all in S.
            # Then 0, a, c is a triangle.
            # So we must fail if s is in S OR s is in new_elements.
            # Actually, if s is in new_elements, it will be added to S.
            # So we just need to check if s is in S (current) OR s is in new_elements.
            # But since we are adding them, we should treat new_elements as part of the set for this check.
            # So if s is in S U new_elements, fail.
            
    # Re-do the check for new+new
    for i in range(len(new_elements)):
        for j in range(i, len(new_elements)):
            s = (new_elements[i] + new_new_elements[j] if False else new_elements[j]) % n
            if s in S or s in new_elements:
                return False

    return True

def _check_independence_number_less_than(S, n, k, deadline):
    """
    Check if the Cayley graph defined by S on Z_n has independence number < k.
    Equivalent to: Complement graph has no clique of size k.
    Complement connection set T = {d in 1..n-1 | d not in S}.
    """
    # Build T
    T = set()
    for d in range(1, n):
        if d not in S:
            T.add(d)
            
    # B&B for clique of size k in graph with connection set T
    # We use a recursive search on the set of vertices.
    # To speed up, we can use bitsets.
    
    # Precompute neighbors for each vertex in the complement graph
    # neighbors[v] is a bitmask of vertices connected to v
    neighbors = [0] * n
    for v in range(n):
        mask = 0
        for u in range(n):
            if u == v: continue
            diff = (u - v) % n
            if diff in T:
                mask |= (1 << u)
        neighbors[v] = mask
        
    # B&B
    # State: current clique (list), candidates (bitmask)
    # We want to find if there is a clique of size k.
    
    def bnb(candidates_mask, current_size):
        if time.time() > deadline:
            return False
        if current_size >= k:
            return True
        
        # Pruning: count bits in candidates_mask
        if current_size + bin(candidates_mask).count('1') < k:
            return False
            
        # Iterate through candidates
        # To improve pruning, we can pick vertices with high degree in the subgraph?
        # For now, just iterate.
        
        # Extract candidates list
        candidates = []
        temp_mask = candidates_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            v = lsb.bit_length() - 1
            candidates.append(v)
            temp_mask ^= lsb
            
        for i in range(len(candidates)):
            v = candidates[i]
            # New candidates: intersection of candidates (after i) and neighbors[v]
            # We only consider candidates with index > v to avoid duplicates?
            # No, standard clique finding uses an ordered list.
            # If we use bitmasks, we can just intersect.
            # But we need to ensure we don't reuse vertices.
            
            # Let's stick to the list-based B&B from the current program but optimized with bitsets for intersection.
            # Actually, for n=40, simple list B&B is fine if we prune well.
            
            # New candidates are those w in candidates[i+1:] such that w is neighbor of v.
            new_candidates_mask = 0
            for j in range(i + 1, len(candidates)):
                w = candidates[j]
                if (neighbors[v] >> w) & 1:
                    new_candidates_mask |= (1 << w)
                    
            if bnb(new_candidates_mask, current_size + 1):
                return True
                
        return False
        
    # Initial candidates: all vertices
    initial_mask = (1 << n) - 1
    return bnb(initial_mask, 0)
