import time
import random
import sys
import math

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    # Fast subset sum computation
    def compute_sums(elements):
        """Compute all non-empty subset sums. Returns set or None if duplicates."""
        sums = set()
        for x in elements:
            new_sums = set()
            for s in sums:
                ns = s + x
                if ns in sums:
                    return None
                new_sums.add(ns)
            if x in sums:
                return None
            new_sums.add(x)
            sums = sums.union(new_sums)
        return sums

    # Check divisibility constraint on a set of sums
    def check_divisibility(sums):
        """Check no two distinct sums divide each other. Returns True if valid."""
        s_list = sorted(sums)
        n = len(s_list)
        for i in range(n):
            si = s_list[i]
            for j in range(i+1, n):
                sj = s_list[j]
                if sj % si == 0:
                    return False
        return True

    # Full validation
    def is_valid(elements):
        if not elements:
            return True
        if len(elements) != len(set(elements)):
            return False
        if any(x < 1 or x > 256 for x in elements):
            return False
        sums = compute_sums(elements)
        if sums is None:
            return False
        return check_divisibility(sums)

    # Build compatibility graph using bitsets
    # Two numbers a, b are compatible if adding b to a set containing a doesn't create violations
    # We need to check: for any subset S of {a,b}, the sums of S and any superset don't divide each other
    # Simplified: a and b are compatible if:
    # 1. a != b
    # 2. a+b is not a multiple of a or b (i.e., a doesn't divide a+b and b doesn't divide a+b)
    #    Actually: a divides a+b iff a divides b, and b divides a+b iff b divides a
    # 3. a doesn't divide b and b doesn't divide a
    # 4. No other constraints for pairs (but for larger sets, we need to check all subset sums)
    
    # For the clique approach, we need a pairwise compatibility that's necessary but not sufficient
    # for the full set constraint. Let's use a stronger condition:
    # a and b are compatible if:
    # - a != b
    # - a does not divide b and b does not divide a
    # - (a+b) is not a multiple of a (i.e., a doesn't divide b) - already covered
    # - (a+b) is not a multiple of b (i.e., b doesn't divide a) - already covered
    # - a doesn't divide (a+b) - always true since a+b > a
    # - b doesn't divide (a+b) - always true since a+b > b
    # Wait, a divides a+b iff a divides b. So the key constraint is:
    # a does not divide b AND b does not divide a
    
    # But this is only necessary, not sufficient. For a set {a,b,c}, we also need:
    # - a doesn't divide a+c, a doesn't divide b+c, etc.
    # This makes the graph approach tricky. Let's use a different approach.
    
    # Alternative: Use the graph where vertices are 1-256, and edge (a,b) exists if
    # the set {a,b} is valid. Then use branch-and-bound to find the largest clique,
    # but verify the full set is valid at the end.
    
    # Build pairwise compatibility
    n = 256
    # adj[i] is a bitset (Python int) of vertices compatible with vertex i (0-indexed, so vertex i+1)
    adj = [0] * n
    
    for i in range(n):
        a = i + 1
        for j in range(i+1, n):
            b = j + 1
            # Check if {a, b} is a valid set
            # Subset sums: a, b, a+b
            # Check: no two divide each other
            # a doesn't divide b, b doesn't divide a, a doesn't divide a+b (true), b doesn't divide a+b (true)
            # a+b doesn't divide a (false, a+b > a), a+b doesn't divide b (false)
            # So just need: a % b != 0 and b % a != 0
            if a % b != 0 and b % a != 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Tomita's maximum clique algorithm with bitset operations
    best_clique = []
    best_size = 0
    
    def popcount(x):
        return bin(x).count('1')
    
    def lmax_clique(R, P, Q):
        """
        R: current clique (list of vertex indices)
        P: candidates that can extend R
        Q: candidates that can extend R and have been considered
        """
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        # Pruning: if |R| + |P| <= best_size, no need to continue
        if len(R) + popcount(P) <= best_size:
            return
        
        if P == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
        
        # Color sort for better pruning
        # Simple version: just iterate through P
        while P != 0:
            if time.time() > deadline:
                return
            
            # Pick a vertex from P
            # Use a heuristic: pick the vertex with highest degree in P
            v = _pick_vertex(P)
            
            P &= ~(1 << v)
            
            # Recurse with v added to clique
            R.append(v)
            P_new = P & adj[v]
            lmax_clique(R, P_new, P)
            R.pop()
            
            # If P becomes empty, break
            if P == 0:
                break
    
    def _pick_vertex(P):
        """Pick a vertex from P using a simple heuristic."""
        # For now, just pick the lowest set bit
        # v = (P & -P).bit_length() - 1
        # Better: pick vertex with max degree in P
        best_v = -1
        best_deg = -1
        temp = P
        while temp != 0:
            v = (temp & -temp).bit_length() - 1
            deg = popcount(P & adj[v])
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp &= temp - 1
        return best_v if best_v >= 0 else 0
    
    # Run maximum clique search
    initial_P = (1 << n) - 1  # All vertices
    lmax_clique([], initial_P, initial_P)
    
    # Convert clique to actual values
    if best_clique:
        candidate = [v + 1 for v in best_clique]
        candidate.sort()
        # Verify the full set is valid
        if is_valid(candidate):
            return candidate
        else:
            # The clique may not be fully valid due to higher-order constraints
            # Try to fix by removing problematic elements
            # Or fall back to a greedy approach
            pass
    
    # Fallback: greedy construction
    def greedy_construct():
        current = []
        sums = set()
        
        # Try numbers in decreasing order (larger numbers have fewer divisibility issues)
        for x in range(256, 0, -1):
            if time.time() > deadline:
                break
            
            # Try adding x
            new_sums = set()
            valid = True
            for s in sums:
                ns = s + x
                if ns in sums or ns in new_sums:
                    valid = False
                    break
                new_sums.add(ns)
            if not valid:
                continue
            
            if x in sums or x in new_sums:
                continue
            
            all_sums = sums.union(new_sums)
            all_sums.add(x)
            
            # Quick divisibility check
            s_list = sorted(all_sums)
            for i in range(len(s_list)):
                si = s_list[i]
                for j in range(i+1, len(s_list)):
                    sj = s_list[j]
                    if sj % si == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current.append(x)
                sums = all_sums
        
        return current
    
    result = greedy_construct()
    
    if not result:
        # Fallback valid sets
        fallbacks = [
            [255, 256],
            [254, 256],
            [253, 255],
            [250, 251],
            [200, 201],
            [100, 101],
            [100],
        ]
        for fs in fallbacks:
            if is_valid(fs):
                result = fs
                break
    
    if not result:
        result = [100]
    
    # Ensure valid Python ints in range
    result = [int(x) for x in result if 1 <= x <= 256]
    seen = set()
    unique = []
    for x in result:
        if x not in seen:
            seen.add(x)
            unique.append(x)
    result = unique
    
    return result
