"""
4-coloring of K_n for R(3,3,3,3) > n.
Uses a robust backtracking solver to find a valid 3-coloring of K_16,
then blows it up to K_48, and attempts to extend to K_50.
"""

import time
import random
import sys

sys.setrecursionlimit(10000)


def _verify_no_mono_triangle(adj, n):
    """Verify no monochromatic triangle in a 4-coloring."""
    for c in range(4):
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j] != c:
                    continue
                for k in range(j + 1, n):
                    if adj[i][k] == c and adj[j][k] == c:
                        return False
    return True


def _find_3coloring_k16(deadline):
    """
    Find a 3-coloring of K_16 with no monochromatic triangle.
    Uses backtracking with forward checking.
    """
    n = 16
    # adj[i][j] = color (0,1,2) or -1
    adj = [[-1] * n for _ in range(n)]
    
    # Order vertices to improve backtracking (e.g., by degree in the constraint graph, but here it's complete)
    # We just go 0..15
    
    # Precompute constraints:
    # When coloring edge (i, j) with color c, we must ensure no triangle (i, j, k) is monochromatic c.
    # This means if adj[i][k] == c and adj[j][k] == c, we cannot color (i,j) with c.
    # Or more simply, when we assign colors, we check consistency.
    
    def is_valid_assignment(i, j, c):
        """Check if coloring edge (i, j) with c is valid given current assignments."""
        # Check for k < i or k < j or k > max(i,j)?
        # Actually, we can just check all k != i, j.
        # If adj[i][k] == c and adj[j][k] == c, then triangle (i,j,k) would be mono c if (i,j) is c.
        # So if both are already c, we cannot assign c to (i,j).
        for k in range(n):
            if k == i or k == j:
                continue
            if adj[i][k] == c and adj[j][k] == c:
                return False
        return True

    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n * (n - 1) // 2:
            return True
        
        i = idx // (n - 1 - idx // n) # This mapping is tricky, let's use explicit loops
        # Better: use a stack or explicit i,j loops
        
        # Let's use a simple recursive structure with i, j
        # We need to map linear index to (i,j)
        # Or just loop i, j in the recursion.
        
        # Refactor: use a list of edges
        # Or simpler: 
        # Find next uncolored edge (i, j)
        # This is slow if done naively.
        
        # Let's use a specific ordering: 
        # Color edges (0,1), (0,2)...(0,15), (1,2)...
        
        # To avoid complexity, let's use a different approach:
        # Construct the 3-coloring using the known sum-free partition of Z_2^4.
        # This is deterministic and fast.
        
        return True # Placeholder, we will replace this with the deterministic construction below

    # Deterministic Construction for K_16 3-coloring:
    # Vertices are 0..15 (4-bit integers).
    # We partition the non-zero differences into 3 sum-free sets.
    # One known partition for Z_2^4:
    # S0 = {0001, 0010, 0100, 1000, 1111} ? 
    # Let's verify sum-free property for S0:
    # 0001+0010=0011 (not in S0)
    # 0001+0100=0101 (not in S0)
    # 0001+1000=1001 (not in S0)
    # 0001+1111=1110 (not in S0)
    # 0010+0100=0110 (not in S0)
    # 0010+1000=1010 (not in S0)
    # 0010+1111=1101 (not in S0)
    # 0100+1000=1100 (not in S0)
    # 0100+1111=1011 (not in S0)
    # 1000+1111=0111 (not in S0)
    # So S0 is sum-free.
    
    # Now we need S1 and S2 to cover the rest and be sum-free.
    # Remaining: 15 - 5 = 10 elements.
    # Rest = {0011, 0101, 1001, 1110, 0110, 1010, 1101, 1100, 1011, 0111}
    
    # Let's try S1 = {0011, 0101, 1001, 1100, 0111}
    # Check S1 sum-free:
    # 0011+0101=0110 (not in S1)
    # 0011+1001=1010 (not in S1)
    # 0011+1100=1111 (in S0, ok)
    # 0011+0111=0100 (in S0, ok)
    # 0101+1001=1100 (in S1! BAD)
    # So S1 is not sum-free.
    
    # Let's use a solver to find the partition.
    # Since n=16 is small, we can brute force the partition of 15 elements into 3 sets of size 5.
    # Number of ways: 15! / (5!5!5!) = 756756. This is small enough to brute force with pruning.
    
    elements = [i for i in range(1, 16)]
    # We need to assign each element to 0, 1, or 2.
    # Constraint: No x, y, z in same set with x+y=z (and x,y,z distinct? No, x+y=z implies distinct in Z_2^4 unless x=0 or y=0, but 0 is not in the set).
    # Actually, x+y=z means x,y,z are distinct non-zero elements.
    
    import itertools
    
    best_coloring = None
    # Use a greedy backtracking to assign sets
    sets = [[], [], []]
    
    def assign(idx):
        if time.time() > deadline:
            return False
        if idx == 15:
            # Check if all sets are sum-free
            for s in sets:
                s_set = set(s)
                for i in range(len(s)):
                    for j in range(i+1, len(s)):
                        if s[i] ^ s[j] in s_set:
                            return False
            # Valid
            nonlocal best_coloring
            # Build adj matrix
            adj_local = [[0]*16 for _ in range(16)]
            for i in range(16):
                for j in range(i+1, 16):
                    diff = i ^ j
                    for c in range(3):
                        if diff in sets[c]:
                            adj_local[i][j] = c
                            adj_local[j][i] = c
                            break
            best_coloring = adj_local
            return True
        return False
        
    # This simple assign won't work because it doesn't backtrack properly on the sets.
    # Let's do a proper backtracking.
    
    def backtrack_sets(idx):
        if time.time() > deadline:
            return False
        if idx == 15:
            # Verify
            for c in range(3):
                s_set = set(sets[c])
                for x in sets[c]:
                    for y in sets[c]:
                        if x != y and (x ^ y) in s_set:
                            return False
            nonlocal best_coloring
            adj_local = [[0]*16 for _ in range(16)]
            for i in range(16):
                for j in range(i+1, 16):
                    diff = i ^ j
                    for c in range(3):
                        if diff in sets[c]:
                            adj_local[i][j] = c
                            adj_local[j][i] = c
                            break
            best_coloring = adj_local
            return True
            
        elem = elements[idx]
        for c in range(3):
            # Pruning: Check if adding elem to sets[c] creates a sum-free violation
            # Violation if there exists x in sets[c] such that x ^ elem is also in sets[c]
            # Or elem ^ x is in sets[c]
            ok = True
            for x in sets[c]:
                if (x ^ elem) in sets[c]:
                    ok = False
                    break
            if ok:
                sets[c].append(elem)
                if backtrack_sets(idx + 1):
                    return True
                sets[c].pop()
        return False

    if backtrack_sets(0):
        return best_coloring
    
    # Fallback: Random search if backtracking fails (unlikely)
    for _ in range(1000):
        if time.time() > deadline:
            break
        random.shuffle(elements)
        # Re-run backtrack with shuffled order? 
        # Or just random assignment
        sets = [[], [], []]
        for e in elements:
            # Find a valid set
            valid_sets = []
            for c in range(3):
                ok = True
                for x in sets[c]:
                    if (x ^ e) in sets[c]:
                        ok = False
                        break
                if ok:
                    valid_sets.append(c)
            if not valid_sets:
                break
            c = random.choice(valid_sets)
            sets[c].append(e)
        else:
            # Check final validity
            valid = True
            for c in range(3):
                s_set = set(sets[c])
                for x in sets[c]:
                    for y in sets[c]:
                        if x != y and (x ^ y) in s_set:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                adj_local = [[0]*16 for _ in range(16)]
                for i in range(16):
                    for j in range(i+1, 16):
                        diff = i ^ j
                        for c in range(3):
                            if diff in sets[c]:
                                adj_local[i][j] = c
                                adj_local[j][i] = c
                                break
                return adj_local
    return None


def _construct_50():
    t0 = time.time()
    deadline = t0 + 110.0
    
    base_adj = _find_3coloring_k16(deadline)
    if base_adj is None:
        # Emergency fallback: return a small valid graph
        # K_4 is 4-colorable
        n = 4
        adj = [[0]*n for _ in range(n)]
        # 0-1:0, 0-2:1, 0-3:2, 1-2:3, 1-3:0, 2-3:1 (Check triangles)
        # This is getting risky. Let's just return n=1 if all else fails.
        return {'n': 1, 'colors': []}
        
    # Verify base
    if not _verify_no_mono_triangle(base_adj, 16):
        return {'n': 1, 'colors': []}
        
    # Blow up to 48
    n = 48
    adj = [[0] * n for _ in range(n)]
    
    for i in range(16):
        for j in range(i + 1, 16):
            c = base_adj[i][j]
            for a in range(3):
                for b in range(3):
                    vi = i * 3 + a
                    vj = j * 3 + b
                    adj[vi][vj] = c
                    adj[vj][vi] = c
    
    # Intra-group edges
    for i in range(16):
        v0 = i * 3
        v1 = i * 3 + 1
        v2 = i * 3 + 2
        adj[v0][v1] = adj[v1][v0] = 0
        adj[v0][v2] = adj[v2][v0] = 1
        adj[v1][v2] = adj[v2][v1] = 2
        
    if not _verify_no_mono_triangle(adj, 48):
        return {'n': 1, 'colors': []}
        
    # Try to extend to 50
    n_final, adj_final = _try_extend(adj, 48, max_new=2, deadline=deadline)
    
    if n_final < 48:
        n_final, adj_final = 48, adj
    
    colors = []
    for i in range(n_final):
        for j in range(i + 1, n_final):
            colors.append(adj_final[i][j])
            
    return {'n': n_final, 'colors': colors}


def _try_extend(adj, n, max_new=2, deadline=None):
    current_adj = [row[:] for row in adj]
    current_n = n
    best_n = n
    
    for new_v in range(max_new):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        found = False
        for attempt in range(200):
            if deadline is not None and time.time() > deadline - 5.0:
                break
                
            colors_new = [-1] * current_n
            valid = True
            
            for i in range(current_n):
                available = []
                for col in range(4):
                    ok = True
                    for j in range(i):
                        if colors_new[j] == col and current_adj[i][j] == col:
                            for k in range(j):
                                if colors_new[k] == col and current_adj[j][k] == col and current_adj[i][k] == col:
                                    ok = False
                                    break
                        if not ok:
                            break
                    if ok:
                        available.append(col)
                
                if not available:
                    valid = False
                    break
                colors_new[i] = random.choice(available)
            
            if valid:
                # Verify
                good = True
                for col in range(4):
                    grp = [i for i in range(current_n) if colors_new[i] == col]
                    for a in range(len(grp)):
                        for b in range(a+1, len(grp)):
                            if current_adj[grp[a]][grp[b]] == col:
                                good = False
                                break
                        if not good:
                            break
                    if not good:
                        break
                if good:
                    for i in range(current_n):
                        current_adj[i].append(colors_new[i])
                    current_adj.append(colors_new[:] + [0])
                    current_n += 1
                    best_n = current_n
                    found = True
                    break
        if not found:
            break
    
    return best_n, [row[:best_n] for row in current_adj[:best_n]]


def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    return _construct_50()
