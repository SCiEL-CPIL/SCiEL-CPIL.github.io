import time
import random
import sys

_START_TIME = time.time()
_DEADLINE = 115.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _seed_32():
    """Greenwood-Gleason colouring of K32."""
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
        
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
        
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return {'n': 32, 'colors': colors}

def _compute_index(n, i, j):
    """Compute the index of pair (i,j) in lexicographic edge ordering."""
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

def _build_adj_bitsets(colors, n):
    """Build adjacency bitsets for a given coloring. adj[c][i] is bitmask of neighbors of i in color c."""
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj

def _extend_vertex(colors, n, adj):
    """
    Try to extend the coloring by one vertex (vertex n).
    Returns new colors list (length (n+1)*n//2) and new adj if successful, else None.
    Uses backtracking to assign colors to edges (n, u) for u in 0..n-1.
    """
    # We need to assign color c_u for edge (n, u) for each u in 0..n-1.
    # Constraint: For any u < w, if c_u == c_w == c, then (u, w) must NOT be color c.
    # This means the set {u | c_u == c} must be an independent set in G_c.
    
    # Pre-compute neighbors for each color
    # adj[c][u] is bitmask of neighbors of u in color c.
    
    # We will assign colors to u=0..n-1.
    # To speed up, we can use a greedy approach with backtracking.
    # Order of u: We can pick u with the fewest remaining options (MRV heuristic).
    
    # For simplicity and speed, let's try a random greedy order first, with restarts if it fails.
    # But backtracking is safer.
    
    # Let's use a recursive backtracking solver.
    # State: assigned colors for a subset of u's.
    # Candidates for each color c: bitmask of u's that can still be assigned color c.
    
    # Initial candidates: all u's are candidates for all colors.
    # But we must satisfy: if u is assigned c, then no neighbor of u in G_c can be assigned c.
    
    # Let's define the state as:
    # assigned: list of length n, -1 if unassigned, else color.
    # candidates: list of 4 bitmasks. candidates[c] has bit u set if u can be assigned color c.
    
    # We pick an unassigned u.
    # For each possible color c where candidates[c] has bit u:
    #   Assign u to c.
    #   Update candidates:
    #     candidates[c] &= ~((1 << u) | adj[c][u])  # u and its neighbors in G_c can no longer be c
    #     # Wait, we also need to ensure that if we assign u to c, it doesn't conflict with PREVIOUSLY assigned vertices?
    #     # Yes, but if we maintain candidates correctly, previously assigned vertices have been removed from candidates.
    #     # However, we must also check that u is compatible with previously assigned vertices.
    #     # If w is assigned c, and (u,w) is color c, then conflict.
    #     # This is handled if we remove neighbors from candidates when we assign.
    #     # But we also need to check: if u is assigned c, does it conflict with w assigned c?
    #     # If w was assigned c earlier, then u would have been removed from candidates[c] when w was assigned (because u is a neighbor of w in G_c? No, only if (u,w) is color c).
    #     # Yes, if (u,w) is color c, then u is in adj[c][w]. So when w was assigned c, u was removed from candidates[c].
    #     # So if u is in candidates[c], it means it is not adjacent to any previously assigned vertex of color c in G_c.
    
    # So the logic is:
    # 1. Start with candidates[c] = (1<<n)-1 for all c.
    # 2. Pick unassigned u.
    # 3. For each c in 0..3:
    #    if candidates[c] has bit u:
    #       Assign u to c.
    #       new_candidates = candidates copy.
    #       new_candidates[c] &= ~((1 << u) | adj[c][u])
    #       # Also, we must remove u from candidates of OTHER colors? No, u is assigned, so it's done.
    #       # But we need to mark u as assigned.
    #       Recurse.
    
    # To avoid deep recursion and copying, we can use a trail or just copy the bitmask list (small).
    
    def solve(assigned, candidates):
        # Find unassigned u
        unassigned_mask = 0
        for c in range(4):
            unassigned_mask |= candidates[c]
        # Actually, unassigned are those not in any assigned list?
        # Better: keep a bitmask of assigned vertices.
        # Let's pass assigned_mask as well.
        
        # For simplicity in this implementation, let's just pick the first unassigned u.
        # This is less efficient but easier to code correctly.
        
        # Find first u not assigned
        # We can track assigned_mask
        pass
        
    # Let's implement a simpler iterative backtracking with a stack.
    # Or just use a recursive function with bitmasks.
    
    def backtrack(assigned_mask, candidates, assignments):
        # assigned_mask: bitmask of vertices already assigned a color
        # candidates: list of 4 bitmasks
        # assignments: list of length n, color for each vertex
        
        if assigned_mask == (1 << n) - 1:
            return True
        
        # Pick an unassigned vertex u
        # MRV: pick u with fewest candidate colors
        unassigned = ~assigned_mask & ((1 << n) - 1)
        if unassigned == 0:
            return True
            
        # Find u with min count of candidates
        best_u = -1
        best_count = 5
        u_mask = unassigned
        while u_mask:
            u_bit = u_mask & (-u_mask)
            u = u_bit.bit_length() - 1
            count = sum(1 for c in range(4) if (candidates[c] >> u) & 1)
            if count < best_count:
                best_count = count
                best_u = u
            if count == 0:
                return False # Dead end
            u_mask ^= u_bit
            
        u = best_u
        u_bit = 1 << u
        
        # Try each color
        for c in range(4):
            if (candidates[c] >> u) & 1:
                # Assign u to c
                assignments[u] = c
                new_candidates = candidates[:]
                # Remove u and its neighbors in G_c from candidates[c]
                new_candidates[c] &= ~((1 << u) | adj[c][u])
                
                # Also, we must ensure that u is not in candidates of other colors?
                # No, u is assigned, so it shouldn't be considered again.
                # But we don't explicitly remove u from other candidates.
                # However, the check `(candidates[c] >> u) & 1` will fail for other colors if we don't remove it?
                # No, we need to remove u from ALL candidates to mark it as assigned?
                # Actually, if we don't, we might try to assign u to another color later?
                # No, `assigned_mask` will include u.
                # So we just need to update `assigned_mask`.
                
                if backtrack(assigned_mask | u_bit, new_candidates, assignments):
                    return True
                    
                # Backtrack
                assignments[u] = -1
        return False

    assignments = [-1] * n
    initial_candidates = [(1 << n) - 1] * 4
    # Note: We must ensure that the initial state is valid.
    # It is, because no vertices are assigned.
    
    if backtrack(0, initial_candidates, assignments):
        # Success: construct new colors list
        new_colors = colors[:]
        # Append colors for edges (n, u)
        # Lexicographic order: (0,1)...(0,n-1), (1,2)...(1,n-1) ... (n-1, n)
        # The new edges are (0, n), (1, n), ..., (n-1, n).
        # In the new list of size (n+1)*n//2, the last n elements are these edges.
        # Index of (u, n) in new list:
        # Total edges before row u: u * n - u*(u+1)//2 ?
        # Let's just append in order u=0 to n-1.
        # The lexicographic order of pairs (i,j) with i<j.
        # Pairs involving n are (0,n), (1,n), ..., (n-1, n).
        # These are the last n pairs in lexicographic order?
        # (0,1), (0,2), ..., (0,n-1), (1,2), ..., (1,n-1), ..., (n-2, n-1), (0,n)? No.
        # Lexicographic: (0,1), (0,2), ..., (0,n-1), (1,2), ..., (1,n-1), ..., (n-1, n)?
        # No, standard lexicographic is (0,1), (0,2)...(0,n-1), (1,2)...(1,n-1)...(n-2,n-1).
        # When we add vertex n, the new pairs are (0,n), (1,n), ..., (n-1, n).
        # Where do they fit?
        # (0,n) comes after (0,n-1).
        # (1,n) comes after (1,n-1).
        # So they are interleaved.
        
        # Easier: Rebuild the entire colors list? No, O(n^2).
        # Let's compute the indices.
        # Index of (i, j) in K_n (n vertices): i*(n-1) - i*(i+1)//2 + (j-i-1).
        # In K_{n+1}, index of (i, j): i*n - i*(i+1)//2 + (j-i-1).
        
        # We have the old colors for K_n.
        # We need to insert the new edges.
        # It is easier to create a new list of size (n+1)*n//2.
        # And fill it.
        
        new_m = (n + 1) * n // 2
        new_colors = [0] * new_m
        idx = 0
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if j == n:
                    # Edge (i, n)
                    new_colors[idx] = assignments[i]
                else:
                    # Edge (i, j) from old coloring
                    old_idx = i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
                    new_colors[idx] = colors[old_idx]
                idx += 1
        
        # Rebuild adj for new n
        new_adj = _build_adj_bitsets(new_colors, n + 1)
        return new_colors, new_adj
        
    return None

def construct():
    try:
        best = _seed_32()
        best_n = 32
        colors = best['colors']
        adj = _build_adj_bitsets(colors, 32)
        
        # Try to extend to 50
        target_n = 50
        current_n = 32
        
        # Time budget
        start_time = time.time()
        
        while current_n < target_n:
            if time.time() - start_time > _DEADLINE - 5:
                break
                
            # Try to extend with a few random restarts if it fails
            success = False
            for attempt in range(100):
                # Shuffling the order of u in the backtracking might help?
                # My current backtracking uses MRV, which is deterministic.
                # If it fails, it's likely a dead end.
                # We can randomize the initial candidates or the tie-breaking in MRV.
                
                res = _extend_vertex(colors, current_n, adj)
                if res is not None:
                    colors, adj = res
                    current_n += 1
                    success = True
                    break
                else:
                    # If it fails, we might need to perturb the previous step?
                    # For now, just break and return best found.
                    # A more robust approach would be to backtrack to the previous vertex and try a different assignment.
                    # This is complex.
                    # Given the time, we assume the greedy/MRV backtracking will find a path.
                    break
            
            if not success:
                break
                
        if current_n > best_n:
            return {'n': current_n, 'colors': colors}
        else:
            return best
            
    except Exception as e:
        try:
            return _seed_32()
        except:
            return {'n': 1, 'colors': []}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
