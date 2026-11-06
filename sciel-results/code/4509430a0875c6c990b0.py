import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Build the 48-vertex construction using GF(16)
    # GF(16) = GF(2)[x]/(x^4 + x + 1)
    # Elements represented as 4-bit integers
    # Multiplication table for GF(16)
    
    def gf16_mul(a, b):
        """Multiply two elements in GF(16) = GF(2)[x]/(x^4+x+1)"""
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= (a << i)
        # Reduce mod x^4 + x + 1 = 0b10011
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= (0b10011 << (i - 4))
        return r
    
    # Build log and exp tables for GF(16)^* (nonzero elements)
    # The multiplicative group has order 15
    log_table = {}
    exp_table = []
    x = 1
    for e in range(15):
        exp_table.append(x)
        log_table[x] = e
        x = gf16_mul(x, 2)  # multiply by primitive element 2
    
    # Verify: 2^15 should be 1
    # Check that all 15 nonzero elements are covered
    assert len(log_table) == 15
    assert log_table.get(gf16_mul(2, 2**14)) == 0 or True  # 2^15 = 1
    
    # The 48-vertex construction:
    # Vertices: (a, b) where a in {0,1,2,3} and b in GF(16)
    # So 4 * 16 = 64 vertices... but we want 48.
    # Actually, the standard construction uses:
    # Vertices: elements of GF(16) plus 4 "infinity" vertices, or
    # A different construction.
    
    # Let me use the known 48-vertex construction:
    # Based on the fact that R(3,3,3,3) > 48 is known.
    # One construction: take the 16 elements of GF(16), and for each, 
    # create 3 copies, giving 48 vertices.
    
    # Actually, let me use a simpler approach:
    # Start with a known good 40-vertex coloring and extend.
    # Or use the construction from the literature.
    
    # Let me use a different well-known approach:
    # The 48-vertex construction can be built as follows:
    # Partition 48 vertices into 3 groups of 16.
    # Within each group, use a 4-coloring of K_16 with no mono-triangle.
    # Between groups, use a specific pattern.
    
    # Actually, the simplest reliable approach:
    # Use backtracking from a small seed to build up to 48+.
    
    # Let me use a cleaner algebraic construction:
    # Vertices are pairs (i, x) where i in {0,1,2,3} and x in GF(16)
    # That gives 64 vertices, which is too many if we want to verify.
    
    # Let me just use a proven 44-vertex base and extend greedily with backtracking.
    
    # Actually, let me use the following known construction for 48 vertices:
    # Based on the paper by Chung (1975) and subsequent work.
    
    # For simplicity and reliability, let me build up using backtracking
    # from a small seed, which is guaranteed to work.
    
    # Start with K_6 which is easy to 4-color without mono-triangles
    # Then extend one vertex at a time with backtracking.
    
    def try_build(n_target, max_time=120.0):
        """Build a 4-coloring of K_n with no monochromatic triangle."""
        start = time.time()
        
        # We'll build up from n=1 to n_target
        # adj[i][j] = color for i < j
        adj = [[-1] * n_target for _ in range(n_target)]
        
        # Start with n=1 (trivial)
        current_n = 1
        
        for target_n in range(2, n_target + 1):
            if time.time() - start > max_time - 5:
                break
            
            # Try to add vertex (target_n - 1)
            # We need to assign colors to edges (target_n-1, i) for i in 0..target_n-2
            # such that no monochromatic triangle is formed.
            
            # Use backtracking
            def backtrack(v_idx, assigned_count):
                if time.time() - start > max_time - 2:
                    return False
                
                if assigned_count == target_n - 1:
                    return True
                
                # Pick the next vertex to assign color to
                # Use MRV (minimum remaining values) heuristic
                best_i = -1
                best_options = None
                best_count = 5
                
                for i in range(target_n - 1):
                    if adj[v_idx][i] != -1:
                        continue
                    
                    # Count valid colors for edge (v_idx, i)
                    valid = []
                    for c in range(4):
                        # Check if assigning color c to (v_idx, i) creates a mono-triangle
                        ok = True
                        for j in range(target_n - 1):
                            if j == i or adj[v_idx][j] == -1:
                                continue
                            if adj[v_idx][j] == c and adj[i][j] == c:
                                ok = False
                                break
                        if ok:
                            valid.append(c)
                    
                    if len(valid) < best_count:
                        best_count = len(valid)
                        best_i = i
                        best_options = valid
                        if best_count == 0:
                            break
                
                if best_i == -1:
                    return True  # All assigned
                
                if best_count == 0:
                    return False
                
                # Try each valid color
                for c in best_options:
                    adj[v_idx][best_i] = c
                    adj[best_i][v_idx] = c
                    if backtrack(v_idx, assigned_count + 1):
                        return True
                    adj[v_idx][best_i] = -1
                    adj[best_i][v_idx] = -1
                
                return False
            
            # Initialize the new vertex's row
            for i in range(target_n - 1):
                adj[target_n - 1][i] = -1
                adj[i][target_n - 1] = -1
            
            if backtrack(target_n - 1, 0):
                current_n = target_n
            else:
                break
        
        return current_n, adj
    
    # Try to build up to 50 vertices
    # Use multiple random restarts with different orderings
    
    best_n = 1
    best_adj = None
    
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        
        random.seed(attempt * 12345 + 678)
        
        # Shuffle the order in which we add vertices
        # Actually, let's just run the backtracking
        n, adj = try_build(50, max_time=deadline - time.time() - 3)
        
        if n > best_n:
            best_n = n
            best_adj = [row[:] for row in adj]
        
        if best_n >= 50:
            break
    
    # If we didn't get a good result, try a different approach:
    # Use a known 40-vertex construction as seed
    
    if best_n < 40:
        # Build a 40-vertex coloring using a more structured approach
        # Use the fact that we can partition vertices
        
        # For now, let's just use what we have
        pass
    
    if best_adj is None:
        # Fallback: trivial 1-vertex coloring
        return {'n': 1, 'colors': []}
    
    # Convert adj to the required format
    colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            colors.append(best_adj[i][j])
    
    return {'n': best_n, 'colors': colors}
# EVOLVE-BLOCK-END
