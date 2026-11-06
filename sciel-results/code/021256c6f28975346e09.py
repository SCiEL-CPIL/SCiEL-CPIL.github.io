"""Bitset-based backtracking solver for R(K2,2,2) > 30."""
import numpy as np
import time

def construct():
    """
    Find a 2-coloring of K_n with no monochromatic K_2,2,2, maximizing n.
    Uses bitset-based backtracking with targeted K_2,2,2 detection.
    """
    
    def check_k222_with_edge(adj, i, j, n):
        """Check if edge (i,j) in adj (list of int bitmasks) participates in a K_2,2,2.
        In K_2,2,2, i and j are in different parts. We need to find 4 more vertices
        completing the K_2,2,2 structure."""
        # i in part A, j in part B. Need a1 in A (a1!=i), b1 in B (b1!=j), c1,c2 in C.
        # Conditions: a1 adj to j,c1,c2; b1 adj to i,c1,c2; c1 adj to c2;
        # a1 adj to b1; c1,c2 adj to i,j,a1,b1.
        # 
        # Simplified: c1,c2 must be in N(i)∩N(j)∩N(a1)∩N(b1) and c1-c2 edge.
        # a1 in N(j)\\{i}, b1 in N(i)\\{j}, a1-b1 edge.
        
        # Get N(i) and N(j) as bitmasks
        Ni = adj[i]
        Nj = adj[j]
        
        # a1 candidates: neighbors of j, not i
        a1_candidates = Nj & ~(1 << i)
        # b1 candidates: neighbors of i, not j
        b1_candidates = Ni & ~(1 << j)
        
        # For each (a1, b1) pair where a1-b1 edge exists:
        # c1,c2 must be in N(i)∩N(j)∩N(a1)∩N(b1) and c1-c2 edge.
        
        # Precompute N(i)∩N(j)
        common_ij = Ni & Nj
        common_ij &= ~((1 << i) | (1 << j))
        
        # Iterate over a1 candidates
        mask_a1 = a1_candidates
        while mask_a1:
            a1 = (mask_a1 & -mask_a1).bit_length() - 1
            mask_a1 &= mask_a1 - 1
            
            # b1 must be adjacent to a1
            b1_valid = b1_candidates & adj[a1]
            if not b1_valid:
                continue
            
            # c1,c2 must be in common_ij ∩ N(a1) ∩ N(b1)
            # But N(b1) depends on b1, so iterate over b1 too
            mask_b1 = b1_valid
            while mask_b1:
                b1 = (mask_b1 & -mask_b1).bit_length() - 1
                mask_b1 &= mask_b1 - 1
                
                common_all = common_ij & adj[a1] & adj[b1]
                common_all &= ~((1 << a1) | (1 << b1))
                
                # Check if common_all contains an edge (c1-c2)
                mask_c = common_all
                while mask_c:
                    c1 = (mask_c & -mask_c).bit_length() - 1
                    mask_c &= mask_c - 1
                    # c2 must be in common_all, adjacent to c1, and c2 > c1 to avoid dup
                    c2_candidates = common_all & adj[c1] & ~((1 << c1))
                    # Only need to check if any c2 exists
                    if c2_candidates:
                        return True
        
        return False
    
    def full_k222_check(adj, n):
        """Full check for K_2,2,2 using bitsets."""
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj[a] >> b) & 1:
                    continue
                for c in range(b + 1, n):
                    if not (adj[a] >> c) & 1 or not (adj[b] >> c) & 1:
                        continue
                    common = adj[a] & adj[b] & adj[c]
                    common &= ~((1 << a) | (1 << b) | (1 << c))
                    # Check if common contains an edge
                    mask = common
                    while mask:
                        v = (mask & -mask).bit_length() - 1
                        mask &= mask - 1
                        if common & adj[v] & ~((1 << v)):
                            return True
        return False
    
    def solve(n, timeout=100):
        """Backtracking with bitsets and targeted K_2,2,2 check."""
        start_time = time.time()
        adj0 = [0] * n
        adj1 = [0] * n
        colors = [[-1] * n for _ in range(n)]
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        edge_idx = [0]
        
        def backtrack():
            if time.time() - start_time > timeout:
                return False
            if edge_idx[0] >= len(edges):
                return True
            
            i, j = edges[edge_idx[0]]
            edge_idx[0] += 1
            
            # Try color 0
            adj0[i] |= (1 << j)
            adj0[j] |= (1 << i)
            colors[i][j] = colors[j][i] = 0
            
            if not check_k222_with_edge(adj0, i, j, n):
                if backtrack():
                    return True
            
            adj0[i] &= ~(1 << j)
            adj0[j] &= ~(1 << i)
            
            # Try color 1
            adj1[i] |= (1 << j)
            adj1[j] |= (1 << i)
            colors[i][j] = colors[j][i] = 1
            
            if not check_k222_with_edge(adj1, i, j, n):
                if backtrack():
                    return True
            
            adj1[i] &= ~(1 << j)
            adj1[j] &= ~(1 << i)
            
            colors[i][j] = colors[j][i] = -1
            edge_idx[0] -= 1
            return False
        
        if backtrack():
            result = []
            for i in range(n):
                for j in range(i + 1, n):
                    result.append(colors[i][j])
            return {'n': n, 'colors': result}
        return None
    
    # Try n=30
    result = solve(30, timeout=95)
    if result is not None:
        return result
    
    # Try n=29
    result = solve(29, timeout=20)
    if result is not None:
        return result
    
    # Fallback: n=17 Paley coloring
    p = 17
    squares = {x * x % p for x in range(1, p)}
    colors = []
    for i in range(p):
        for j in range(i + 1, p):
            colors.append(0 if (j - i) % p in squares else 1)
    return {'n': p, 'colors': colors}

# EVOLVE-BLOCK-END
