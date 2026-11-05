import time
import sys

# Increase recursion limit if necessary
sys.setrecursionlimit(2000)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe, limit is 150s
    
    best_n = 9
    best_colors = None
    
    # Seed solution for n=9
    # K9 split by circular distance: distances 1, 2, 4 give nonagons, distance 3 gives three triangles.
    n_seed = 9
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    # Verify seed is valid? The prompt says it is.
    # Let's assume it is valid and set as initial best.
    best_n = n_seed
    best_colors = colors_seed

    # Try to find larger n
    # We iterate n from 10 up to 20
    # If we find a solution for n, we update best and try n+1
    # If we fail for n within time limit, we stop and return best.
    
    for n in range(10, 21):
        if time.time() > deadline - 10:
            break
        
        solution = solve_n(n, deadline)
        if solution is not None:
            best_n = n
            best_colors = solution
            # Continue to try for larger n
        else:
            # If no solution found for n, it might be due to time limit or infeasibility.
            # Given R_4(C6) > 17 is unknown, 18, 19, 20 are likely infeasible or very hard.
            # If we run out of time or hit a hard wall, we stop.
            # Since we don't know if it's infeasible or just slow, and we want to maximize n,
            # and 17 is the best known, trying 18+ is a gamble.
            # If solve_n(10) fails quickly, we might try 11.
            # If solve_n(10) takes too long, we break.
            # Let's assume if it fails, we break to avoid wasting time on harder instances.
            # Actually, if n=10 is infeasible, n=11 is too. If n=10 is just hard, n=11 is harder.
            # So breaking is a reasonable heuristic.
            break
            
    return {'n': best_n, 'colors': best_colors}

def solve_n(n, deadline):
    total_edges = n * (n - 1) // 2
    colors = [-1] * total_edges
    
    # Map (i, j) to index
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
        # Formula for lexicographic index of (i,j) in 0..n-1:
        # Sum of edges for vertices 0..i-1: i*(n-1) - i*(i-1)/2
        # Plus offset j-i-1
        # Let's verify for n=3:
        # (0,1): 0*(2) - 0 + 0 = 0
        # (0,2): 0*(2) - 0 + 1 = 1
        # (1,2): 1*(2) - 1 + 0 = 1. Correct.
        # (1,3) for n=4: 1*(3) - 0 + 1 = 4?
        # Edges: (0,1)=0, (0,2)=1, (0,3)=2, (1,2)=3, (1,3)=4, (2,3)=5.
        # Formula for (1,3): i=1, j=3.
        # i*(n-1) - i*(i-1)/2 + (j-i-1)
        # 1*3 - 0 + (3-1-1) = 3 + 1 = 4. Correct.
        
    # Bitmasks for adjacency for each color
    # adj[c][u] is bitmask of neighbors of u using color c
    adj = [[0] * n for _ in range(4)]
    
    # Symmetry breaking:
    # Fix colors of edges (0,1), (0,2), (0,3) to 0, 1, 2 respectively.
    # This is valid because colors are symmetric.
    # If n < 4, adjust.
    
    # We will process edges in lexicographic order.
    # We need to skip the fixed edges or handle them specially.
    # Easier: Initialize the fixed edges in adj and colors, and start backtracking from the first non-fixed edge.
    
    fixed_edges = []
    if n >= 2:
        fixed_edges.append(((0, 1), 0))
    if n >= 3:
        fixed_edges.append(((0, 2), 1))
    if n >= 4:
        fixed_edges.append(((0, 3), 2))
        
    # Apply fixed edges
    for (u, v), c in fixed_edges:
        idx = edge_idx(u, v)
        colors[idx] = c
        adj[c][u] |= (1 << v)
        adj[c][v] |= (1 << u)
        
    # Determine start index for backtracking
    start_idx = 0
    for (u, v), c in fixed_edges:
        idx = edge_idx(u, v)
        if idx > start_idx:
            start_idx = idx + 1
            
    # If n is small, start_idx might be total_edges
    if start_idx >= total_edges:
        # Check if the fixed edges form a valid C6-free graph
        # For n<=4, no C6 possible.
        return colors.copy()

    # Helper to check if adding edge (u,v) with color c creates a C6
    def creates_c6(u, v, c):
        # Check if there is a path of length 5 between u and v in adj[c]
        # Path: u - a - b - d - e - v
        # a in N(u), e in N(v)
        # Path a-b-d-e in G \ {u,v}
        
        # Get neighbors of u and v in color c
        # Exclude v from N(u) and u from N(v) just in case, though they aren't connected yet
        Nu = adj[c][u]
        Nv = adj[c][v]
        
        # Iterate over potential a in Nu
        # We can optimize by iterating over bits
        a_mask = Nu
        while a_mask:
            a_bit = a_mask & -a_mask
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            
            # a cannot be v (it's not, since edge not added)
            # a cannot be u (simple path)
            if a == v or a == u:
                continue
                
            # Iterate over potential e in Nv
            e_mask = Nv
            while e_mask:
                e_bit = e_mask & -e_mask
                e = e_bit.bit_length() - 1
                e_mask ^= e_bit
                
                if e == u or e == v:
                    continue
                if a == e:
                    continue
                
                # Check if there is a path of length 3 between a and e in G \ {u,v}
                # Path a - b - d - e
                # b in N(a), d in N(e), b in N(d)
                # b, d not in {u, v, a, e}
                
                forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << e)
                
                # Get neighbors of a and e, excluding forbidden
                Na = adj[c][a] & ~forbidden
                Ne = adj[c][e] & ~forbidden
                
                # Iterate b in Na
                b_mask = Na
                while b_mask:
                    b_bit = b_mask & -b_mask
                    b = b_bit.bit_length() - 1
                    b_mask ^= b_bit
                    
                    # b must be neighbor of some d in Ne, and b must be neighbor of d
                    # So d must be in N(b) and N(e)
                    # d in N(b) & N(e) & ~forbidden
                    # Also d != b
                    
                    potential_ds = adj[c][b] & Ne & ~forbidden
                    # d cannot be b
                    potential_ds &= ~(1 << b)
                    
                    if potential_ds:
                        return True
        
        return False

    # Recursive backtracking
    def backtrack(idx):
        if time.time() > deadline:
            return False, None
            
        if idx == total_edges:
            return True, colors.copy()
            
        # Get edge (u, v) for index idx
        # Reverse of edge_idx
        # We can precompute a list of edges
        # To save time, let's precompute edges list
        # edges[i] = (u, v)
        
        u, v = edges[idx]
        
        # Try colors 0, 1, 2, 3
        for c in range(4):
            if creates_c6(u, v, c):
                continue
                
            # Assign color c
            colors[idx] = c
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            
            success, res = backtrack(idx + 1)
            if success:
                return True, res
                
            # Backtrack
            colors[idx] = -1
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)
            
        return False, None

    # Precompute edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            
    # Start backtracking
    success, res = backtrack(start_idx)
    if success:
        return res
    else:
        return None

if __name__ == "__main__":
    # For local testing
    res = construct()
    print(f"Best n: {res['n']}")
    # print(res['colors'])
