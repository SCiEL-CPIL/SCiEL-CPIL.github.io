import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Construct GF(49) and 4-coloring
    # Field GF(7)[x]/(x^2+1)
    # Elements represented as (a, b) for a + b*i
    # Multiplication: (a,b)*(c,d) = (ac - bd, ad + bc) mod 7
    
    def gf49_mul(p, q):
        a, b = p
        c, d = q
        return ((a*c - b*d) % 7, (a*d + b*c) % 7)

    def gf49_add(p, q):
        return ((p[0] + q[0]) % 7, (p[1] + q[1]) % 7)

    def gf49_sub(p, q):
        return ((p[0] - q[0]) % 7, (p[1] - q[1]) % 7)

    def gf49_pow(base, exp):
        result = (1, 0)
        base = base
        while exp:
            if exp & 1:
                result = gf49_mul(result, base)
            base = gf49_mul(base, base)
            exp >>= 1
        return result

    # Generate all elements
    elements = []
    for a in range(7):
        for b in range(7):
            elements.append((a, b))
    
    # Map element to index
    elem_to_idx = {e: i for i, e in enumerate(elements)}
    
    # Find primitive element
    # Order of GF(49)* is 48.
    # Try (2,1)
    alpha = (2, 1)
    # Verify alpha is primitive (order 48)
    # alpha^24 should be -1 (6,0)
    # alpha^16, alpha^12, alpha^6, alpha^3, alpha^2, alpha^1 should not be 1
    
    # Subgroup H of order 4: {x | x^4 = 1}
    # H = {1, -1, i, -i}
    H = [(1, 0), (6, 0), (0, 1), (0, 6)]
    
    # Cosets C0, C1, C2, C3
    # C_k = alpha^k * H
    cosets = []
    for k in range(4):
        ak = gf49_pow(alpha, k)
        coset = set()
        for h in H:
            prod = gf49_mul(ak, h)
            coset.add(prod)
        cosets.append(coset)
    
    # Assign color to each non-zero element
    color_map = {}
    zero = (0, 0)
    for e in elements:
        if e == zero:
            continue
        for c in range(4):
            if e in cosets[c]:
                color_map[e] = c
                break
    
    # Construct K_49
    n = 49
    graph = [[0] * n for _ in range(n)]
    
    # Vertices 0..48 correspond to elements 0..48
    for i in range(n):
        vi = elements[i]
        for j in range(i + 1, n):
            vj = elements[j]
            diff = gf49_sub(vi, vj)
            c = color_map[diff]
            graph[i][j] = c
            graph[j][i] = c
            
    def format_result(n, graph):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(graph[i][j])
        return {'n': n, 'colors': colors}
    
    best_result = format_result(n, graph)
    current_n = n
    
    # 2. Try to extend to n=50 using fast backtracking
    # Time limit for this phase: 10 seconds
    ext_deadline = start_time + 10
    
    def is_valid_graph(n, graph):
        for i in range(n):
            for j in range(i + 1, n):
                c = graph[i][j]
                for k in range(j + 1, n):
                    if graph[i][k] == c and graph[j][k] == c:
                        return False
        return True

    # Bitmask backtracking for adding vertex
    # For each existing vertex i, we have a set of forbidden colors for the new vertex?
    # No, constraint is: if graph[i][j] == c, then NOT (new[i]==c AND new[j]==c).
    # This is a constraint satisfaction problem.
    
    def try_add_vertex_fast(n, graph):
        # n is current size
        # We want to assign colors to edges (i, n) for i in 0..n-1
        # Precompute for each pair (i, j) < n, the color c_ij
        # Constraint: for all i < j < n, if c_ij == c, then not (x_i == c and x_j == c)
        
        # We can use a recursive search with forward checking.
        # To make it fast, we can use bitmasks for allowed colors.
        
        # For a partial assignment x[0..k-1], for each unassigned vertex j >= k,
        # the set of allowed colors is reduced by constraints from assigned i < k.
        # If x_i == c, then for any j such that graph[i][j] == c, x_j cannot be c.
        
        allowed = [[1, 1, 1, 1] for _ in range(n)] # bitmask 1111 initially
        
        def backtrack(k):
            if k == n:
                return True
            
            # Heuristic: pick variable with fewest remaining values?
            # For simplicity, natural order, but we can check if any is empty.
            
            # Check if any unassigned has 0 options
            # This is expensive to check every time.
            
            # Try colors for vertex k
            # Order colors by least constraining value?
            # Just try 0,1,2,3
            
            # Determine which colors are allowed for k based on previous assignments
            # Actually, 'allowed' array is updated as we go.
            
            # We need to maintain allowed correctly.
            # Let's use a different approach:
            # For each i, maintain a bitmask of forbidden colors for x_i?
            # No, x_i is a single color.
            
            # Let's stick to simple backtracking but with pruning.
            # If time is tight, this might not finish.
            
            # Optimized:
            # For each i < k, x_i is fixed.
            # For each j >= k, if x_i == graph[i][j], then x_j != graph[i][j].
            
            # We can precompute for each i, the list of j such that graph[i][j] == c.
            
            # Let's just do a quick check:
            # If we've spent too much time, return False.
            if time.time() > ext_deadline:
                return False
            
            # Try each color for x[k]
            for c in range(4):
                # Check if c is allowed for k
                # c is forbidden if there exists i < k such that x[i] == c and graph[i][k] == c
                valid = True
                for i in range(k):
                    if graph[i][k] == c and assignment[i] == c:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                assignment[k] = c
                
                # Recurse
                if backtrack(k + 1):
                    return True
                    
                # Undo
                assignment[k] = -1
                
            return False
        
        assignment = [-1] * n
        
        # Randomize vertex order?
        # The structure of GF(49) is symmetric.
        # Adding vertex 49.
        # We can try to add it.
        
        # To improve speed, we can try a few random restarts or just one.
        # Given the symmetry, one attempt is often enough if it's solvable.
        # If not solvable, we need to modify the base graph.
        
        if backtrack(0):
            # Success
            for i in range(n):
                graph[i][n] = assignment[i]
                graph[n][i] = assignment[i]
            return True
        return False

    # Try to extend
    if time.time() < ext_deadline:
        # Make a copy of graph to avoid modifying the best_result if it fails?
        # try_add_vertex_fast modifies graph in place if successful.
        # We should work on a copy or revert if it fails?
        # The function only modifies if it returns True.
        # If it returns False, graph is unchanged (except assignment array which is local).
        
        if try_add_vertex_fast(current_n, graph):
            current_n += 1
            # Resize graph
            new_graph = [[0] * (current_n + 1) for _ in range(current_n + 1)]
            for i in range(current_n):
                for j in range(current_n):
                    new_graph[i][j] = graph[i][j]
            graph = new_graph
            best_result = format_result(current_n, graph)

    return best_result
