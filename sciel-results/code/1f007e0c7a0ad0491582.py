import time
import random
import numpy as np

def _is_valid_fast(matrix, n):
    """
    Check if the coloring of K_n defined by matrix (n x n, upper triangle valid) 
    has no monochromatic triangles.
    matrix[i][j] is the color for edge (i,j).
    """
    for c in range(4):
        # Create adjacency matrix for color c
        A = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i][j] == c:
                    A[i, j] = 1
                    A[j, i] = 1
        
        # Check for triangles: 
        # A triangle exists if there are u, v, w such that A[u,v]=A[v,w]=A[w,u]=1.
        # This is equivalent to: There exists an edge (u,v) in A such that 
        # (A @ A)[u,v] > 0.
        # Because (A @ A)[u,v] is the number of common neighbors of u and v.
        # If u and v are connected (A[u,v]=1) and have a common neighbor w, 
        # then u-v-w-u is a triangle.
        
        # Calculate A^2
        A2 = A @ A
        
        # Check if any edge (u,v) where A[u,v]=1 has A2[u,v] > 0
        # Note: A2 is symmetric. We can just check all entries.
        # If A[u,v] == 1 and A2[u,v] > 0, we have a triangle.
        
        # Vectorized check:
        # mask = (A == 1) & (A2 > 0)
        # if np.any(mask): return False
        
        # Optimization: A2[u,v] counts paths of length 2.
        # If A[u,v]=1, and A2[u,v] >= 1, then triangle exists.
        
        if np.any((A == 1) & (A2 > 0)):
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Base 32-vertex construction
    # Using the standard Clebsch-based construction
    # Vertices 0..15 and 16..31
    # Within block: color(i,j) = log2(i^j) % 3 (using GF(16) discrete log)
    # Cross block: color 3
    
    # Build GF(16) log table
    log_table = {}
    x = 1
    log_table[1] = 0
    for e in range(1, 15):
        x2 = 0
        for i in range(4):
            if (x >> i) & 1:
                x2 ^= (1 << (i + 1))
        if x2 & 0b10000:
            x2 ^= 0b10011
        x = x2
        log_table[x] = e
        
    def _build_base(n_base):
        matrix = [[0] * n_base for _ in range(n_base)]
        for i in range(n_base):
            for j in range(i + 1, n_base):
                if i // 16 == j // 16:
                    c = log_table[i % 16 ^ j % 16] % 3
                else:
                    c = 3
                matrix[i][j] = c
                matrix[j][i] = c
        return matrix

    current_n = 32
    current_matrix = _build_base(32)
    
    # Verify base is valid (should be)
    if not _is_valid_fast(current_matrix, current_n):
        # Fallback to a smaller valid one if base is broken (unlikely)
        current_n = 16
        current_matrix = _build_base(16)

    best_n = current_n
    best_matrix = current_matrix

    def _try_extend(matrix, n):
        """
        Try to extend matrix (n x n) to (n+1) x (n+1).
        Returns new matrix if successful, else None.
        """
        new_n = n + 1
        # Initialize new matrix
        new_matrix = [row[:] for row in matrix]
        for i in range(n):
            new_matrix[i].append(0)
        new_matrix.append([0] * new_n)
        
        new_vertex = new_n - 1
        assign = [0] * n
        
        # Constraints for new vertex:
        # For each pair (u, v) among 0..n-1:
        # If matrix[u][v] == c, then we cannot have assign[u]==c AND assign[v]==c.
        
        # We use backtracking with forward checking / simple pruning
        # To keep it fast, we limit the search or use randomization if it gets stuck.
        
        # Pre-compute constraints:
        # For each color c, and each vertex u, 
        # if assign[u] == c, then for all v where matrix[u][v] == c, assign[v] != c.
        
        def backtrack(idx):
            if idx == n:
                return True
            
            # Heuristic: try colors that are most constrained? 
            # Or just 0,1,2,3
            for c in range(4):
                valid = True
                # Check against all previously assigned vertices
                for j in range(idx):
                    if assign[j] == c and matrix[idx][j] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                assign[idx] = c
                if backtrack(idx + 1):
                    return True
            
            return False

        # Backtracking can be exponential. 
        # For n=32, adding 1 vertex involves 32 variables.
        # However, the constraints are dense.
        # If it takes too long, we should abort.
        # We will implement a simple time check inside the loop? 
        # No, Python function call overhead is high.
        # Instead, we run backtracking, but if it fails quickly, we try random restarts.
        
        # Actually, for this specific structure (Clebsch based), extension might be hard.
        # Let's try a simpler heuristic: Random assignment + Local Repair.
        
        # 1. Randomly assign colors to edges (new_vertex, i)
        for _ in range(100): # 100 attempts
            for i in range(n):
                assign[i] = random.randint(0, 3)
            
            # Check validity
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    if assign[i] == assign[j] and assign[i] == matrix[i][j]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                for i in range(n):
                    new_matrix[i][new_vertex] = assign[i]
                    new_matrix[new_vertex][i] = assign[i]
                return new_matrix
        
        # 2. If random fails, try backtracking with a node limit
        # Implementing a limited backtracker
        nodes_visited = 0
        limit = 10000
        
        def backtrack_limited(idx):
            nonlocal nodes_visited
            if nodes_visited > limit:
                return False
            nodes_visited += 1
            
            if idx == n:
                return True
            
            # Shuffle colors to avoid bias
            colors = [0, 1, 2, 3]
            random.shuffle(colors)
            
            for c in colors:
                valid = True
                for j in range(idx):
                    if assign[j] == c and matrix[idx][j] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                assign[idx] = c
                if backtrack_limited(idx + 1):
                    return True
            
            return False

        if backtrack_limited(0):
            for i in range(n):
                new_matrix[i][new_vertex] = assign[i]
                new_matrix[new_vertex][i] = assign[i]
            return new_matrix

        return None

    # Main loop
    while time.time() < deadline:
        new_matrix = _try_extend(current_matrix, current_n)
        if new_matrix is not None:
            current_matrix = new_matrix
            current_n += 1
            best_n = current_n
            best_matrix = current_matrix
        else:
            # Stuck. Try random perturbation?
            # Or just break?
            # Since we want to maximize n, and extension is hard, 
            # maybe we should try to restart from a slightly smaller valid state?
            # Or just accept we are stuck.
            # Given the 120s limit, and that we are likely stuck at 32 or 33,
            # we should stop trying to extend if we fail a few times.
            break

    # Format output
    out_colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            out_colors.append(best_matrix[i][j])

    return {'n': best_n, 'colors': out_colors}
