import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Helper to get color of edge (i, j) from list
    def list_to_matrix(colors, n):
        mat = [[-1]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                mat[i][j] = colors[idx]
                mat[j][i] = colors[idx]
                idx += 1
        return mat

    def matrix_to_list(mat, n):
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(mat[i][j])
        return colors

    # Count mono triangles for local search
    def count_mono_triangles(mat, n):
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                c = mat[i][j]
                for k in range(j+1, n):
                    if mat[i][k] == c and mat[j][k] == c:
                        count += 1
        return count

    # Local search: Min-conflicts with delta evaluation
    def local_search(mat, n, max_iter=5000, deadline_check=None):
        current_cost = count_mono_triangles(mat, n)
        if current_cost == 0:
            return True, mat
            
        # Precompute for faster delta calculation
        # For each edge (i,j) and color c, count how many k form a mono triangle with (i,j)
        # This is expensive to precompute fully, so we'll do on-the-fly
        
        for _ in range(max_iter):
            if deadline_check and time.time() > deadline_check:
                return False, mat
                
            # Find a random mono triangle
            # Sample random triples until we find one
            found = False
            for _ in range(20):  # Try up to 20 random samples
                i = random.randint(0, n-1)
                j = random.randint(i+1, n-1)
                k = random.randint(j+1, n-1)
                c1 = mat[i][j]
                c2 = mat[i][k]
                c3 = mat[j][k]
                if c1 == c2 == c3:
                    # Found mono triangle (i,j,k) with color c1
                    # Randomly choose one of the 3 edges to recolor
                    edges = [(i,j), (i,k), (j,k)]
                    e1, e2 = random.choice(edges)
                    old_c = mat[e1][e2]
                    
                    # Try each of the other 3 colors
                    best_new_c = -1
                    best_new_cost = current_cost
                    for new_c in range(4):
                        if new_c == old_c:
                            continue
                        mat[e1][e2] = new_c
                        mat[e2][e1] = new_c
                        new_cost = count_mono_triangles(mat, n)
                        if new_cost < best_new_cost:
                            best_new_cost = new_cost
                            best_new_c = new_c
                        mat[e1][e2] = old_c
                        mat[e2][e1] = old_c
                    
                    if best_new_c != -1:
                        mat[e1][e2] = best_new_c
                        mat[e2][e1] = best_new_c
                        current_cost = best_new_cost
                        if current_cost == 0:
                            return True, mat
                        found = True
                        break
                    else:
                        # No improvement, try another random triangle
                        continue
            
            if not found:
                # Stuck, try a random perturbation
                for _ in range(10):
                    i = random.randint(0, n-1)
                    j = random.randint(i+1, n-1)
                    old_c = mat[i][j]
                    new_c = (old_c + 1 + random.randint(0, 2)) % 4
                    mat[i][j] = new_c
                    mat[j][i] = new_c
                    new_cost = count_mono_triangles(mat, n)
                    if new_cost <= current_cost:
                        current_cost = new_cost
                        if current_cost == 0:
                            return True, mat
                    else:
                        mat[i][j] = old_c
                        mat[j][i] = old_c
                        if new_cost < current_cost:
                            current_cost = new_cost
                            if current_cost == 0:
                                return True, mat
                        else:
                            mat[i][j] = old_c
                            mat[j][i] = old_c
        
        return current_cost == 0, mat

    # Start with random 4-coloring of K_50
    n_target = 50
    best_n = 0
    best_colors = None
    
    # Try multiple random initializations
    for attempt in range(30):
        if time.time() > deadline - 5.0:
            break
            
        # Random initialization
        mat = [[-1]*n_target for _ in range(n_target)]
        for i in range(n_target):
            for j in range(i+1, n_target):
                c = random.randint(0, 3)
                mat[i][j] = c
                mat[j][i] = c
        
        # Local search
        valid, mat = local_search(mat, n_target, max_iter=3000, deadline_check=deadline - 5.0)
        
        if valid:
            best_n = n_target
            best_colors = matrix_to_list(mat, n_target)
            break
    
    # If we didn't get 50, try 49, 48, etc.
    if best_n < n_target:
        for n_try in range(49, 30, -1):
            if time.time() > deadline - 5.0:
                break
            for attempt in range(20):
                if time.time() > deadline - 5.0:
                    break
                mat = [[-1]*n_try for _ in range(n_try)]
                for i in range(n_try):
                    for j in range(i+1, n_try):
                        c = random.randint(0, 3)
                        mat[i][j] = c
                        mat[j][i] = c
                
                valid, mat = local_search(mat, n_try, max_iter=2000, deadline_check=deadline - 5.0)
                if valid:
                    best_n = n_try
                    best_colors = matrix_to_list(mat, n_try)
                    break
            if best_n == n_try:
                break
    
    # Fallback: use the known 32-vertex Greenwood-Gleason coloring
    if best_n < 32:
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
            
        base_colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                if i // 16 == j // 16:
                    base_colors.append(log[(i % 16) ^ (j % 16)] % 3)
                else:
                    base_colors.append(3)
                    
        best_n = 32
        best_colors = base_colors

    return {'n': best_n, 'colors': best_colors}
