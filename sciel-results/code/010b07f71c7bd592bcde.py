import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    # Fallback: Paley graph on 17 vertices
    def get_paley_17():
        squares = {x * x % 17 for x in range(1, 17)}
        edges = []
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in squares:
                    edges.append([i, j])
        return {'n': 17, 'edges': edges}

    best_result = get_paley_17()

    # Step 1: Try to find a 35-vertex graph
    # Allocate 20 seconds for this
    t1 = time.time()
    graph_35 = try_find_graph_optimized(35, t1 + 20)
    if graph_35:
        if is_valid(graph_35):
            best_result = graph_35

    # Step 2: Try to extend 35 to 36
    if graph_35 and time.time() < deadline - 5:
        graph_36 = try_extend_to_36(graph_35, deadline)
        if graph_36:
            if is_valid(graph_36):
                return graph_36

    # If 35 failed, try 34?
    if not graph_35:
        t2 = time.time()
        graph_34 = try_find_graph_optimized(34, deadline)
        if graph_34:
            if is_valid(graph_34):
                best_result = graph_34

    return best_result

def is_valid(graph):
    n = graph['n']
    edges = graph['edges']
    A = [[0]*n for _ in range(n)]
    for u, v in edges:
        A[u][v] = 1
        A[v][u] = 1

    # Check K4
    for i in range(n):
        for j in range(i+1, n):
            if A[i][j]:
                for k in range(j+1, n):
                    if A[i][k] and A[j][k]:
                        for l in range(k+1, n):
                            if A[i][l] and A[j][l] and A[k][l]:
                                return False

    # Check Independent 6-set
    for i in range(n):
        for j in range(i+1, n):
            if not A[i][j]:
                for k in range(j+1, n):
                    if not A[i][k] and not A[j][k]:
                        for l in range(k+1, n):
                            if not A[i][l] and not A[j][l] and not A[k][l]:
                                for m in range(l+1, n):
                                    if not A[i][m] and not A[j][m] and not A[k][m] and not A[l][m]:
                                        for p in range(m+1, n):
                                            if not A[i][p] and not A[j][p] and not A[k][p] and not A[l][p] and not A[m][p]:
                                                return False
    return True

def try_find_graph_optimized(n, deadline):
    max_iters = 5000
    best_score = float('inf')
    best_A = None
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        A = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    A[i][j] = 1
                    A[j][i] = 1
        
        current_score = count_violations(A, n)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            if current_score == 0:
                return graph_from_matrix(A)
            
            # Pick a random violation and fix it
            # ... (simplified for brevity, using random flip if score doesn't improve)
            
            # Flip random edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if u > v: u, v = v, u
            
            A[u][v] = 1 - A[u][v]
            A[v][u] = A[u][v]
            
            new_score = count_violations(A, n)
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_A = [row[:] for row in A]
            else:
                # Revert
                A[u][v] = 1 - A[u][v]
                A[v][u] = A[u][v]

    if best_A and best_score == 0:
        return graph_from_matrix(best_A)
    return None

def count_violations(A, n):
    # Count K4s and I6s
    # This is expensive. Use bitmasks.
    neighbors = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if A[i][j]:
                mask |= (1 << j)
        neighbors[i] = mask
        
    k4_count = 0
    for i in range(n):
        for j in range(i+1, n):
            if A[i][j]:
                common = neighbors[i] & neighbors[j]
                common &= ~(1 << i)
                common &= ~(1 << j)
                # Check for edges in common
                # If there is an edge (k,l) in common, it's a K4
                # We just need to know if there is ANY.
                # But for score, we want count? Or just 0/non-zero?
                # For local search, 0/non-zero is enough.
                # But for SA, count is better.
                # Let's just check if ANY K4 exists.
                temp = common
                found_k4 = False
                while temp:
                    lsb = temp & -temp
                    k = lsb.bit_length() - 1
                    # Check if k has any neighbor in common
                    if neighbors[k] & common:
                        found_k4 = True
                        break
                    temp ^= lsb
                if found_k4:
                    k4_count += 1 # Approximation, or just return 1 if any?
                    # Returning 1 if any is faster.
                    # Let's return 1 if any K4 or I6.
                    # Actually, for SA, we need a gradient.
                    # Let's stick to 0/non-zero for speed in this draft.
    
    # Check I6
    # Complement neighbors
    all_mask = (1 << n) - 1
    comp_neighbors = [0] * n
    for i in range(n):
        comp_neighbors[i] = (~neighbors[i] & all_mask) & ~(1 << i)
        
    # Find I6 (K6 in complement)
    # This is hard to count.
    # Just check if ANY exists.
    # Use backtracking for existence.
    if has_clique_size_6(comp_neighbors, n):
        return 1
    if k4_count > 0:
        return 1
    return 0

def has_clique_size_6(comp_neighbors, n):
    # Simple backtracking
    order = sorted(range(n), key=lambda x: bin(comp_neighbors[x]).count('1'), reverse=True)
    
    def backtrack(clique, candidates):
        if len(clique) == 6:
            return True
        if not candidates:
            return False
        if len(candidates) + len(clique) < 6:
            return False
            
        for idx, v in enumerate(candidates):
            valid = True
            for u in clique:
                if not (comp_neighbors[u] & (1 << v)):
                    valid = False
                    break
            if valid:
                new_candidates = []
                for w in candidates[idx+1:]:
                    if comp_neighbors[v] & (1 << w):
                        new_candidates.append(w)
                if backtrack(clique + [v], new_candidates):
                    return True
        return False
        
    return backtrack([], order)

def graph_from_matrix(A):
    n = len(A)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if A[i][j]:
                edges.append([i, j])
    return {'n': n, 'edges': edges}

def try_extend_to_36(graph_35, deadline):
    n = 35
    A = [[0]*n for _ in range(n)]
    for u, v in graph_35['edges']:
        A[u][v] = 1
        A[v][u] = 1
        
    # Precompute K3s and I5s
    k3s = []
    for i in range(n):
        for j in range(i+1, n):
            if A[i][j]:
                for k in range(j+1, n):
                    if A[i][k] and A[j][k]:
                        k3s.append((i, j, k))
                        
    # Find I5s (K5 in complement)
    comp_neighbors = [0] * n
    all_mask = (1 << n) - 1
    neighbors = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if A[i][j]:
                mask |= (1 << j)
        neighbors[i] = mask
        comp_neighbors[i] = (~mask & all_mask) & ~(1 << i)
        
    i5s = []
    # Backtrack to find all K5s in complement
    def find_k5s(clique, candidates):
        if len(clique) == 5:
            i5s.append(tuple(clique))
            return
        if not candidates:
            return
        if len(candidates) + len(clique) < 5:
            return
        for idx, v in enumerate(candidates):
            valid = True
            for u in clique:
                if not (comp_neighbors[u] & (1 << v)):
                    valid = False
                    break
            if valid:
                new_candidates = []
                for w in candidates[idx+1:]:
                    if comp_neighbors[v] & (1 << w):
                        new_candidates.append(w)
                find_k5s(clique + [v], new_candidates)
                
    order = sorted(range(n), key=lambda x: bin(comp_neighbors[x]).count('1'), reverse=True)
    find_k5s([], order)
    
    # Simulated Annealing on M (35-bit int)
    M = random.getrandbits(n)
    current_energy = calc_energy(M, k3s, i5s)
    best_energy = current_energy
    best_M = M
    
    T = 1.0
    for _ in range(100000):
        if time.time() > deadline:
            break
            
        # Flip a random bit
        k = random.randint(0, n-1)
        M_new = M ^ (1 << k)
        
        # Calculate delta
        # This requires checking K3s and I5s involving k
        # For simplicity, recalculate energy (slow but safe)
        # Optimization: Precompute lists of K3s and I5s for each vertex k
        # ... (skipping optimization for now, relying on speed of calc_energy if it's fast enough? No, 100k iters * 35 checks is fine)
        
        new_energy = calc_energy(M_new, k3s, i5s)
        delta = new_energy - current_energy
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            M = M_new
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_M = M
                if best_energy == 0:
                    # Found valid M
                    edges = graph_35['edges'][:]
                    for i in range(n):
                        if best_M & (1 << i):
                            edges.append([35, i])
                    return {'n': 36, 'edges': edges}
        
        T *= 0.999
        if T < 0.01:
            T = 0.01

    return None

def calc_energy(M, k3s, i5s):
    energy = 0
    # K4 violations: K3 fully in M
    for k3 in k3s:
        i, j, k = k3
        if (M & (1 << i)) and (M & (1 << j)) and (M & (1 << k)):
            energy += 1
    # I6 violations: I5 fully NOT in M
    for i5 in i5s:
        i, j, k, l, m = i5
        if not (M & (1 << i)) and not (M & (1 << j)) and not (M & (1 << k)) and not (M & (1 << l)) and not (M & (1 << m)):
            energy += 1
    return energy
