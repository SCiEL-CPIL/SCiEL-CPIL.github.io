import time
import random

def construct():
    n = 18
    start = time.time()
    deadline = start + 135  # 15s safety margin before 150s kill
    
    def compute_L_fast(adj):
        """Compute the largest triangle-free induced subgraph size using backtracking with pruning."""
        # Precompute: for each pair (i,j), tri_common[i][j] = adj[i] & adj[j]
        tri_common = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    tri_common[i][j] = adj[i] & adj[j]
        
        # For a given k, use backtracking to find a k-vertex triangle-free set.
        # A set S is triangle-free iff for every pair (u,v) in S, tri_common[u][v] & S == 0.
        
        def backtrack(k, S, next_v, count):
            """Try to build a triangle-free set of size k, current set S, next vertex >= next_v."""
            if count == k:
                return True
            remaining = n - next_v
            if count + remaining < k:
                return False
            for v in range(next_v, n):
                if S & (1 << v):
                    continue
                # Check if adding v creates a triangle with any pair in S
                temp = S
                valid = True
                while temp:
                    u = (temp & -temp).bit_length() - 1
                    if tri_common[u][v] & S:
                        valid = False
                        break
                    temp &= temp - 1
                if valid:
                    if backtrack(k, S | (1 << v), v + 1, count + 1):
                        return True
            return False
        
        for k in range(n, 2, -1):
            if backtrack(k, 0, 0, 0):
                return k
        return 2
    
    def is_k4_free(adj):
        """Check if graph is K4-free using bitmasks."""
        for a in range(n):
            for b in range(a+1, n):
                if not (adj[a] & (1 << b)): continue
                for c in range(b+1, n):
                    if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)): continue
                    common = adj[a] & adj[b] & adj[c]
                    if common:
                        return False
        return True
    
    def make_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    random.seed(42)
    
    def random_k4_free(density=0.5):
        adj = [0]*n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Repair to be K4-free
        changed = True
        iters = 0
        while changed and iters < 500:
            changed = False
            iters += 1
            for a in range(n):
                if changed: break
                for b in range(a+1, n):
                    if changed: break
                    if not (adj[a] & (1 << b)): continue
                    for c in range(b+1, n):
                        if changed: break
                        if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)): continue
                        common = adj[a] & adj[b] & adj[c]
                        if common:
                            d = (common & -common).bit_length() - 1
                            adj[a] &= ~(1 << d)
                            adj[d] &= ~(1 << a)
                            changed = True
        return adj
    
    # Start with Turan(18,3)
    adj_turan = [0]*n
    for i in range(n):
        for j in range(i+1, n):
            pi, pj = i % 3, j % 3
            if pi != pj:
                adj_turan[i] |= (1 << j)
                adj_turan[j] |= (1 << i)
    
    L_turan = compute_L_fast(adj_turan)
    best_L = L_turan
    best_adj = adj_turan[:]
    
    print(f"Turan L = {L_turan}", flush=True)
    
    # Try random K4-free graphs with various densities
    densities = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for density in densities:
        if time.time() > deadline - 10:
            break
        for trial in range(30):
            if time.time() > deadline - 10:
                break
            adj = random_k4_free(density)
            if not is_k4_free(adj):
                continue
            L = compute_L_fast(adj)
            if L < best_L:
                best_L = L
                best_adj = adj[:]
                print(f"New best L = {L}, density {density}, trial {trial}", flush=True)
    
    # Local search: try edge flips to reduce L
    if best_adj is not None:
        current_adj = best_adj[:]
        current_L = best_L
        
        # Multiple rounds of local search
        for round_num in range(50):
            if time.time() > deadline - 5:
                break
            improved = False
            # Shuffle edge order for diversity
            edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(edges_to_try)
            
            for i, j in edges_to_try:
                if time.time() > deadline - 5:
                    break
                # Try flipping edge (i,j)
                old_state = bool(current_adj[i] & (1 << j))
                
                if old_state:
                    current_adj[i] &= ~(1 << j)
                    current_adj[j] &= ~(1 << i)
                else:
                    current_adj[i] |= (1 << j)
                    current_adj[j] |= (1 << i)
                
                if is_k4_free(current_adj):
                    L = compute_L_fast(current_adj)
                    if L < current_L:
                        current_L = L
                        best_L = L
                        best_adj = [a | 0 for a in current_adj]
                        improved = True
                        print(f"Local search round {round_num}: improved L to {L}", flush=True)
                    else:
                        # Revert
                        if old_state:
                            current_adj[i] |= (1 << j)
                            current_adj[j] |= (1 << i)
                        else:
                            current_adj[i] &= ~(1 << j)
                            current_adj[j] &= ~(1 << i)
                else:
                    # Revert
                    if old_state:
                        current_adj[i] |= (1 << j)
                        current_adj[j] |= (1 << i)
                    else:
                        current_adj[i] &= ~(1 << j)
                        current_adj[j] &= ~(1 << i)
            
            if not improved:
                # No improvement this round, try random restarts from current
                for _ in range(5):
                    if time.time() > deadline - 5:
                        break
                    adj2 = [a | 0 for a in current_adj]
                    for _ in range(random.randint(3, 5)):
                        i, j = random.randint(0, n-1), random.randint(0, n-1)
                        if i == j: continue
                        i, j = min(i,j), max(i,j)
                        if adj2[i] & (1 << j):
                            adj2[i] &= ~(1 << j)
                            adj2[j] &= ~(1 << i)
                        else:
                            adj2[i] |= (1 << j)
                            adj2[j] |= (1 << i)
                    if is_k4_free(adj2):
                        L2 = compute_L_fast(adj2)
                        if L2 < current_L:
                            current_L = L2
                            best_L = L2
                            best_adj = [a | 0 for a in adj2]
                            current_adj = adj2[:]
                            improved = True
                            print(f"Random restart improved L to {L2}", flush=True)
                if not improved:
                    break
    
    if best_adj is None:
        return make_edge_list(adj_turan)
    
    return make_edge_list(best_adj)
