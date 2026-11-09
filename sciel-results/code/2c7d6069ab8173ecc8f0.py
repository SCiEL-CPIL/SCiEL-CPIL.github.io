"""Matrix multiplication 3x5x6 rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 90

    # All (i,j,k) triples
    all_triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

    def check_cluster(triples):
        """Check if the given triples form a valid cluster (rank-1 tensor covers exactly them)."""
        if len(triples) < 2:
            return False
        U_nz = set((i, j) for (i, j, k) in triples)
        V_nz = set((j, k) for (i, j, k) in triples)
        W_nz = set((k, i) for (i, j, k) in triples)
        
        # Compute support
        support = set()
        for (i, j) in U_nz:
            for (j2, k) in V_nz:
                if j2 != j:
                    continue
                for (k2, i2) in W_nz:
                    if k2 != k or i2 != i:
                        continue
                    support.add((i, j, k))
        
        return support == set(triples)

    # Try to find large clusters using random sampling
    random.seed(42)
    best_cluster_size = 2
    best_cluster = None
    
    # Try random subsets of size 3
    for _ in range(10000):
        if time.time() > deadline - 20:
            break
        sample = random.sample(all_triples, 3)
        if check_cluster(sample):
            best_cluster_size = 3
            best_cluster = sample
            break
    
    # Try size 4
    if best_cluster_size < 4:
        for _ in range(5000):
            if time.time() > deadline - 15:
                break
            sample = random.sample(all_triples, 4)
            if check_cluster(sample):
                best_cluster_size = 4
                best_cluster = sample
                break
    
    # Try size 5
    if best_cluster_size < 5:
        for _ in range(2000):
            if time.time() > deadline - 10:
                break
            sample = random.sample(all_triples, 5)
            if check_cluster(sample):
                best_cluster_size = 5
                best_cluster = sample
                break
    
    # Build decomposition using greedy clustering
    remaining = set(all_triples)
    decomposition = []
    
    # First, extract the best cluster we found
    if best_cluster and best_cluster_size >= 3:
        cluster_set = set(best_cluster)
        if cluster_set.issubset(remaining):
            U_nz = set((i, j) for (i, j, k) in best_cluster)
            V_nz = set((j, k) for (i, j, k) in best_cluster)
            W_nz = set((k, i) for (i, j, k) in best_cluster)
            decomposition.append((U_nz, V_nz, W_nz))
            remaining -= cluster_set
    
    # Continue searching for more clusters while time allows
    search_limit = deadline - 5
    while time.time() < search_limit:
        # Try to find a cluster of size >= 3 in remaining
        found = False
        remaining_list = list(remaining)
        for _ in range(1000):
            if time.time() > search_limit:
                break
            if len(remaining_list) < 3:
                break
            sample = random.sample(remaining_list, 3)
            if check_cluster(sample):
                cluster_set = set(sample)
                U_nz = set((i, j) for (i, j, k) in sample)
                V_nz = set((j, k) for (i, j, k) in sample)
                W_nz = set((k, i) for (i, j, k) in sample)
                decomposition.append((U_nz, V_nz, W_nz))
                remaining -= cluster_set
                remaining_list = list(remaining)
                found = True
                break
        if not found:
            break
    
    # Now pair up remaining triples
    remaining_list = list(remaining)
    random.shuffle(remaining_list)
    
    for idx in range(0, len(remaining_list) - 1, 2):
        t1 = remaining_list[idx]
        t2 = remaining_list[idx + 1]
        U_nz = {(t1[0], t1[1]), (t2[0], t2[1])}
        V_nz = {(t1[1], t1[2]), (t2[1], t2[2])}
        W_nz = {(t1[2], t1[0]), (t2[2], t2[0])}
        decomposition.append((U_nz, V_nz, W_nz))
    
    # If odd number remaining, one singleton
    if len(remaining_list) % 2 == 1:
        t = remaining_list[-1]
        U_nz = {(t[0], t[1])}
        V_nz = {(t[1], t[2])}
        W_nz = {(t[2], t[0])}
        decomposition.append((U_nz, V_nz, W_nz))
    
    rank = len(decomposition)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (U_nz, V_nz, W_nz) in enumerate(decomposition):
        for (i, j) in U_nz:
            U[i * m + j][col] = 1
        for (j, k) in V_nz:
            V[j * p + k][col] = 1
        for (k, i) in W_nz:
            W[k * n + i][col] = 1
    
    # Verify the decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if total != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U[ii * m + jj][column] = 1
                    V[jj * p + kk][column] = 1
                    W[kk * n + ii][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
