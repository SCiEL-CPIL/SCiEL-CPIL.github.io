import numpy as np
import itertools
import random
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def construct():
    # 1. Generate all norm-2 integer vectors (squared norm 4)
    vectors = []
    
    # Type A: one +/- 2
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            vectors.append(v)
            
    # Type B: four +/- 1
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, val in zip(combo, signs):
                v[pos] = val
            vectors.append(v)
            
    N = len(vectors)
    
    # 2. Build conflict graph
    vec_arr = np.array(vectors, dtype=np.int8)
    dots = vec_arr @ vec_arr.T
    
    # Conflict if dot > 2
    rows, cols = np.where(dots > 2)
    valid = rows < cols
    rows = rows[valid]
    cols = cols[valid]
    
    adj = [[] for _ in range(N)]
    for r, c in zip(rows, cols):
        adj[r].append(c)
        adj[c].append(r)
        
    degrees = np.array([len(a) for a in adj], dtype=np.int32)
    
    # 3. Spectral Partitioning
    data = np.ones(len(rows), dtype=np.int32)
    row_sym = np.concatenate([rows, cols])
    col_sym = np.concatenate([cols, rows])
    data_sym = np.concatenate([data, data])
    A_sparse = csr_matrix((data_sym, (row_sym, col_sym)), shape=(N, N))
    
    D_sparse = csr_matrix(np.diag(degrees))
    L = D_sparse - A_sparse
    
    try:
        eigvals, eigvecs = eigsh(L, k=2, which='SM')
        idx = np.argsort(eigvals)
        fiedler_vec = eigvecs[:, idx[1]]
    except Exception:
        fiedler_vec = np.random.randn(N)
        
    def clean_set(partition):
        sorted_part = sorted(partition, key=lambda x: degrees[x], reverse=True)
        s = set()
        in_s = [False] * N
        for v in sorted_part:
            conflict = False
            for u in adj[v]:
                if in_s[u]:
                    conflict = True
                    break
            if not conflict:
                s.add(v)
                in_s[v] = True
        return s
        
    def local_search(S):
        S = set(S)
        in_S = [False] * N
        for v in S:
            in_S[v] = True
            
        improved = True
        max_iter = 100
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # Find connected components in V \ S
            # We use BFS/DFS
            unvisited = [v for v in range(N) if not in_S[v]]
            visited_comp = [False] * N
            
            while unvisited:
                start_v = unvisited.pop(0)
                if visited_comp[start_v]:
                    continue
                
                comp = []
                queue = [start_v]
                visited_comp[start_v] = True
                
                while queue:
                    u = queue.pop(0)
                    comp.append(u)
                    for v in adj[u]:
                        if not in_S[v] and not visited_comp[v]:
                            visited_comp[v] = True
                            queue.append(v)
                            
                # Calculate neighbors in S
                R = set()
                for v in comp:
                    for u in adj[v]:
                        if in_S[u]:
                            R.add(u)
                            
                # Calculate Max Independent Set in comp (Greedy)
                # Sort by degree descending within comp
                sorted_comp = sorted(comp, key=lambda x: degrees[x], reverse=True)
                I = set()
                in_I = [False] * N
                for v in sorted_comp:
                    conflict = False
                    for u in adj[v]:
                        if in_I[u]:
                            conflict = True
                            break
                    if not conflict:
                        I.add(v)
                        in_I[v] = True
                        
                # Kempe flip condition: gain = |I| - |R|
                if len(I) > len(R):
                    # Remove R from S
                    for v in R:
                        in_S[v] = False
                        S.remove(v)
                    # Add I to S
                    for v in I:
                        in_S[v] = True
                        S.add(v)
                    improved = True
                    # Reset visited_comp for next iteration since graph structure of V\S changed
                    # Actually, we just continue the loop, but the 'unvisited' list is stale.
                    # It's safer to break and restart the component search loop
                    break
                    
            # If we broke due to improvement, the outer while loop will restart
            # If we finished all components without improvement, loop ends
            
        return S
        
    best_set = set()
    best_size = 0
    
    # Try both partitions
    P1 = [i for i in range(N) if fiedler_vec[i] > 0]
    P2 = [i for i in range(N) if fiedler_vec[i] <= 0]
    
    S1_clean = clean_set(P1)
    S2_clean = clean_set(P2)
    
    S1_final = local_search(S1_clean)
    S2_final = local_search(S2_clean)
    
    if len(S1_final) >= len(S2_final):
        best_set = S1_final
    else:
        best_set = S2_final
        
    # Optional: Try a random restart if we have time, but spectral is usually strong.
    # Let's add one random restart to be safe against bad spectral splits
    # We can mix the two sets or just try a random subset
    # Given the time limit, let's stick to the best of the two spectral partitions.
    # However, we can also try a "merge" strategy if the sets are disjoint? No, they are from same graph.
    
    final_vectors = []
    for i in range(N):
        if i in best_set:
            final_vectors.append(vectors[i])
            
    return np.array(final_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
