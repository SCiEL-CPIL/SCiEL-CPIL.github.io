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
    start_time = time.time()
    time_budget = 115.0  # Leave 5s margin
    
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
    
    # Conflict if dot > 2 (i.e., distance squared < 4)
    rows, cols = np.where(dots > 2)
    valid = rows < cols
    rows = rows[valid]
    cols = cols[valid]
    
    adj = [[] for _ in range(N)]
    for r, c in zip(rows, cols):
        adj[r].append(c)
        adj[c].append(r)
        
    degrees = np.array([len(a) for a in adj], dtype=np.int32)
    
    # 3. Spectral Partitioning - compute Fiedler vector and additional eigenvectors
    data = np.ones(len(rows), dtype=np.int32)
    row_sym = np.concatenate([rows, cols])
    col_sym = np.concatenate([cols, rows])
    data_sym = np.concatenate([data, data])
    A_sparse = csr_matrix((data_sym, (row_sym, col_sym)), shape=(N, N))
    
    D_sparse = csr_matrix(np.diag(degrees))
    L = D_sparse - A_sparse
    
    eigvecs = []
    try:
        eigvals, eigvecs_all = eigsh(L, k=8, which='SM')
        idx = np.argsort(eigvals)
        # Skip the trivial eigenvector (index 0), use indices 1-7
        for i in idx[1:8]:
            eigvecs.append(eigvecs_all[:, i])
    except Exception:
        eigvecs = [np.random.randn(N) for _ in range(7)]
    
    def clean_independent_set(partition):
        """Greedy independent set from a candidate partition, using degree-based ordering."""
        if not partition:
            return set()
        sorted_part = sorted(partition, key=lambda x: degrees[x], reverse=True)
        s = set()
        for v in sorted_part:
            conflict = False
            for u in adj[v]:
                if u in s:
                    conflict = True
                    break
            if not conflict:
                s.add(v)
        return s
    
    def kempe_chain_local_search(S, max_iter=300):
        """
        Kempe-chain based local search to maximize independent set size.
        Tries: (1) add any vertex that fits, (2) component swaps, (3) pair swaps.
        """
        S = set(S)
        in_S = [False] * N
        for v in S:
            in_S[v] = True
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # Collect all vertices not in S
            T = [v for v in range(N) if not in_S[v]]
            if not T:
                break
            
            # Phase 1: Try individual vertex additions (fast wins)
            for v in T:
                if in_S[v]:
                    continue
                can_add = True
                for u in adj[v]:
                    if in_S[u]:
                        can_add = False
                        break
                if can_add:
                    in_S[v] = True
                    S.add(v)
                    improved = True
            
            if improved:
                continue
            
            # Phase 2: Find connected components in T
            visited_comp = [False] * N
            components = []
            
            for start_v in T:
                if visited_comp[start_v]:
                    continue
                
                comp = []
                queue = [start_v]
                visited_comp[start_v] = True
                
                while queue:
                    u = queue.pop()
                    comp.append(u)
                    for v in adj[u]:
                        if not in_S[v] and not visited_comp[v]:
                            visited_comp[v] = True
                            queue.append(v)
                
                components.append(comp)
            
            # Phase 3: Try component swaps
            best_gain = 0
            best_I = None
            best_R = None
            
            for comp in components:
                if len(comp) < 2:
                    continue
                # Compute neighbors in S
                R = set()
                for v in comp:
                    for u in adj[v]:
                        if in_S[u]:
                            R.add(u)
                
                # Greedy IS within comp
                sorted_comp = sorted(comp, key=lambda x: degrees[x], reverse=True)
                I = set()
                for v in sorted_comp:
                    conflict = False
                    for u in adj[v]:
                        if u in I:
                            conflict = True
                            break
                    if not conflict:
                        I.add(v)
                
                gain = len(I) - len(R)
                if gain > best_gain:
                    best_gain = gain
                    best_I = I
                    best_R = R
            
            if best_I is not None and best_gain > 0:
                for v in best_R:
                    in_S[v] = False
                    S.discard(v)
                for v in best_I:
                    in_S[v] = True
                    S.add(v)
                improved = True
                continue
            
            # Phase 4: Try pair swaps (remove one from S, add one from T)
            S_list = list(S)
            S_sorted = sorted(S_list, key=lambda x: degrees[x])
            T_sorted = sorted(T, key=lambda x: degrees[x], reverse=True)
            
            found_swap = False
            for v_remove in S_sorted[:20]:
                for v_add in T_sorted[:20]:
                    can_add = True
                    for u in adj[v_add]:
                        if in_S[u] and u != v_remove:
                            can_add = False
                            break
                    if can_add:
                        in_S[v_remove] = False
                        S.discard(v_remove)
                        in_S[v_add] = True
                        S.add(v_add)
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    break
            
        return S
    
    def spectral_partition_2way(evec):
        """Partition into 2 sets based on sign of centered eigenvector."""
        evec = evec - np.mean(evec)
        if np.std(evec) < 1e-10:
            return [[i for i in range(N) if i % 2 == 0], [i for i in range(N) if i % 2 == 1]]
        P1 = [i for i in range(N) if evec[i] > 0]
        P2 = [i for i in range(N) if evec[i] <= 0]
        return [P1, P2]
    
    def spectral_partition_kway(evec, k=4):
        """Partition into k sets using quantiles."""
        evec = evec - np.mean(evec)
        if np.std(evec) < 1e-10:
            parts = [[] for _ in range(k)]
            for i in range(N):
                parts[i % k].append(i)
            return parts
        thresholds = np.percentile(evec, np.linspace(0, 100, k + 1))
        parts = [[] for _ in range(k)]
        for i in range(N):
            for k_idx in range(k):
                if thresholds[k_idx] <= evec[i] < thresholds[k_idx + 1]:
                    parts[k_idx].append(i)
                    break
            else:
                parts[k - 1].append(i)
        return parts
    
    # 4. Generate multiple partition strategies
    all_partitions = []
    
    # Strategy 1: Fiedler vector 2-way
    fiedler = eigvecs[0]
    all_partitions.append(spectral_partition_2way(fiedler))
    
    # Strategy 2: Each eigenvector 2-way
    for evec in eigvecs:
        all_partitions.append(spectral_partition_2way(evec))
    
    # Strategy 3: 4-way partitions from eigenvectors
    for evec in eigvecs[:4]:
        parts4 = spectral_partition_kway(evec, 4)
        # Merge pairs to get 2-way partitions
        all_partitions.append([parts4[0] + parts4[1], parts4[2] + parts4[3]])
        all_partitions.append([parts4[0] + parts4[2], parts4[1] + parts4[3]])
        all_partitions.append([parts4[0] + parts4[3], parts4[1] + parts4[2]])
    
    # Strategy 4: 2D embedding quadrant splits
    for i in range(min(len(eigvecs), 4)):
        for j in range(i + 1, min(len(eigvecs), 4)):
            X = np.column_stack([eigvecs[i], eigvecs[j]])
            X = X - X.mean(axis=0)
            medians = np.median(X, axis=0)
            parts = [[], [], [], []]
            for idx in range(N):
                if X[idx, 0] >= medians[0] and X[idx, 1] >= medians[1]:
                    parts[0].append(idx)
                elif X[idx, 0] < medians[0] and X[idx, 1] >= medians[1]:
                    parts[1].append(idx)
                elif X[idx, 0] >= medians[0] and X[idx, 1] < medians[1]:
                    parts[2].append(idx)
                else:
                    parts[3].append(idx)
            all_partitions.append([parts[0] + parts[1], parts[2] + parts[3]])
            all_partitions.append([parts[0] + parts[2], parts[1] + parts[3]])
            all_partitions.append([parts[0] + parts[3], parts[1] + parts[2]])
    
    # Strategy 5: Random perturbations of Fiedler
    for _ in range(5):
        noise = np.random.randn(N) * 0.03
        perturbed = fiedler + noise
        all_partitions.append(spectral_partition_2way(perturbed))
    
    # 5. Process partitions: clean, then merge via Kempe search
    best_set = set()
    best_size = 0
    
    for parts in all_partitions:
        if time.time() - start_time > time_budget:
            break
        if len(parts) < 2:
            continue
        
        cleaned_parts = []
        for part in parts:
            cleaned_parts.append(clean_independent_set(part))
        
        if all(len(p) == 0 for p in cleaned_parts):
            continue
        
        # Start with largest cleaned part
        current = max(cleaned_parts, key=len)
        
        # Merge others one by one
        for other in cleaned_parts:
            if other is current:
                continue
            # Try to add vertices from other into current
            in_cur = [False] * N
            for v in current:
                in_cur[v] = True
            for v in other:
                if in_cur[v]:
                    continue
                can_add = True
                for u in adj[v]:
                    if in_cur[u]:
                        can_add = False
                        break
                if can_add:
                    in_cur[v] = True
                    current.add(v)
        
        # Run Kempe chain local search
        current = kempe_chain_local_search(current)
        
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
    
    # 6. Additional: greedy with spectral ordering
    order = np.argsort(fiedler)
    greedy_set = clean_independent_set(list(order))
    greedy_set = kempe_chain_local_search(greedy_set)
    
    if len(greedy_set) > best_size:
        best_set = greedy_set
        best_size = len(greedy_set)
    
    # 7. Final repair: try to add any remaining vertices
    in_best = [False] * N
    for v in best_set:
        in_best[v] = True
    
    for v in range(N):
        if in_best[v]:
            continue
        can_add = True
        for u in adj[v]:
            if in_best[u]:
                can_add = False
                break
        if can_add:
            in_best[v] = True
            best_set.add(v)
    
    final_vectors = []
    for i in range(N):
        if i in best_set:
            final_vectors.append(vectors[i])
            
    return np.array(final_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
