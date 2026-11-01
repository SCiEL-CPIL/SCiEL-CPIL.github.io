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
    
    # 3. Compute Fiedler vector (2nd smallest eigenvector of Laplacian)
    data = np.ones(len(rows), dtype=np.int32)
    row_sym = np.concatenate([rows, cols])
    col_sym = np.concatenate([cols, rows])
    data_sym = np.concatenate([data, data])
    A_sparse = csr_matrix((data_sym, (row_sym, col_sym)), shape=(N, N))
    
    D_sparse = csr_matrix(np.diag(degrees))
    L = D_sparse - A_sparse
    
    fiedler_vec = None
    try:
        eigvals, eigvecs_all = eigsh(L, k=3, which='SM')
        idx = np.argsort(eigvals)
        # The smallest eigenvalue should be 0 (trivial), next is Fiedler
        fiedler_vec = eigvecs_all[:, idx[1]]
    except Exception:
        fiedler_vec = np.random.randn(N)
    
    # 4. Partition using Fiedler vector
    fiedler_vec = fiedler_vec - np.mean(fiedler_vec)
    P1 = [i for i in range(N) if fiedler_vec[i] > 0]
    P2 = [i for i in range(N) if fiedler_vec[i] <= 0]
    
    # 5. Clean partitions into independent sets
    def clean_independent_set(partition):
        """Greedy independent set from a candidate partition."""
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
    
    IS1 = clean_independent_set(P1)
    IS2 = clean_independent_set(P2)
    
    # 6. Kempe-chain local search
    def kempe_chain_local_search(S, max_iter=300):
        """
        Kempe-chain based local search to maximize independent set size.
        Exploits bipartite-like structure by trying chain flips.
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
            
            # Try individual vertex additions first
            for v in T:
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
            
            # Find connected components in T
            visited_comp = set()
            components = []
            
            for start_v in T:
                if start_v in visited_comp:
                    continue
                
                comp = []
                queue = [start_v]
                visited_comp.add(start_v)
                
                while queue:
                    u = queue.pop(0)
                    comp.append(u)
                    for v in adj[u]:
                        if not in_S[v] and v not in visited_comp:
                            visited_comp.add(v)
                            queue.append(v)
                
                components.append(comp)
            
            # For each component, compute gain of swapping
            best_gain = 0
            best_comp = None
            best_I = None
            best_R = None
            
            for comp in components:
                # Compute neighbors in S
                R = set()
                for v in comp:
                    for u in adj[v]:
                        if in_S[u]:
                            R.add(u)
                
                # Greedy max independent set within comp
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
                    best_comp = comp
                    best_I = I
                    best_R = R
            
            # Try single-vertex swaps if no positive gain component
            if best_gain <= 0:
                S_sorted = sorted(S, key=lambda x: degrees[x])
                T_sorted = sorted(T, key=lambda x: degrees[x], reverse=True)
                
                for v_remove in S_sorted[:15]:
                    for v_add in T_sorted[:15]:
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
                            break
                    if improved:
                        break
            
            if not improved and best_comp is not None and best_gain > 0:
                for v in best_R:
                    in_S[v] = False
                    S.discard(v)
                for v in best_I:
                    in_S[v] = True
                    S.add(v)
                improved = True
        
        return S
    
    # 7. Refine both partitions
    IS1 = kempe_chain_local_search(IS1)
    IS2 = kempe_chain_local_search(IS2)
    
    # 8. Merge the two independent sets
    def merge_sets(S1, S2):
        """Merge two independent sets, preferring vertices from larger set."""
        if len(S1) < len(S2):
            S1, S2 = S2, S1
        
        S = set(S1)
        in_S = [False] * N
        for v in S:
            in_S[v] = True
        
        # Try to add vertices from S2 that don't conflict
        for v in S2:
            if in_S[v]:
                continue
            conflict = False
            for u in adj[v]:
                if in_S[u]:
                    conflict = True
                    break
            if not conflict:
                in_S[v] = True
                S.add(v)
        
        return S
    
    best_set = merge_sets(IS1, IS2)
    
    # 9. Final Kempe-chain refinement on merged set
    best_set = kempe_chain_local_search(best_set)
    
    # 10. Try multiple random perturbations of Fiedler partition
    for restart in range(5):
        noise = np.random.randn(N) * 0.1
        perturbed = fiedler_vec + noise
        P1_r = [i for i in range(N) if perturbed[i] > 0]
        P2_r = [i for i in range(N) if perturbed[i] <= 0]
        
        IS1_r = clean_independent_set(P1_r)
        IS2_r = clean_independent_set(P2_r)
        
        IS1_r = kempe_chain_local_search(IS1_r, max_iter=100)
        IS2_r = kempe_chain_local_search(IS2_r, max_iter=100)
        
        merged_r = merge_sets(IS1_r, IS2_r)
        merged_r = kempe_chain_local_search(merged_r, max_iter=100)
        
        if len(merged_r) > len(best_set):
            best_set = merged_r
    
    # 11. Final repair: add any remaining non-conflicting vertices
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
