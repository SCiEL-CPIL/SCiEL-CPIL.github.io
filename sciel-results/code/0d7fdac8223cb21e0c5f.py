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
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import eigsh

def construct():
    # 1. Generate all norm-2 integer vectors (squared norm 4)
    vectors = []
    
    # Type A: one +/- 2 (2*D vectors)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            vectors.append(v)
            
    # Type B: four +/- 1 (C(11,4) * 16 vectors)
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, val in zip(combo, signs):
                v[pos] = val
            vectors.append(v)
            
    N = len(vectors)
    
    # 2. Build conflict graph using dot products
    vec_arr = np.array(vectors, dtype=np.int8)
    dots = vec_arr @ vec_arr.T
    
    # Conflict if dot > 2 (distance < max_norm)
    # For unit vectors normalized to norm 2, min distance >= max norm means dot <= 2
    rows, cols = np.where(dots > 2)
    valid = rows < cols
    rows = rows[valid]
    cols = cols[valid]
    
    adj = [[] for _ in range(N)]
    for r, c in zip(rows, cols):
        adj[r].append(c)
        adj[c].append(r)
        
    degrees = np.array([len(a) for a in adj], dtype=np.int32)
    
    # 3. Identify bipartite structure
    # Type A vectors (one +/-2) have even "weight" in some sense
    # Let's classify: Type A = indices 0 to 2*D-1, Type B = rest
    type_A_count = 2 * D
    is_type_A = np.zeros(N, dtype=bool)
    is_type_A[:type_A_count] = True
    
    # Build sparse adjacency
    data = np.ones(len(rows), dtype=np.int32)
    row_sym = np.concatenate([rows, cols])
    col_sym = np.concatenate([cols, rows])
    data_sym = np.concatenate([data, data])
    A_sparse = csr_matrix((data_sym, (row_sym, col_sym)), shape=(N, N))
    
    # 4. Compute Fiedler vector (second smallest eigenvalue)
    D_sparse = csr_matrix(np.diag(degrees))
    L = D_sparse - A_sparse
    
    try:
        # Compute smallest eigenvalues
        eigvals, eigvecs_all = eigsh(L, k=6, which='SM', tol=1e-8)
        idx = np.argsort(eigvals)
        # Fiedler vector is index 1 (after the trivial 0 eigenvalue)
        fiedler = eigvecs_all[:, idx[1]]
    except Exception:
        # Fallback: use sign-based partition
        fiedler = np.random.randn(N)
    
    # 5. Spectral partition using Fiedler vector
    fiedler_centered = fiedler - np.mean(fiedler)
    
    # Partition by sign of Fiedler vector
    partition_1 = set(i for i in range(N) if fiedler_centered[i] > 0)
    partition_2 = set(i for i in range(N) if fiedler_centered[i] <= 0)
    
    # 6. Clean independent sets from each partition
    def clean_independent_set(vertex_set, adj, degrees):
        """Greedy independent set using degree-based ordering (lowest degree first for better packing)."""
        vertices = list(vertex_set)
        # Sort by degree ascending (low-degree vertices are easier to place)
        vertices.sort(key=lambda x: degrees[x])
        
        in_set = set()
        in_set_arr = np.zeros(N, dtype=bool)
        
        for v in vertices:
            # Check if v conflicts with any vertex in current set
            can_add = True
            for u in adj[v]:
                if in_set_arr[u]:
                    can_add = False
                    break
            if can_add:
                in_set.add(v)
                in_set_arr[v] = True
        
        return in_set
    
    set_1 = clean_independent_set(partition_1, adj, degrees)
    set_2 = clean_independent_set(partition_2, adj, degrees)
    
    # 7. Kempe-chain local search to improve independent set
    def kempe_local_search(S, adj, N, max_iterations=50):
        """
        Kempe-chain based local search for maximum independent set.
        Tries to swap subsets of vertices to increase set size.
        """
        S = set(S)
        in_S = np.zeros(N, dtype=bool)
        for v in S:
            in_S[v] = True
        
        for iteration in range(max_iterations):
            improved = False
            
            # Find all vertices not in S
            T = [v for v in range(N) if not in_S[v]]
            if not T:
                break
            
            # Find connected components in T (induced subgraph)
            visited = set()
            components = []
            
            for start in T:
                if start in visited:
                    continue
                
                comp = []
                stack = [start]
                visited.add(start)
                
                while stack:
                    u = stack.pop()
                    comp.append(u)
                    for v in adj[u]:
                        if not in_S[v] and v not in visited:
                            visited.add(v)
                            stack.append(v)
                
                components.append(comp)
            
            # Try to find a profitable Kempe chain
            best_gain = 0
            best_chain = None
            best_remove = None
            best_add = None
            
            for comp in components:
                if len(comp) > 200:  # Skip very large components for efficiency
                    continue
                
                # Find neighbors of comp that are in S
                R = set()
                for v in comp:
                    for u in adj[v]:
                        if in_S[u]:
                            R.add(u)
                
                if len(R) == 0:
                    # Can add entire component
                    # Find max independent set in comp
                    comp_sorted = sorted(comp, key=lambda x: degrees[x])
                    I = set()
                    for v in comp_sorted:
                        can_add = True
                        for u in adj[v]:
                            if u in I:
                                can_add = False
                                break
                        if can_add:
                            I.add(v)
                    
                    gain = len(I) - len(R)
                    if gain > best_gain:
                        best_gain = gain
                        best_chain = comp
                        best_remove = R
                        best_add = I
                else:
                    # Compute max independent set in comp
                    comp_sorted = sorted(comp, key=lambda x: degrees[x])
                    I = set()
                    for v in comp_sorted:
                        can_add = True
                        for u in adj[v]:
                            if u in I:
                                can_add = False
                                break
                        if can_add:
                            I.add(v)
                    
                    gain = len(I) - len(R)
                    if gain > best_gain:
                        best_gain = gain
                        best_chain = comp
                        best_remove = R
                        best_add = I
            
            # Also try simple 1-for-1 swaps
            if best_gain <= 0:
                # Try removing low-degree from S and adding high-degree from T
                S_list = sorted(S, key=lambda x: degrees[x])
                T_list = sorted(T, key=lambda x: degrees[x], reverse=True)
                
                for v_rem in S_list[:20]:
                    for v_add in T_list[:20]:
                        # Check if v_add can be added after removing v_rem
                        can_add = True
                        for u in adj[v_add]:
                            if in_S[u] and u != v_rem:
                                can_add = False
                                break
                        if can_add:
                            # Perform swap
                            in_S[v_rem] = False
                            S.discard(v_rem)
                            in_S[v_add] = True
                            S.add(v_add)
                            improved = True
                            break
                    if improved:
                        break
            
            # Apply best Kempe chain if profitable
            if not improved and best_gain > 0 and best_chain is not None:
                for v in best_remove:
                    in_S[v] = False
                    S.discard(v)
                for v in best_add:
                    in_S[v] = True
                    S.add(v)
                improved = True
            
            if not improved:
                break
        
        return S
    
    # Apply local search to both sets
    set_1 = kempe_local_search(set_1, adj, N)
    set_2 = kempe_local_search(set_2, adj, N)
    
    # 8. Merge the two sets
    def merge_sets(S1, S2, adj, N):
        """Merge two independent sets, keeping the larger and adding non-conflicting vertices from the other."""
        if len(S1) >= len(S2):
            S = set(S1)
            other = S2
        else:
            S = set(S2)
            other = S1
        
        in_S = np.zeros(N, dtype=bool)
        for v in S:
            in_S[v] = True
        
        # Try to add vertices from other
        for v in other:
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
        
        return S
    
    merged = merge_sets(set_1, set_2, adj, N)
    
    # 9. Final local search on merged set
    merged = kempe_local_search(merged, adj, N, max_iterations=30)
    
    # 10. Final repair: try to add any remaining vertices
    in_final = np.zeros(N, dtype=bool)
    for v in merged:
        in_final[v] = True
    
    for v in range(N):
        if in_final[v]:
            continue
        can_add = True
        for u in adj[v]:
            if in_final[u]:
                can_add = False
                break
        if can_add:
            in_final[v] = True
            merged.add(v)
    
    # 11. Try additional spectral partitions with different eigenvectors
    try:
        # Use next few eigenvectors for alternative partitions
        for eig_idx in range(2, min(6, len(idx))):
            if eig_idx >= len(idx):
                break
            evec = eigvecs_all[:, idx[eig_idx]]
            evec_centered = evec - np.mean(evec)
            
            p1 = set(i for i in range(N) if evec_centered[i] > 0)
            p2 = set(i for i in range(N) if evec_centered[i] <= 0)
            
            s1 = clean_independent_set(p1, adj, degrees)
            s2 = clean_independent_set(p2, adj, degrees)
            
            s1 = kempe_local_search(s1, adj, N, max_iterations=10)
            s2 = kempe_local_search(s2, adj, N, max_iterations=10)
            
            merged_alt = merge_sets(s1, s2, adj, N)
            merged_alt = kempe_local_search(merged_alt, adj, N, max_iterations=10)
            
            if len(merged_alt) > len(merged):
                merged = merged_alt
    except Exception:
        pass
    
    # 12. Convert to final output
    final_vectors = []
    for i in range(N):
        if i in merged:
            final_vectors.append(vectors[i])
            
    return np.array(final_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
