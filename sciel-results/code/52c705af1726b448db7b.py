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
    
    # Compute eigenvectors
    try:
        eigvals, eigvecs_all = eigsh(L, k=6, which='SM')
        idx = np.argsort(eigvals)
        # Skip the trivial eigenvector (index 0), use indices 1-5
        eigvecs = [eigvecs_all[:, i] for i in idx[1:6]]
    except Exception:
        eigvecs = [np.random.randn(N) for _ in range(5)]
        
    def greedy_independent_set(candidates, in_S_mask=None):
        """
        Greedy independent set from a list of candidate vertices.
        If in_S_mask is provided, it checks against that mask as well.
        """
        if in_S_mask is None:
            in_S_mask = [False] * N
        s = set()
        # Sort by degree descending (high degree vertices are "harder" to place, so place them first to block others? 
        # Or sort ascending? Usually for Max IS, picking high degree first is bad because they block many. 
        # Picking low degree first is often better for size, but high degree first can be better for density.
        # Let's stick to degree descending as in the original, but we will also try ascending in some restarts.
        sorted_cand = sorted(candidates, key=lambda x: degrees[x], reverse=True)
        for v in sorted_cand:
            if v in s:
                continue
            conflict = False
            for u in adj[v]:
                if in_S_mask[u] or u in s:
                    conflict = True
                    break
            if not conflict:
                s.add(v)
        return s

    def local_search_optimize(S):
        """
        Optimizes an independent set S using 1-for-1 swaps and greedy additions.
        """
        S = set(S)
        in_S = [False] * N
        for v in S:
            in_S[v] = True
        
        improved = True
        max_iter = 50
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # Phase 1: Greedy Add
            # Try to add any vertex not in S that has no neighbors in S
            added_any = False
            # We iterate in random order to break ties/symmetry
            order = np.random.permutation(N)
            for v in order:
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
                    added_any = True
            
            if added_any:
                improved = True
                continue # Restart loop as S changed significantly
            
            # Phase 2: 1-for-1 Swaps
            # Remove u from S, add v to S if v has no neighbors in S \ {u}
            # We want to find v that is "almost" in S (conflicts with only one or few vertices in S)
            # Actually, v must conflict with NO vertices in S \ {u}.
            # So v can only conflict with u among all vertices in S.
            
            # Strategy: 
            # 1. Identify vertices in T (not in S) that have exactly 1 neighbor in S.
            # 2. If we remove that neighbor, we can add v.
            # 3. We should only do this if it helps? 1-for-1 keeps size same. 
            #    Why do it? To enable future 1-for-2 or 1-for-3 moves, or just to shuffle to a better local optimum.
            #    Actually, pure 1-for-1 swaps don't increase size. They are useful for "annealing" to a better position.
            #    However, if we are stuck at a local max for size, 1-for-1 swaps alone won't increase size.
            #    We need 1-for-k swaps where k>1.
            
            # Let's try a simpler heuristic: 
            # For each v in T, count how many neighbors are in S.
            # If count == 1, let u be that neighbor.
            # If we remove u, v can be added.
            # This is a 1-for-1 swap. Size unchanged.
            # To increase size, we need to remove a set of vertices R from S and add a set A from T such that |A| > |R| and A is independent and has no neighbors in S\R.
            
            # Efficient 1-for-2 search:
            # Find pairs (u1, u2) in S that share a common neighbor v in T? No.
            # Find v in T such that its neighbors in S are exactly {u1, u2}.
            # If we remove u1 and u2, we can add v. Size change: -2 + 1 = -1. Bad.
            # We need |A| > |R|.
            # So we need v in T with neighbors in S being a set R of size k, and we add v and maybe other vertices?
            # Simplest gain: Remove 1 vertex u, add 2 vertices v1, v2.
            # Condition: v1, v2 not in S, v1-v2 not connected, and no neighbors in S \ {u}.
            # This means v1's neighbors in S are subset of {u}, v2's neighbors in S are subset of {u}.
            
            # Let's try to find such moves.
            # Pre-compute for each v in T: neighbors_in_S = [u for u in adj[v] if in_S[u]]
            # If len(neighbors_in_S) == 0: Add v (handled in Phase 1)
            # If len(neighbors_in_S) == 1: Let u = neighbors_in_S[0]. 
            #    If we remove u, v becomes valid.
            #    Can we find another v2 that also only conflicts with u?
            #    If yes, remove u, add v and v2. Gain +1.
            
            # Build map: neighbor_u -> list of v in T that ONLY conflict with u (and no other S vertices)
            # Note: v might conflict with other T vertices, that's fine as long as they are not in S.
            
            candidates_by_blocker = {}
            for v in range(N):
                if in_S[v]:
                    continue
                neigh_in_S = []
                for u in adj[v]:
                    if in_S[u]:
                        neigh_in_S.append(u)
                        if len(neigh_in_S) > 2: # Only interested in 1 or 2 blockers for small swaps
                            break
                if len(neigh_in_S) == 1:
                    u = neigh_in_S[0]
                    if u not in candidates_by_blocker:
                        candidates_by_blocker[u] = []
                    candidates_by_blocker[u].append(v)
                elif len(neigh_in_S) == 2:
                    # Store for potential 2-for-1? No, 2-for-1 reduces size.
                    # We want 1-for-2 or 2-for-3.
                    pass
            
            # Try 1-for-2 swaps
            best_swap_gain = 0
            best_swap_u = -1
            best_swap_vs = []
            
            for u, vs_list in candidates_by_blocker.items():
                if len(vs_list) >= 2:
                    # We can remove u and add any 2 from vs_list, provided they don't conflict with each other.
                    # Check if any pair in vs_list is independent.
                    found_pair = False
                    for i in range(len(vs_list)):
                        for j in range(i+1, len(vs_list)):
                            v1 = vs_list[i]
                            v2 = vs_list[j]
                            # Check if v1 and v2 are connected
                            if v2 not in adj[v1]: # O(1) check if adj was a set, but it's a list. 
                                # Need to check if v2 is in adj[v1]. 
                                # Since adj is list, this is O(degree). 
                                # To speed up, we can just assume they are independent if they are both "only blocked by u" and we check explicitly.
                                # Let's do explicit check.
                                is_connected = False
                                for w in adj[v1]:
                                    if w == v2:
                                        is_connected = True
                                        break
                                if not is_connected:
                                    found_pair = True
                                    break
                        if found_pair:
                            break
                    if found_pair:
                        # Gain is +1 (remove 1, add 2)
                        # We prefer swaps that remove low degree vertices? Or just take the first one?
                        # Let's take the first one found.
                        best_swap_gain = 1
                        best_swap_u = u
                        # Find the specific pair
                        for i in range(len(vs_list)):
                            for j in range(i+1, len(vs_list)):
                                v1 = vs_list[i]
                                v2 = vs_list[j]
                                is_connected = False
                                for w in adj[v1]:
                                    if w == v2:
                                        is_connected = True
                                        break
                                if not is_connected:
                                    best_swap_vs = [v1, v2]
                                    break
                            if best_swap_vs:
                                break
                        break
            
            if best_swap_gain > 0:
                # Execute swap
                in_S[best_swap_u] = False
                S.discard(best_swap_u)
                for v in best_swap_vs:
                    in_S[v] = True
                    S.add(v)
                improved = True
                continue
            
            # If no 1-for-2 found, try 1-for-1 swaps to shuffle (annealing)
            # This helps escape local optima where no 1-for-2 is available but a sequence of 1-for-1 leads to one.
            # We will perform a few random 1-for-1 swaps.
            if random.random() < 0.5: # Do this sometimes
                # Find a random v in T with exactly 1 blocker u
                # And a random u in S that blocks some v
                # Just pick a random u from candidates_by_blocker keys
                if candidates_by_blocker:
                    u = random.choice(list(candidates_by_blocker.keys()))
                    vs_list = candidates_by_blocker[u]
                    if vs_list:
                        v = random.choice(vs_list)
                        # Perform 1-for-1 swap
                        in_S[u] = False
                        S.discard(u)
                        in_S[v] = True
                        S.add(v)
                        improved = True
                        continue
            
            # If no improvements, break
            break
        
        return S

    # 4. Generate multiple initial sets and optimize
    best_set = set()
    best_size = 0
    
    # Strategy 1: Spectral Partitions
    # Use Fiedler vector (index 0 in eigvecs list which is 2nd smallest eigenvector overall)
    fiedler = eigvecs[0]
    
    # Create 2-way partitions from Fiedler
    mean_f = np.mean(fiedler)
    P1 = [i for i in range(N) if fiedler[i] > mean_f]
    P2 = [i for i in range(N) if fiedler[i] <= mean_f]
    
    initial_sets = []
    
    # Greedy on P1
    S1 = greedy_independent_set(P1)
    initial_sets.append(S1)
    
    # Greedy on P2
    S2 = greedy_independent_set(P2)
    initial_sets.append(S2)
    
    # Greedy on Full Set (ordered by Fiedler)
    order = np.argsort(fiedler)
    S3 = greedy_independent_set(list(order))
    initial_sets.append(S3)
    
    # Strategy 2: Randomized Spectral
    for i in range(5):
        noise = np.random.randn(N) * 0.1
        perturbed = fiedler + noise
        mean_p = np.mean(perturbed)
        P1_r = [j for j in range(N) if perturbed[j] > mean_p]
        S_r = greedy_independent_set(P1_r)
        initial_sets.append(S_r)
        
    # Strategy 3: Pure Random Greedy
    for i in range(5):
        random_order = np.random.permutation(N)
        S_rand = greedy_independent_set(list(random_order))
        initial_sets.append(S_rand)
        
    # Optimize all initial sets
    for S_init in initial_sets:
        S_opt = local_search_optimize(S_init)
        if len(S_opt) > best_size:
            best_size = len(S_opt)
            best_set = S_opt
            
    # Final repair: try to add any remaining vertices
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
