import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Generate all weight-4 vectors
    # Indices 0..10
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    
    vectors = []
    vector_supports = [] # To speed up conflict check
    
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for i, sig in zip(s, signs):
                v[i] = sig
            vectors.append(v)
            vector_supports.append(s)
            
    vectors_np = np.array(vectors, dtype=np.int64)
    n = len(vectors_np)
    
    # Build conflict graph
    # Conflict if supports share 3 indices and signs match on those indices
    # We can precompute a map from support to list of indices
    # Or just check pairwise
    
    # For speed, we can group by support
    # But n=5280 is small enough for O(n^2) with early exit
    
    adj = [set() for _ in range(n)]
    
    # Optimization: 
    # Two vectors conflict only if their supports share exactly 3 indices.
    # Let's map each support to its vectors.
    support_to_vecs = {}
    for i, s in enumerate(vector_supports):
        if s not in support_to_vecs:
            support_to_vecs[s] = []
        support_to_vecs[s].append(i)
        
    # For each pair of supports that share 3 indices, check conflicts
    # This is more efficient than all pairs of vectors
    
    support_list = list(support_to_vecs.keys())
    
    # Precompute intersection sizes?
    # There are 330 supports. 330^2 = 108900 pairs.
    # For each pair, if intersection size is 3, check the 16x16=256 vector pairs.
    # 108900 * (check cost) is fine.
    
    for i in range(len(support_list)):
        s1 = support_list[i]
        vecs1 = support_to_vecs[s1]
        for j in range(i+1, len(support_list)):
            s2 = support_list[j]
            # Check intersection
            inter = set(s1).intersection(s2)
            if len(inter) != 3:
                continue
            
            vecs2 = support_to_vecs[s2]
            
            # For each v1 in vecs1, v2 in vecs2
            # Check if signs match on inter
            for idx1 in vecs1:
                v1 = vectors_np[idx1]
                for idx2 in vecs2:
                    v2 = vectors_np[idx2]
                    # Check signs on inter
                    conflict = True
                    for k in inter:
                        if v1[k] != v2[k]:
                            conflict = False
                            break
                    if conflict:
                        adj[idx1].add(idx2)
                        adj[idx2].add(idx1)
                        
    # Greedy Independent Set
    # Sort nodes by degree descending
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    used = [False] * n
    independent_set = []
    
    for u in order:
        if not used[u]:
            independent_set.append(u)
            # Mark u and neighbors as used
            used[u] = True
            for v in adj[u]:
                used[v] = True
                
    result = vectors_np[independent_set]
    
    # Verify
    # Check time
    if time.time() > deadline:
         # Return best found so far (which is the greedy result)
         pass
         
    return result
