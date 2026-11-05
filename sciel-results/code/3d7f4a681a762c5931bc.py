import numpy as np

def construct():
    n = 256
    nums = list(range(1, 257))
    
    # Build conflict matrix
    # C[i, j] = 1 if i+1 and j+1 conflict
    C = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            if (a * b) % (a + b) == 0:
                C[i, j] = 1
                C[j, i] = 1

    # Build non-conflict matrix for spectral ordering
    # N[i, j] = 1 if i != j and no conflict
    N = np.ones((n, n), dtype=np.int8) - np.eye(n, dtype=np.int8) - C

    # Compute principal eigenvector of N
    # Using power iteration
    v = np.ones(n, dtype=np.float64)
    v /= np.linalg.norm(v)
    for _ in range(100):
        new_v = N.astype(np.float64) @ v
        norm = np.linalg.norm(new_v)
        if norm < 1e-12:
            break
        new_v /= norm
        if np.allclose(new_v, v, atol=1e-10):
            break
        v = new_v

    # Sort indices by eigenvector value descending
    order = np.argsort(-v)
    
    # Greedy construction
    selected = set()
    for idx in order:
        i = int(idx)
        # Check if i conflicts with any selected
        can_add = True
        # Use bitset or loop? Loop is fine for N=256
        for j in selected:
            if C[i, j] == 1:
                can_add = False
                break
        if can_add:
            selected.add(i)

    # Local Search: Try to increase size by swapping
    # Strategy: Find u in unselected that conflicts with exactly 1 selected node s.
    # Check if there exists v in unselected such that:
    # 1. v is compatible with u
    # 2. v is compatible with selected \ {s}
    # If so, remove s, add u and v.
    
    improved = True
    while improved:
        improved = False
        unselected = [i for i in range(n) if i not in selected]
        
        for u in unselected:
            # Find conflicts of u in selected
            conflicts_in_S = [s for s in selected if C[u, s] == 1]
            
            if len(conflicts_in_S) == 1:
                s = conflicts_in_S[0]
                # Try to find a partner v
                found_partner = False
                for v in unselected:
                    if v == u:
                        continue
                    # v must be compatible with u
                    if C[u, v] == 1:
                        continue
                    # v must be compatible with selected \ {s}
                    # i.e., v should not conflict with any s' in selected, s' != s
                    compatible_with_rest = True
                    for s_prime in selected:
                        if s_prime == s:
                            continue
                        if C[v, s_prime] == 1:
                            compatible_with_rest = False
                            break
                    if compatible_with_rest:
                        # Perform swap: remove s, add u and v
                        selected.remove(s)
                        selected.add(u)
                        selected.add(v)
                        improved = True
                        found_partner = True
                        break
                if found_partner:
                    break # Restart the while loop to update unselected

    result = sorted([i + 1 for i in selected])
    return result
