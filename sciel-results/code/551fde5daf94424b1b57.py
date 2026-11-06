import time
import random
import numpy as np

def construct():
    deadline = time.time() + 145
    
    # We know n=26 is possible. We try to find it.
    # If we fail, we fall back to smaller n.
    targets = [26, 25, 24, 23]
    
    best_n = 0
    best_colors = []
    
    for n in targets:
        if time.time() > deadline - 5:
            break
            
        # Allocate more time for larger n
        if n == 26:
            t_limit = 100
        elif n == 25:
            t_limit = 30
        else:
            t_limit = 15
            
        actual_limit = min(deadline - time.time() - 5, t_limit)
        if actual_limit < 2:
            continue
            
        success, colors = _find_coloring(n, time.time() + actual_limit)
        
        if success:
            best_n = n
            best_colors = colors
            break
            
    # Fallback if 26 failed, ensure we return something valid
    if best_n == 0:
        # Try a smaller n quickly
        for n in [22, 21, 20]:
            success, colors = _find_coloring(n, time.time() + 5)
            if success:
                best_n = n
                best_colors = colors
                break
        if best_n == 0:
            # Trivial n=1
            best_n = 1
            best_colors = []

    return {'n': best_n, 'colors': best_colors}

def _find_coloring(n, deadline):
    if n < 3:
        return True, []
        
    m = n * (n - 1) // 2
    
    # Precompute edge endpoints
    edge_u = np.zeros(m, dtype=np.int32)
    edge_v = np.zeros(m, dtype=np.int32)
    pair_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_u[idx] = i
            edge_v[idx] = j
            pair_to_idx[(i, j)] = idx
            pair_to_idx[(j, i)] = idx
            idx += 1
            
    # Adjacency matrices for each color: A[c] is n x n
    A = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
    # Squared matrices: Sq[c] = A[c] @ A[c]
    Sq = [np.zeros((n, n), dtype=np.int32) for _ in range(5)]
    
    # Initial random coloring
    colors = np.random.randint(0, 5, m, dtype=np.int8)
    
    # Build initial A and Sq
    for c in range(5):
        mask = (colors == c)
        for k in np.where(mask)[0]:
            u = edge_u[k]
            v = edge_v[k]
            A[c][u, v] = 1
            A[c][v, u] = 1
        Sq[c] = A[c] @ A[c]
        
    def calc_violations():
        total = 0
        for c in range(5):
            # We only care about i < j
            # Sq[c][i, j] is number of common neighbors
            # Violation if Sq[c][i, j] >= 2
            # We can use np.triu_indices
            iu, ju = np.triu_indices(n, k=1)
            vals = Sq[c][iu, ju]
            total += np.sum(np.maximum(0, vals - 1))
        return total
        
    current_violations = calc_violations()
    if current_violations == 0:
        return True, colors.tolist()
        
    best_violations = current_violations
    best_colors = colors.copy()
    
    # Simulated Annealing
    T = 2.0
    min_T = 0.01
    steps = 0
    
    while time.time() < deadline - 1:
        k = random.randint(0, m - 1)
        u = edge_u[k]
        v = edge_v[k]
        old_c = colors[k]
        new_c = random.randint(0, 4)
        
        if new_c == old_c:
            continue
            
        # Compute delta in violations
        delta = 0
        
        # Update for old_c: remove edge (u,v)
        # Sq_new = Sq_old - A E_uv - A E_vu - E_uv A - E_vu A + E_uu + E_vv
        # We need to check changes in objective for pairs (x,y) affected.
        # Affected pairs are those where x or y is u or v.
        # Also, the term (u,u) and (v,v) change, but they are diagonal and not in objective (i<j).
        
        # Helper to get current objective contribution for a set of pairs
        # We will just recalculate the delta by checking the specific pairs.
        
        # Pairs affected in old_c:
        # (u, y) for y != u, v
        # (v, y) for y != u, v
        # (u, v) is not in objective (i<j, but Sq is symmetric, we count i<j).
        # Wait, Sq[u,v] is common neighbors of u and v.
        # Does removing (u,v) change Sq[u,v]?
        # Sq[u,v] = sum_z A[u,z] A[z,v].
        # The edge (u,v) is not part of this sum (z cannot be u or v in a simple path? No, z can be any vertex).
        # If z=u, A[u,u]=0. If z=v, A[v,v]=0.
        # So Sq[u,v] does NOT depend on A[u,v].
        # However, Sq[x,y] for x!=u,v depends on A[x,u] and A[u,y] etc.
        
        # Let's just use the matrix update logic to compute the change in Sq values for the affected indices.
        
        # Affected indices in Sq[old_c]:
        # Row u, Col u (diagonal, ignore for obj)
        # Row v, Col v (diagonal, ignore)
        # Row u, all cols
        # Row v, all cols
        # All rows, Col u
        # All rows, Col v
        
        # We need to sum max(0, Sq_new[i,j] - 1) - max(0, Sq_old[i,j] - 1) for i<j in affected set.
        
        # Let's identify the unique pairs (i,j) with i<j that are affected.
        # Affected rows/cols: u, v.
        # Pairs: (u, j) for j != u, v. (v, j) for j != u, v.
        # Also (u, v) itself? No, Sq[u,v] doesn't change.
        
        # Let's compute the change in Sq for these pairs.
        # Change in Sq[x, y]:
        # If x == u and y != u, v:
        #   Sq_new[u, y] = Sq_old[u, y] - A[u, u] - A[u, v] - A[u, y] - A[v, y] + 0
        #   Wait, the formula was:
        #   Delta Sq = - (A E_uv + A E_vu + E_uv A + E_vu A) + E_uu + E_vv
        #   For Sq[u, y] (y not u,v):
        #   (A E_uv)[u, y] = A[u, u] = 0
        #   (A E_vu)[u, y] = A[u, v]
        #   (E_uv A)[u, y] = A[v, y]  (Row u of E_uv A is Row v of A? No. E_uv A has row u = row v of A? No. E_uv has 1 at (u,v). (E_uv A)_{u, y} = sum_k E_uv[u,k] A[k,y] = A[v,y].)
        #   (E_vu A)[u, y] = 0
        #   So Delta Sq[u, y] = - (A[u, v] + A[v, y])
        #   Similarly for Sq[v, y] (y not u,v):
        #   Delta Sq[v, y] = - (A[v, u] + A[u, y]) = - (1 + A[u, y])
        
        # Let's just apply the update to a temporary copy of the relevant parts?
        # Or just calculate the new Sq values for the affected pairs.
        
        # Affected pairs for old_c:
        # (u, j) for j in 0..n-1, j != u
        # (v, j) for j in 0..n-1, j != v
        # Note: (u,v) is in both lists, but Sq[u,v] doesn't change.
        
        # Let's iterate over j
        # For each j != u, v:
        #   pair (u, j): 
        #     old_val = Sq[old_c][u, j]
        #     new_val = old_val - A[old_c][u, v] - A[old_c][v, j]  # A[u,v] is 1
        #     delta += max(0, new_val - 1) - max(0, old_val - 1)
        #   pair (v, j):
        #     old_val = Sq[old_c][v, j]
        #     new_val = old_val - A[old_c][v, u] - A[old_c][u, j] # A[v,u] is 1
        #     delta += max(0, new_val - 1) - max(0, old_val - 1)
        
        # Wait, is this formula correct?
        # Sq_new[u, j] = Sq_old[u, j] - A[u, v] - A[v, j]
        # Let's verify with small example.
        # Path u-z-j.
        # If z=v, path u-v-j. Exists if A[u,v]=1 and A[v,j]=1.
        # Removing A[u,v] removes this path. So count decreases by 1 if A[v,j]=1.
        # So Sq_new[u, j] = Sq_old[u, j] - A[v, j].
        # My formula had -A[u,v] - A[v,j]. A[u,v] is 1.
        # Why -A[u,v]?
        # Let's re-derive.
        # Sq[u, j] = sum_z A[u,z] A[z,j].
        # Terms with z=v: A[u,v] A[v,j].
        # Since we remove A[u,v], this term becomes 0.
        # So decrease is A[u,v] A[v,j] = 1 * A[v,j] = A[v,j].
        # So Sq_new[u, j] = Sq_old[u, j] - A[v, j].
        
        # Similarly for Sq[v, j]:
        # Terms with z=u: A[v,u] A[u,j].
        # Decrease is A[v,u] A[u,j] = A[u,j].
        # So Sq_new[v, j] = Sq_old[v, j] - A[u, j].
        
        # What about Sq[u, v]?
        # Sq[u, v] = sum_z A[u,z] A[z,v].
        # Terms with z?
        # z cannot be u or v (diagonal 0).
        # Does removing A[u,v] affect Sq[u,v]?
        # The sum is over z. A[u,v] is not in the sum (unless z=v, but A[v,v]=0).
        # So Sq[u,v] is unchanged.
        
        # So for old_c, delta is:
        for j in range(n):
            if j == u or j == v:
                continue
            # Pair (u, j)
            i, k = u, j
            if i > k: i, k = k, i
            old_val = Sq[old_c][i, k]
            # Determine which one is u and which is j
            if u < j:
                # u is i, j is k
                # Sq[u, j] decreases by A[old_c][v, j]
                dec = A[old_c][v, j]
            else:
                # j is i, u is k
                # Sq[j, u] decreases by A[old_c][v, u] = 1?
                # Sq[j, u] = sum_z A[j,z] A[z,u].
                # Term z=v: A[j,v] A[v,u].
                # Decrease is A[j,v] * 1 = A[j,v].
                dec = A[old_c][j, v]
            
            new_val = old_val - dec
            delta += max(0, new_val - 1) - max(0, old_val - 1)
            
            # Pair (v, j)
            i, k = v, j
            if i > k: i, k = k, i
            old_val = Sq[old_c][i, k]
            if v < j:
                # v is i, j is k
                # Sq[v, j] decreases by A[old_c][u, j]
                dec = A[old_c][u, j]
            else:
                # j is i, v is k
                # Sq[j, v] decreases by A[old_c][u, v] = 1?
                # Sq[j, v] = sum_z A[j,z] A[z,v].
                # Term z=u: A[j,u] A[u,v].
                # Decrease is A[j,u] * 1 = A[j,u].
                dec = A[old_c][j, u]
            
            new_val = old_val - dec
            delta += max(0, new_val - 1) - max(0, old_val - 1)
            
        # Now for new_c: add edge (u,v)
        # Sq_new[u, j] = Sq_old[u, j] + A[new_c][v, j]
        # Sq_new[v, j] = Sq_old[v, j] + A[new_c][u, j]
        
        for j in range(n):
            if j == u or j == v:
                continue
            i, k = u, j
            if i > k: i, k = k, i
            old_val = Sq[new_c][i, k]
            if u < j:
                inc = A[new_c][v, j]
            else:
                inc = A[new_c][j, v]
            new_val = old_val + inc
            delta += max(0, new_val - 1) - max(0, old_val - 1)
            
            i, k = v, j
            if i > k: i, k = k, i
            old_val = Sq[new_c][i, k]
            if v < j:
                inc = A[new_c][u, j]
            else:
                inc = A[new_c][j, u]
            new_val = old_val + inc
            delta += max(0, new_val - 1) - max(0, old_val - 1)
            
        # Accept if delta < 0 or with prob exp(-delta/T)
        if delta < 0 or random.random() < np.exp(-delta / T):
            # Apply change
            # Update A
            A[old_c][u, v] = 0
            A[old_c][v, u] = 0
            A[new_c][u, v] = 1
            A[new_c][v, u] = 1
            colors[k] = new_c
            
            # Update Sq
            # We can just recompute the affected parts or use the delta logic in reverse?
            # Easier: update Sq using the same logic but with actual values.
            
            # Update Sq[old_c]
            for j in range(n):
                if j == u or j == v:
                    continue
                # Pair (u, j)
                i, k = u, j
                if i > k: i, k = k, i
                if u < j:
                    dec = A[old_c][v, j]
                else:
                    dec = A[old_c][j, v]
                Sq[old_c][i, k] -= dec
                Sq[old_c][k, i] -= dec
                
                # Pair (v, j)
                i, k = v, j
                if i > k: i, k = k, i
                if v < j:
                    dec = A[old_c][u, j]
                else:
                    dec = A[old_c][j, u]
                Sq[old_c][i, k] -= dec
                Sq[old_c][k, i] -= dec
            
            # Update Sq[new_c]
            for j in range(n):
                if j == u or j == v:
                    continue
                # Pair (u, j)
                i, k = u, j
                if i > k: i, k = k, i
                if u < j:
                    inc = A[new_c][v, j]
                else:
                    inc = A[new_c][j, v]
                Sq[new_c][i, k] += inc
                Sq[new_c][k, i] += inc
                
                # Pair (v, j)
                i, k = v, j
                if i > k: i, k = k, i
                if v < j:
                    inc = A[new_c][u, j]
                else:
                    inc = A[new_c][j, u]
                Sq[new_c][i, k] += inc
                Sq[new_c][k, i] += inc
            
            current_violations += delta
            if current_violations == 0:
                return True, colors.tolist()
                
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors.copy()
                
        T *= 0.999
        if T < min_T:
            T = min_T
            
    return False, best_colors.tolist()

