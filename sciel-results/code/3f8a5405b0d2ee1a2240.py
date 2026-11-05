import time
import random
import numpy as np

def _calc_c4_count(A_sq):
    """
    Calculate number of C4s from A^2 matrix.
    Formula: 1/8 * (||A^2||_F^2 - sum(diag(A^2)^2) - sum(A^2) + sum(diag(A^2)))
    """
    s1 = np.sum(A_sq * A_sq)
    diag = np.diag(A_sq)
    s2 = np.sum(diag * diag)
    s3 = np.sum(A_sq)
    s4 = np.sum(diag)
    return (s1 - s2 - s3 + s4) / 8.0

def _update_a_sq(A, A_sq, u, v, sign):
    """
    Update A and A_sq when adding (sign=1) or removing (sign=-1) edge (u,v).
    A is the adjacency matrix.
    A_sq is the squared adjacency matrix.
    """
    # Update A
    A[u, v] += sign
    A[v, u] += sign
    
    # Update A_sq
    # A_new = A + sign * E_uv
    # A_new^2 = A^2 + sign * (A E + E A) + sign^2 * E^2
    # E has 1 at (u,v) and (v,u)
    
    # Term 1: A E
    # (A E)_{i, u} = A_{i, v}
    # (A E)_{i, v} = A_{i, u}
    # Note: A is updated already? 
    # If we use the OLD A for the cross terms, we must be careful.
    # Let's use the current A (which is NEW if we added, OLD if we removed? No, we update A first).
    # If we update A first, then A is NEW.
    # We want A_old^2 -> A_new^2.
    # A_new = A_old + D.
    # A_new^2 = A_old^2 + A_old D + D A_old + D^2.
    # Here D = sign * E_uv.
    # So we need A_old.
    # If sign=1 (add), A is currently A_new. We need A_old = A - D = A - sign*E.
    # If sign=-1 (remove), A is currently A_new (which is A_old - E). We need A_old = A + E.
    # So A_old = A - sign * E_uv.
    
    # Let's just compute the delta using the OLD A values.
    # To get OLD A values, we can temporarily revert? Or just use the fact that 
    # only row/col u and v change.
    
    # Let's do it safely:
    # 1. Save old values of A[u, :] and A[:, v] etc?
    # Actually, simpler:
    # The change in A_sq is:
    # Delta_A_sq = sign * (A_old E + E A_old) + sign^2 * E^2
    # Since sign^2 = 1, E^2 has 1 at (u,u) and (v,v).
    # E^2 = [[0..1..0], ..., [0..1..0], ...] (1 at u,u and v,v)
    
    # We need A_old.
    # If we are adding (sign=1), A is A_new. A_old = A - E.
    # If we are removing (sign=-1), A is A_new. A_old = A + E.
    # So A_old = A - sign * E.
    
    # Let's extract A_old rows/cols needed.
    # (A_old E)_{i, u} = A_old_{i, v}
    # (A_old E)_{i, v} = A_old_{i, u}
    # (E A_old)_{u, j} = A_old_{v, j}
    # (E A_old)_{v, j} = A_old_{u, j}
    
    # A_old_{i, v} = A_{i, v} - sign * E_{i, v}
    # E_{i, v} is 1 only if i==u.
    # So A_old_{i, v} = A_{i, v} - (sign if i==u else 0)
    
    # Similarly A_old_{i, u} = A_{i, u} - (sign if i==v else 0)
    
    # A_old_{v, j} = A_{v, j} - (sign if j==u else 0)
    # A_old_{u, j} = A_{u, j} - (sign if j==v else 0)
    
    # We update A_sq in place.
    
    # 1. Update column u of A_sq (from A_old E)
    # A_sq[:, u] += sign * A_old[:, v]
    # A_sq[:, v] += sign * A_old[:, u]
    
    # 2. Update row u of A_sq (from E A_old)
    # A_sq[u, :] += sign * A_old[v, :]
    # A_sq[v, :] += sign * A_old[u, :]
    
    # 3. Update diagonal from E^2
    # A_sq[u, u] += 1
    # A_sq[v, v] += 1
    
    # Let's implement this carefully to avoid overwriting issues.
    # We need A_old[:, v] and A_old[:, u].
    # A_old[:, v] = A[:, v] - sign * (1 if i==u else 0)
    # So A_old[:, v] is A[:, v] with A[u, v] reduced by sign.
    # But A[u, v] is currently A_new[u, v] = A_old[u, v] + sign.
    # So A_old[u, v] = A[u, v] - sign.
    # For i != u, A_old[i, v] = A[i, v].
    
    # Let's copy the columns/rows we need.
    col_v_old = A[:, v].copy()
    col_v_old[u] -= sign # Revert A[u,v] to old value
    
    col_u_old = A[:, u].copy()
    col_u_old[v] -= sign # Revert A[v,u] to old value
    
    row_v_old = A[v, :].copy()
    row_v_old[u] -= sign # Revert A[v,u] to old value
    
    row_u_old = A[u, :].copy()
    row_u_old[v] -= sign # Revert A[u,v] to old value
    
    # Apply updates to A_sq
    # A_sq[:, u] += sign * col_v_old
    # A_sq[:, v] += sign * col_u_old
    # A_sq[u, :] += sign * row_v_old
    # A_sq[v, :] += sign * row_u_old
    # A_sq[u, u] += 1
    # A_sq[v, v] += 1
    
    # Note: The intersection at (u,u), (u,v), (v,u), (v,v) gets multiple updates.
    # This is correct because the formula is additive.
    
    A_sq[:, u] += sign * col_v_old
    A_sq[:, v] += sign * col_u_old
    A_sq[u, :] += sign * row_v_old
    A_sq[v, :] += sign * row_u_old
    A_sq[u, u] += 1
    A_sq[v, v] += 1

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_n = 4
    best_colors = [0] * 6
    
    # Try n from 26 down to 5? Or up?
    # We know 26 is possible. Let's try 26 first.
    # If we find it, we try 27, 28.
    
    target_ns = [26, 27, 28, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]
    
    for n in target_ns:
        if time.time() > deadline - 10: # Leave 10s margin
            break
            
        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        # Initialize A and A_sq
        A = [np.zeros((n, n), dtype=int) for _ in range(5)]
        A_sq = [np.zeros((n, n), dtype=int) for _ in range(5)]
        
        for idx in range(num_edges):
            i = 0
            j = 0
            # Find i, j for idx
            # This is slow. Precompute edge list.
            pass
        
        # Precompute edge list
        edges = []
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                edges.append((i, j))
                idx += 1
        
        for idx, (i, j) in enumerate(edges):
            c = colors[idx]
            A[c][i, j] = 1
            A[c][j, i] = 1
        
        # Compute initial A_sq
        for c in range(5):
            A_sq[c] = A[c] @ A[c]
        
        # Initial Cost
        total_cost = 0
        for c in range(5):
            total_cost += _calc_c4_count(A_sq[c])
        
        if total_cost == 0:
            best_n = n
            best_colors = colors.copy()
            continue # Try larger n? No, we are iterating down? 
            # Actually, if we find valid for 26, we should try 27.
            # My list is 26, 27, 28... so it will try 27 next.
            
        # Simulated Annealing
        T = 1.0
        T_min = 0.001
        alpha = 0.999
        steps = 0
        max_steps = 500000 # High limit, time controlled
        
        while T > T_min and time.time() < deadline - 5:
            steps += 1
            # Pick random edge
            idx = random.randint(0, num_edges - 1)
            u, v = edges[idx]
            old_c = colors[idx]
            new_c = random.randint(0, 4)
            
            if old_c == new_c:
                continue
                
            # Calculate Delta Cost
            # Remove from old_c
            c4_old_old = _calc_c4_count(A_sq[old_c])
            _update_a_sq(A[old_c], A_sq[old_c], u, v, -1)
            c4_old_new = _calc_c4_count(A_sq[old_c])
            
            # Add to new_c
            c4_new_old = _calc_c4_count(A_sq[new_c])
            _update_a_sq(A[new_c], A_sq[new_c], u, v, 1)
            c4_new_new = _calc_c4_count(A_sq[new_c])
            
            delta = (c4_old_new - c4_old_old) + (c4_new_new - c4_new_old)
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                colors[idx] = new_c
                total_cost += delta
            else:
                # Reject: Revert
                # Remove from new_c
                _update_a_sq(A[new_c], A_sq[new_c], u, v, -1)
                # Add to old_c
                _update_a_sq(A[old_c], A_sq[old_c], u, v, 1)
                
            T *= alpha
            
            if total_cost == 0:
                best_n = n
                best_colors = colors.copy()
                # If we found a solution for n, we can break and try n+1?
                # But we are iterating n in target_ns.
                # If we found 26, we break this loop and go to 27.
                break
                
    return {'n': best_n, 'colors': best_colors}
