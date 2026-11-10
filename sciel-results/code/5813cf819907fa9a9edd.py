import numpy as np
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    
    # Field operations for F_49 = F_7[alpha] where alpha^2 = 3
    def add_f(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    def mul_f(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = (a0b0 + a1b1*k) + (a0b1 + a1b0)*alpha
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
        
    # Precompute all field elements as pairs
    field_elements = []
    for a in range(p):
        for b in range(p):
            field_elements.append((a, b))
            
    # Map pair to index
    pair_to_idx = {pair: i for i, pair in enumerate(field_elements)}
    
    # Precompute multiplication table
    mul_table = np.zeros((q, q), dtype=np.int64)
    for i, a in enumerate(field_elements):
        for j, b in enumerate(field_elements):
            prod = mul_f(a, b)
            mul_table[i, j] = pair_to_idx[prod]
            
    # Precompute addition table
    add_table = np.zeros((q, q), dtype=np.int64)
    for i, a in enumerate(field_elements):
        for j, b in enumerate(field_elements):
            s = add_f(a, b)
            add_table[i, j] = pair_to_idx[s]
            
    n_dirs = q + 1  # 50
    total_points = q * q
    
    # Precompute lines: lines[point_idx][dir_idx] is an array of point indices
    # point_idx = x_idx * q + y_idx
    lines = [None] * total_points
    for point_idx in range(total_points):
        x_idx = point_idx // q
        y_idx = point_idx % q
        
        lines[point_idx] = [None] * n_dirs
        for d in range(n_dirs):
            if d < q:
                # Slope m = field_elements[d]
                # Points: (x + t, y + t*m)
                # t_idx ranges 0..q-1
                # nx_idx = add_table[x_idx, t_idx]
                # ny_idx = add_table[y_idx, mul_table[t_idx, d]]
                t_indices = np.arange(q)
                nx_indices = add_table[x_idx, t_indices]
                tm_indices = mul_table[t_indices, d]
                ny_indices = add_table[y_idx, tm_indices]
                pt_indices = nx_indices * q + ny_indices
                lines[point_idx][d] = pt_indices
            else:
                # Vertical
                t_indices = np.arange(q)
                nx_indices = np.full(q, x_idx, dtype=np.int64)
                ny_indices = add_table[y_idx, t_indices]
                pt_indices = nx_indices * q + ny_indices
                lines[point_idx][d] = pt_indices

    def greedy_nikodym(n_restarts=50):
        best_S = set()
        best_size = 0
        
        for restart in range(n_restarts):
            S = set()
            # C[y] is a bitmask of valid directions for y
            # Initially all directions are valid
            full_mask = (1 << n_dirs) - 1
            C = [full_mask] * total_points
            
            order = list(range(total_points))
            random.shuffle(order)
            
            for x in order:
                # Check if x has any valid direction
                if C[x] == 0:
                    continue
                
                # Determine affected points and directions to remove
                # affected[y] = bitmask of directions to remove from y
                affected = {}
                
                for d in range(n_dirs):
                    pts = lines[x][d]
                    bit = 1 << d
                    for y in pts:
                        if y != x:
                            if y in affected:
                                affected[y] |= bit
                            else:
                                affected[y] = bit
                                
                # Check if adding x is valid
                valid = True
                for y, mask in affected.items():
                    if (C[y] & ~mask) == 0:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Add x to S
                S.add(x)
                
                # Update C for affected points
                for y, mask in affected.items():
                    C[y] &= ~mask
                    
            if len(S) > best_size:
                best_size = len(S)
                best_S = S.copy()
                
        return best_S

    S = greedy_nikodym(n_restarts=50)
    
    # N = complement of S
    N = [i for i in range(total_points) if i not in S]
    
    # Convert to array of shape (k, 2, 2)
    arr = np.zeros((len(N), 2, 2), dtype=np.int64)
    for i, point_idx in enumerate(N):
        x_idx = point_idx // q
        y_idx = point_idx % q
        x = field_elements[x_idx]
        y = field_elements[y_idx]
        arr[i, 0, 0] = x[0] % p
        arr[i, 0, 1] = x[1] % p
        arr[i, 1, 0] = y[0] % p
        arr[i, 1, 1] = y[1] % p
        
    return arr
