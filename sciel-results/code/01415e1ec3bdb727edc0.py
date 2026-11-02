import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 120.0  # seconds, safety margin
    
    n, m, p = 3, 4, 6
    terms = []
    
    def time_ok():
        return time.time() - start_time < DEADLINE
    
    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        """Strassen 2x2x2: A[i1:i3, j1:j3] * B[j1:j3, k1:k3] -> C[i1:i3, k1:k3]"""
        a, b = (i1, j1), (i1, j2)
        c, d = (i2, j1), (i2, j2)
        e, f = (j1, k1), (j1, k2)
        g, h = (j2, k1), (j2, k2)
        C11, C12 = (i1, k1), (i1, k2)
        C21, C22 = (i2, k1), (i2, k2)
        
        # M1 = (a+d)(e+h)
        terms.append((
            {a: 1, d: 1}, {e: 1, h: 1}, {C11: 1, C22: 1}
        ))
        # M2 = (c+d)e
        terms.append((
            {c: 1, d: 1}, {e: 1}, {C21: 1, C22: -1}
        ))
        # M3 = a(f-h)
        terms.append((
            {a: 1}, {f: 1, h: -1}, {C12: 1, C22: 1}
        ))
        # M4 = d(g-e)
        terms.append((
            {d: 1}, {g: 1, e: -1}, {C11: 1, C21: 1}
        ))
        # M5 = (a+b)h
        terms.append((
            {a: 1, b: 1}, {h: 1}, {C11: -1, C12: 1}
        ))
        # M6 = (c-a)(e+f)
        terms.append((
            {c: 1, a: -1}, {e: 1, f: 1}, {C22: 1}
        ))
        # M7 = (b-d)(g+h)
        terms.append((
            {b: 1, d: -1}, {g: 1, h: 1}, {C11: 1}
        ))
    
    def add_schoolbook(i, j, k):
        terms.append((
            {(i, j): 1}, {(j, k): 1}, {(i, k): 1}
        ))
    
    # Row 0: schoolbook (4 middle * 6 right = 24 terms)
    for j in range(m):
        for k in range(p):
            add_schoolbook(0, j, k)
    
    # Rows 1-2: Strassen on 2x2x2 blocks
    # Split m=4 into two 2-col groups, p=6 into three 2-col groups
    for j1 in [0, 2]:
        for k1 in [0, 2, 4]:
            if not time_ok():
                break
            add_strassen_block(1, 2, j1, j1 + 1, k1, k1 + 1)
        if not time_ok():
            break
    
    rank = len(terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][col] = coeff
    
    # Convert to numpy for rank reduction
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Try to reduce rank by finding linearly dependent columns
    # A term t is redundant if column t of U, V, W can be expressed as linear combo of others
    # This is equivalent to checking if the tensor is in the span of the other rank-1 terms
    
    best_terms = terms
    best_rank = rank
    
    if time_ok() and rank > 1:
        # Attempt iterative rank reduction
        # For each term, check if removing it still allows reconstruction
        # This is expensive, so we do a simpler check: look for duplicate or proportional terms
        
        # First, check for exact duplicates or scalar multiples
        seen = set()
        unique_terms = []
        for term in terms:
            # Create a canonical key from the term
            u_key = tuple(sorted(term[0].items()))
            v_key = tuple(sorted(term[1].items()))
            w_key = tuple(sorted(term[2].items()))
            key = (u_key, v_key, w_key)
            if key not in seen:
                seen.add(key)
                unique_terms.append(term)
        
        if len(unique_terms) < len(terms):
            best_terms = unique_terms
            best_rank = len(unique_terms)
        
        # Try to find linear dependencies using numpy
        if time_ok():
            U_np = np.array(best_terms and [[0]*best_rank for _ in range(n*m)] or U, dtype=float)
            # Rebuild from best_terms
            U_np = np.zeros((n*m, best_rank))
            V_np = np.zeros((m*p, best_rank))
            W_np = np.zeros((p*n, best_rank))
            
            for col, (u_dict, v_dict, w_dict) in enumerate(best_terms):
                for (i, j), coeff in u_dict.items():
                    U_np[i * m + j, col] = coeff
                for (j, k), coeff in v_dict.items():
                    V_np[j * p + k, col] = coeff
                for (i, k), coeff in w_dict.items():
                    W_np[k * n + i, col] = coeff
            
            # Check for linearly dependent columns using SVD
            # If the combined tensor has rank < best_rank, we can reduce
            # But this is complex; instead, try a greedy removal approach
            
            # Greedy: for each term, check if it's approximately in the span of others
            reduced_terms = list(best_terms)
            changed = True
            while changed and time_ok():
                changed = False
                for idx in range(len(reduced_terms)):
                    if len(reduced_terms) <= 1:
                        break
                    # Build matrices without term idx
                    U_test = np.zeros((n*m, len(reduced_terms)-1))
                    V_test = np.zeros((m*p, len(reduced_terms)-1))
                    W_test = np.zeros((p*n, len(reduced_terms)-1))
                    
                    for col, (u_dict, v_dict, w_dict) in enumerate(reduced_terms):
                        if col == idx:
                            continue
                        target_col = col if col < idx else col - 1
                        for (i, j), coeff in u_dict.items():
                            U_test[i * m + j, target_col] = coeff
                        for (j, k), coeff in v_dict.items():
                            V_test[j * p + k, target_col] = coeff
                        for (i, k), coeff in w_dict.items():
                            W_test[k * n + i, target_col] = coeff
                    
                    # Check if the removed term's contribution can be replicated
                    # This is a complex check; skip for now to save time
                    
                    if not time_ok():
                        break
                
                # If we found a removable term, we'd remove it here
                # For now, just break to save time
                break
        
        best_terms = reduced_terms if 'reduced_terms' in dir() else best_terms
        best_rank = len(best_terms)
    
    # Final construction from best_terms
    rank = best_rank
    U_final = [[0] * rank for _ in range(n * m)]
    V_final = [[0] * rank for _ in range(m * p)]
    W_final = [[0] * rank for _ in range(p * n)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(best_terms):
        for (i, j), coeff in u_dict.items():
            U_final[i * m + j][col] = coeff
        for (j, k), coeff in v_dict.items():
            V_final[j * p + k][col] = coeff
        for (i, k), coeff in w_dict.items():
            W_final[k * n + i][col] = coeff
    
    return {"U": U_final, "V": V_final, "W": W_final}
