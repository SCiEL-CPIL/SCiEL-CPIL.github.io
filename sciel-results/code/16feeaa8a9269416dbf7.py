import time
import numpy as np

DIMENSIONS = [3, 4, 4]
DEADLINE = 120.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    def elapsed():
        return time.time() - start_time
    
    def time_remaining():
        return DEADLINE - 5.0 - elapsed()
    
    def schoolbook_decomp():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    def verify_decomp(U, V, W, tol=1e-8):
        rank = len(U[0])
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_rank_47():
        """Try to construct a rank 47 decomposition"""
        if time_remaining() < 10.0:
            return None
        
        # Start with schoolbook and try to merge two terms
        # Take terms for (i1,j1,k1) and (i2,j2,k2) and see if we can combine them
        
        # Strategy: Take the schoolbook and perturb two columns
        # Let's try: take columns for (0,0,0) and (0,0,1) and modify them
        
        # Actually, let's try a more systematic approach
        # Use the fact that we have more unknowns than equations
        
        # Initialize with schoolbook-like structure but with rank 47
        # Keep 46 terms as schoolbook, and use 1 term to cover the remaining 2 entries
        
        # Schoolbook terms are indexed by (i,j,k)
        # Let's keep all terms except (0,3,3), and try to cover (0,3,3) with modifications to other terms
        
        # Better: try to find if there's a linear relation among the 48 schoolbook terms
        # that allows us to drop one
        
        # For now, let's try a direct construction
        # Use rank 47 with integer/half-integer entries
        
        # Try: modify the last two schoolbook terms
        # Term 46: (i=2,j=3,k=2) -> U[2*4+3]=U[11], V[3*4+2]=V[14], W[2*3+2]=W[8]
        # Term 47: (i=2,j=3,k=3) -> U[11], V[3*4+3]=V[15], W[3*3+2]=W[11]
        
        # These two terms share the same U entry (U[11])
        # Can we merge them? We'd need V and W to also align, but they don't.
        
        # Let's try a different pair: terms that share U, V, or W entries
        # Term (i,j,k) and (i,j,k') share U[i,j]
        # Term (i,j,k) and (i',j,k) share V[j,k]  
        # Term (i,j,k) and (i,j',k) share W[k,i]... wait, W is indexed by (k,i)
        
        # Two terms (i,j,k) and (i,j,k') share U[i*m+j]
        # Two terms (i,j,k) and (i',j,k) share V[j*p+k]
        # Two terms (i,j,k) and (i,j',k) share... W[k*n+i] is same if i is same
        
        # So terms with same i and j share U, terms with same j and k share V, terms with same k and i share W
        
        # Let's try to merge terms (0,0,0) and (0,0,1): they share U[0*4+0]=U[0]
        # U[0] appears in both. V[0*4+0]=V[0] and V[0*4+1]=V[1] are different.
        # W[0*3+0]=W[0] and W[1*3+0]=W[3] are different.
        
        # To merge, we'd need to set U[0] to something, and have the V and W entries work out.
        # But V[0] and V[1] are in different rows, so they're independent.
        
        # This approach is getting complex. Let me try a numerical approach.
        
        # Use least squares to find a rank 47 decomposition
        if time_remaining() < 20.0:
            return None
        
        try:
            # Initialize with schoolbook and drop one term, then adjust
            # Drop term (2,3,3) which is the last one
            # Keep terms 0-45 (46 terms) and add 1 more term to cover the gap
            
            # Actually, let's just try to solve it directly
            # We need 48 equations, 47*(12+16+12)=1880 unknowns
            
            # Use a simple iterative method
            # Initialize U, V, W with random values scaled appropriately
            
            rng = np.random.default_rng(42)
            
            for attempt in range(50):
                if time_remaining() < 15.0:
                    break
                
                # Initialize with schoolbook-like structure
                U = np.zeros((n * m, 47))
                V = np.zeros((m * p, 47))
                W = np.zeros((p * n, 47))
                
                # Fill first 46 terms with schoolbook
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col < 46:
                                U[i * m + j, col] = 1
                                V[j * p + k, col] = 1
                                W[k * n + i, col] = 1
                            col += 1
                
                # The 47th term (index 46) needs to help cover the missing entry
                # The missing entry is (i=2,j=3,k=3) since we have 48 entries and 46 schoolbook terms
                # Wait, we have 48 entries total. If we use 46 schoolbook terms, we're missing 2 entries.
                # Let's say we drop the last two: (2,3,2) and (2,3,3)
                
                # Reset: use 46 schoolbook terms for the first 46 (i,j,k) in order
                # The last two (i,j,k) pairs are (2,3,2) and (2,3,3)
                
                # For the 47th term, we need to cover both (2,3,2) and (2,3,3)
                # U[2*4+3]=U[11] must be non-zero for both
                # V[3*4+2]=V[14] for (2,3,2) and V[3*4+3]=V[15] for (2,3,3)
                # W[2*3+2]=W[8] for (2,3,2) and W[3*3+2]=W[11] for (2,3,3)
                
                # So we need:
                # U[11,46] * V[14,46] * W[8,46] = 1  (for entry (2,3,2))
                # U[11,46] * V[15,46] * W[11,46] = 1  (for entry (2,3,3))
                
                # And for all other (i,j,k), the 47th term should contribute 0
                # This means for all other (i,j,k), at least one of U[i*m+j,46], V[j*p+k,46], W[k*n+i,46] must be 0
                
                # Set U[11,46] = 1
                # Then V[14,46] * W[8,46] = 1 and V[15,46] * W[11,46] = 1
                # Let V[14,46] = 1, W[8,46] = 1, V[15,46] = 1, W[11,46] = 1
                
                U[11, 46] = 1
                V[14, 46] = 1
                W[8, 46] = 1
                V[15, 46] = 1
                W[11, 46] = 1
                
                # Now check: does this 47th term accidentally contribute to other (i,j,k) entries?
                # U[11,46]=1 means it contributes to all (i,j,k) where i*m+j=11, i.e., i=2,j=3
                # For i=2,j=3, k can be 0,1,2,3
                # V[3*4+k, 46] = V[12+k, 46]
                # We set V[14,46]=1 and V[15,46]=1, so V[12,46]=0 and V[13,46]=0
                # So for k=0: V[12,46]=0, no contribution
                # For k=1: V[13,46]=0, no contribution
                # For k=2: V[14,46]=1, W[2*3+2]=W[8,46]=1, contribution = 1*1*1=1
                # For k=3: V[15,46]=1, W[3*3+2]=W[11,46]=1, contribution = 1*1*1=1
                
                # But we also need the schoolbook terms to not contribute to (2,3,2) and (2,3,3)
                # In our setup, the schoolbook terms are for the first 46 (i,j,k) in order
                # (2,3,2) is the 47th entry (index 46) and (2,3,3) is the 48th (index 47)
                # So we didn't include schoolbook terms for these two entries
                # Good!
                
                # Now verify
                U_list = [[int(x) for x in row] for row in U]
                V_list = [[int(x) for x in row] for row in V]
                W_list = [[int(x) for x in row] for row in W]
                
                if verify_decomp(U_list, V_list, W_list):
                    return U_list, V_list, W_list
                else:
                    # Check which entries fail
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for r in range(47):
                                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                                if abs(val - 1.0) > 1e-8:
                                    pass  # Debug: print(f"Fail at ({i},{j},{k}): {val}")
        
        except Exception:
            pass
        
        return None
    
    # Try to find rank 47
    result = try_rank_47()
    if result is not None:
        return {"U": result[0], "V": result[1], "W": result[2]}
    
    # Fall back to schoolbook
    U, V, W = schoolbook_decomp()
    return {"U": U, "V": V, "W": W}
