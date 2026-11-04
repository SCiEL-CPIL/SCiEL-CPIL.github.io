import numpy as np
import time
from itertools import product, combinations

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    def verify(U, V, W):
        """Verify that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all valid (i,j,k)"""
        rank = U.shape[1]
        # Check valid entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check invalid entries are 0
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if j == j2 and k == k2 and i == i2:
                        continue
                    s = 0.0
                    for r in range(rank):
                        s += U[a, r] * V[b, r] * W[c, r]
                    if abs(s) > 1e-9:
                        return False
        return True

    def snap_to_half_int(U, V, W):
        """Snap values to nearest half-integer"""
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        return U_s, V_s, W_s

    # Start with the known rank-17 solution
    rank17 = 17
    U17 = np.zeros((n*m, rank17))
    V17 = np.zeros((m*p, rank17))
    W17 = np.zeros((p*n, rank17))
    
    # Strassen 2x2x2 for i in {0,1}, j in {0,1}, k in {0,1}
    # Col 0: (A00+A11)(B00+B11) -> contributes to C00, C10
    U17[0,0]=1; U17[4,0]=1
    V17[0,0]=1; V17[4,0]=1
    W17[0,0]=1; W17[3,0]=1
    
    # Col 1: (A10+A11)B00 -> C10, C11
    U17[3,1]=1; U17[4,1]=1
    V17[0,1]=1
    W17[1,1]=1; W17[3,1]=-1
    
    # Col 2: A00(B01-B11) -> C01, C11
    U17[0,2]=1
    V17[1,2]=1; V17[4,2]=-1
    W17[2,2]=1; W17[3,2]=1
    
    # Col 3: A11(B10-B00) -> C10, C11
    U17[4,3]=1
    V17[3,3]=1; V17[0,3]=-1
    W17[0,3]=1; W17[1,3]=1
    
    # Col 4: (A00+A01)B11 -> C01, C11
    U17[0,4]=1; U17[1,4]=1
    V17[4,4]=1
    W17[0,4]=-1; W17[2,4]=1
    
    # Col 5: (A10-A00)(B00+B01) -> C10, C11
    U17[3,5]=1; U17[0,5]=-1
    V17[0,5]=1; V17[1,5]=1
    W17[3,5]=1
    
    # Col 6: (A01-A11)(B10+B11) -> C01, C11
    U17[1,6]=1; U17[4,6]=-1
    V17[3,6]=1; V17[4,6]=1
    W17[0,6]=1
    
    # 2x1x1 for j in {0,1}, k=2
    U17[0,7]=1; V17[2,7]=1; W17[4,7]=1
    U17[1,8]=1; V17[5,8]=1; W17[4,8]=1
    U17[3,9]=1; V17[2,9]=1; W17[5,9]=1
    U17[4,10]=1; V17[5,10]=1; W17[5,10]=1
    
    # 2x1x3 for j=2
    U17[2,11]=1; V17[6,11]=1; W17[0,11]=1
    U17[2,12]=1; V17[7,12]=1; W17[2,12]=1
    U17[2,13]=1; V17[8,13]=1; W17[4,13]=1
    U17[5,14]=1; V17[6,14]=1; W17[1,14]=1
    U17[5,15]=1; V17[7,15]=1; W17[3,15]=1
    U17[5,16]=1; V17[8,16]=1; W17[5,16]=1
    
    if not verify(U17, V17, W17):
        raise ValueError("Rank 17 solution verification failed")
    
    best_U, best_V, best_W = U17.copy(), V17.copy(), W17.copy()
    best_rank = 17

    # Tensor completion approach for rank 16
    # Strategy: Fix a sparsity pattern (support) for U, V, W with rank 16
    # and solve the resulting linear system for the non-zero entries.
    
    # The key insight: we can try to merge one of the simple product terms
    # (columns 7-16) with the Strassen columns by allowing the Strassen columns
    # to have additional non-zero entries in the k=2 or j=2 blocks.
    
    # Let's try a structured approach:
    # Keep the 7 Strassen columns (0-6) but allow them to have additional
    # non-zero entries in the k=2 block (V entries at j=0,1 and k=2)
    # and reduce the number of direct product columns from 10 to 9.
    
    # For rank 16, we need 16 columns. We have 7 Strassen + 9 direct = 16.
    # The 9 direct columns would be a subset of the original 10.
    
    # Actually, let's try a different structured pattern:
    # Allow the Strassen columns to also contribute to the k=2 block.
    # This means V columns 0-6 can have non-zero entries at k=2.
    # And we can reduce the direct product columns.
    
    # Let's set up the linear system for rank 16 with a specific support pattern.
    
    # Support pattern for rank 16:
    # Columns 0-6: Strassen columns, but V can have additional entries at k=2
    # Columns 7-15: 9 direct product columns (one of the original 10 is removed)
    
    # For each possible column to remove from the direct product columns,
    # and for each possible set of additional V entries in the Strassen columns,
    # solve the linear system.
    
    # This is still combinatorial. Let's use a more targeted approach.
    
    # Try: remove column 10 (A11*B12 -> C12) and allow V col 0 to have an entry at (1,2)
    # and V col 1 to have an entry at (1,2), etc.
    
    # Actually, let's just do a randomized search with the constraint that
    # entries are in {-1, 0, 1, 0.5, -0.5} and use the rank-17 solution as a starting point.
    
    # Randomized local search for rank 16
    def random_search_rank16(deadline):
        """Try to find a rank-16 solution via randomized local search"""
        np.random.seed(42)
        
        # Start from rank 17, remove a random column, and try to fix
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            col_to_remove = np.random.randint(17)
            
            # Create rank-16 matrices
            U16 = np.delete(U17, col_to_remove, axis=1)
            V16 = np.delete(V17, col_to_remove, axis=1)
            W16 = np.delete(W17, col_to_remove, axis=1)
            
            # Try to add non-zero entries to compensate
            # Focus on the k=2 block (V entries at j=0,1 and k=2)
            # and the j=2 block (U entries at i=0,1 and j=2)
            
            # Add a few random non-zero entries
            for _ in range(5):
                # Add to U at j=2
                if np.random.random() < 0.3:
                    i = np.random.randint(n)
                    r = np.random.randint(16)
                    U16[i*m+2, r] = np.random.choice([-1, -0.5, 0.5, 1])
                
                # Add to V at k=2
                if np.random.random() < 0.3:
                    j = np.random.randint(2)  # j=0 or 1
                    r = np.random.randint(16)
                    V16[j*p+2, r] = np.random.choice([-1, -0.5, 0.5, 1])
                
                # Add to W at k=2
                if np.random.random() < 0.3:
                    i = np.random.randint(n)
                    r = np.random.randint(16)
                    W16[2*n+i, r] = np.random.choice([-1, -0.5, 0.5, 1])
            
            # Snap to half-integers
            U16_s, V16_s, W16_s = snap_to_half_int(U16, V16, W16)
            
            # Verify
            if verify(U16_s, V16_s, W16_s):
                return U16_s, V16_s, W16_s
        
        return None
    
    # Try randomized search
    result = random_search_rank16(deadline - 10)
    if result is not None:
        U_s, V_s, W_s = result
        best_U, best_V, best_W = U_s, V_s, W_s
        best_rank = 16
    
    # If we didn't find rank 16, return rank 17
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
