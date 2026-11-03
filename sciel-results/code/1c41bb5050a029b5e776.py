import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # 1. Strassen 2x2 Decomposition (Rank 7)
    def strassen_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3 = A00(B10-B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[0, 2] = 1; W[1, 2] = 1
        
        # M4 = (A00-A11)(B01+B11)
        U[0, 3] = 1; U[3, 3] = -1
        V[1, 3] = 1; V[3, 3] = 1
        W[0, 3] = 1; W[3, 3] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1; W[3, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1; W[0, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1; W[2, 6] = 1
        
        return U, V, W

    # 2. Construct 4x4x4 Decomposition (Rank 49)
    def strassen_4x4():
        U2, V2, W2 = strassen_2x2()
        rank = 49
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        
        for t1 in range(7):
            for t2 in range(7):
                col = t1 * 7 + t2
                # Kronecker product logic
                # U_row = i1*2 + i2
                for i1 in range(2):
                    for i2 in range(2):
                        row = i1 * 2 + i2
                        U[row, col] = U2[i1, t1] * U2[i2, t2]
                        
                        V[row, col] = V2[i1, t1] * V2[i2, t2]
                        
                        W[row, col] = W2[i1, t1] * W2[i2, t2]
        return U, V, W

    # 3. Construct 4x4x1 Decomposition (Rank 4)
    def rect_4x4x1():
        rank = 4
        U = np.zeros((16, rank))
        V = np.zeros((20, rank))
        W = np.zeros((20, rank))
        
        for col in range(4):
            # Term col corresponds to j=col
            # A_{i, col} * B_{col, 0} -> C_{i, 0}
            for i in range(4):
                # U index: i*m + col = i*4 + col
                U[i * 4 + col, col] = 1.0
                # W index: k*n + i = 0*4 + i = i
                W[i, col] = 1.0
            
            # V index: j*p + k = col*5 + 0 = col*5
            V[col * 5, col] = 1.0
            
        return U, V, W

    # 4. Combine to 4x4x5 (Rank 53)
    U44, V44, W44 = strassen_4x4()
    U41, V41, W41 = rect_4x4x1()
    
    rank_total = 53
    U = np.zeros((16, rank_total))
    V = np.zeros((20, rank_total))
    W = np.zeros((20, rank_total))
    
    U[:, :49] = U44
    U[:, 49:] = U41
    
    V[:16, :49] = V44
    V[16:20, :49] = 0
    V[:, 49:] = V41
    
    W[:16, :49] = W44
    W[16:20, :49] = 0
    W[:, 49:] = W41

    # 5. Verification
    def verify_decomp(U, V, W, n, m, p):
        # Check all on-support components (should be 1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check off-support components (should be 0)
        # Tensor index: a = i*m + j, b = j'*p + k, c = k'*n + i'
        # Non-zero only if j == j', i == i', k == k'
        for a in range(n * m):
            i_a = a // m
            j_a = a % m
            for b in range(m * p):
                j_b = b // p
                k_b = b % p
                if j_a != j_b:
                    # Off-support: must be 0
                    for c in range(p * n):
                        k_c = c // n
                        i_c = c % n
                        val = np.dot(U[a], V[b] * W[c])
                        if abs(val) > 1e-9:
                            return False
                else:
                    # j_a == j_b, check i and k alignment
                    for c in range(p * n):
                        k_c = c // n
                        i_c = c % n
                        if i_a != i_c or k_b != k_c:
                            val = np.dot(U[a], V[b] * W[c])
                            if abs(val) > 1e-9:
                                return False
        return True

    if not verify_decomp(U, V, W, n, m, p):
        # Fallback to Schoolbook if Strassen construction fails (should not happen)
        rank = 80
        U = np.zeros((16, rank))
        V = np.zeros((20, rank))
        W = np.zeros((20, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
    else:
        rank = 53

    # Optional: Try to reduce rank further using time
    if time.time() < deadline - 10:
        # Prune linearly dependent columns
        # Build the full tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Then try to find a smaller subset of columns that spans the same tensor
        
        # First, build the full 3D tensor
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        # Try greedy pruning: remove columns one by one if the tensor is preserved
        # We'll work with a list of active column indices
        active = list(range(rank))
        
        # Try removing columns in order of smallest contribution
        # Compute "importance" of each column as the norm of its contribution
        importances = []
        for r in range(rank):
            contrib = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            importances.append((np.linalg.norm(contrib), r))
        
        importances.sort(key=lambda x: x[0])
        
        # Try removing columns from least important to most
        for _, r_to_remove in importances:
            if time.time() > deadline - 15:
                break
            if r_to_remove not in active:
                continue
            
            # Try removing this column
            new_active = [x for x in active if x != r_to_remove]
            if len(new_active) < len(active):
                # Check if the tensor is preserved
                T_reduced = np.zeros((n*m, m*p, p*n))
                for r in new_active:
                    T_reduced += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                
                if np.allclose(T, T_reduced, atol=1e-9):
                    active = new_active
        
        # Update U, V, W with only active columns
        if len(active) < rank:
            U = U[:, active]
            V = V[:, active]
            W = W[:, active]
            rank = len(active)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
