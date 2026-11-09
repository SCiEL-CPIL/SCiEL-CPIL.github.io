    def verify(U, V, W, n, m, p):
        # U: nm x rank
        # V: mp x rank
        # W: pn x rank
        rank = U.shape[1]
        # Check C = A B
        # C[i,k] = sum_j A[i,j] B[j,k]
        # Decomposition: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Note: The tensor definition is T[i,j,k] = 1.
        # The decomposition is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = T[i,j,k]
        
        # Let's test with random A, B
        for _ in range(10):
            A = np.random.randint(-5, 5, (n, m))
            B = np.random.randint(-5, 5, (m, p))
            C = A @ B
            
            # Compute C from decomposition
            C_decomp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    val = 0
                    for j in range(m):
                        # Sum over r
                        # U index: i*m + j
                        # V index: j*p + k
                        # W index: k*n + i
                        u_vec = U[i*m + j, :]
                        v_vec = V[j*p + k, :]
                        w_vec = W[k*n + i, :]
                        # The term is sum_r U V W
                        val += np.dot(u_vec, v_vec * w_vec) # No, it's sum_r U_r V_r W_r
                        # Actually, C[i,k] = sum_j ( sum_r U[i,j,r] V[j,k,r] ) * W[k,i,r]?
                        # No.
                        # The bilinear form is sum_r U[i,j,r] V[j,k,r] W[k,i,r].
                        # This is not a standard matrix product.
                        # It is sum_r (U[i,j,r] * V[j,k,r]) * W[k,i,r].
                        # This is not separable in the way A @ B is.
                        
                        # Wait, the tensor decomposition is:
                        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                        # The matrix multiplication is:
                        # C[i,k] = sum_j A[i,j] B[j,k]
                        # So we need:
                        # sum_j A[i,j] B[j,k] = sum_r ( sum_j A[i,j] U[i,j,r] ) * ( sum_j B[j,k] V[j,k,r] ) * W[k,i,r] ?
                        # No.
                        # The standard definition is:
                        # C[i,k] = sum_r U[i,j,r] ...
                        
                        # Let's clarify the interface.
                        # U is (n*m, rank). Index (i*m+j, r).
                        # V is (m*p, rank). Index (j*p+k, r).
                        # W is (p*n, rank). Index (k*n+i, r).
                        
                        # The decomposition of the tensor T[i,j,k] is:
                        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        
                        # The matrix multiplication is:
                        # C[i,k] = sum_j A[i,j] B[j,k]
                        
                        # If we substitute the decomposition:
                        # sum_j A[i,j] B[j,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                        # This must hold for all A, B.
                        # This implies that the tensor T is decomposed as sum_r U_r \otimes V_r \otimes W_r.
                        
                        # So to verify:
                        # For each r, we have a term.
                        # We can't just sum over j inside the r sum easily.
                        
                        # Better verification:
                        # Check if T[i,j,k] == sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                        # for all i, j, k.
                        # T[i,j,k] is 1 if the indices are valid (which they always are in the tensor).
                        # So we just need to check if the sum equals 1 for all i,j,k.
                        
                        pass
                    
            # This verification is O(n*m*p*rank).
            # 3*3*8*64 = 4608 operations. Very fast.
    