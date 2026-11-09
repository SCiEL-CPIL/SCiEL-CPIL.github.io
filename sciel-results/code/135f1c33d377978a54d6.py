<<<<<<< SEARCH
    def verify_decomposition_fast(U, V, W, tol=1e-9):
        """Faster verification using numpy."""
        r = len(U[0]) if U else 0
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # For each (i,j,k), check sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
        # This is equivalent to: for each i,k, sum_j (sum_r U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r]
        # Let's compute M[i,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r]
        # Then check if sum_r M[i,k] * W[k*n+i,r] == 1 for all i,k
        # But M depends on r, so this isn't quite right.
        
        # Alternative: compute the full tensor and check
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # We need T[i,j,k] = 1 for all i,j,k
        
        # Efficient computation:
        # For fixed i,k: sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] * V[j*p+k,r])
        # Let A_ik[r] = sum_j U[i*m+j,r] * V[j*p+k,r]
        # Then T_ik = sum_r A_ik[r] * W[k*n+i,r]
        
        # Compute for all i,k:
        # A_ik[r] = sum_j U[i*m+j, r] * V[j*p+k, r]
        # This is like a batched matrix product
        
        # Reshape U to (n, m, r), V to (m, p, r)
        U3 = U_arr.reshape(n, m, r)
        V3 = V_arr.reshape(m, p, r)
        W3 = W_arr.reshape(p, n, r)
        
        # A_ik[r] = sum_j U3[i,j,r] * V3[j,k,r]
        # This is element-wise product summed over j
        # A[i,k,r] = sum_j U3[i,j,r] * V3[j,k,r]
        
        # Use einsum: A[i,k,r] = sum_j U3[i,j,r] * V3[j,k,r]
        A = np.einsum('ijr,jkr->ikr', U3, V3)
        
        # T[i,k] = sum_r A[i,k,r] * W3[k,i,r]
        # Note: W3[k,i,r] = W_arr[k*n+i, r]
        T = np.einsum('ikr,ikir->ik', A, W3)
        
        # Check if T[i,k] == 1 for all i,k
        return np.allclose(T, 1.0, atol=tol)
=======
    def verify_decomposition_fast(U, V, W, tol=1e-9):
        """Faster verification using numpy."""
        r = len(U[0]) if U else 0
        if r == 0:
            return False
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U3 = U_arr.reshape(n, m, r)
        V3 = V_arr.reshape(m, p, r)
        W3 = W_arr.reshape(p, n, r)
        
        # A[i,k,r] = sum_j U3[i,j,r] * V3[j,k,r]
        A = np.einsum('ijr,jkr->ikr', U3, V3)
        
        # T[i,k] = sum_r A[i,k,r] * W3[k,i,r]
        # W3 is (p, n, r), so W3[k, i, r] is the element.
        T = np.einsum('ikr,kir->ik', A, W3)
        
        # Check if T[i,k] == 1 for all i,k
        return np.allclose(T, 1.0, atol=tol)
>>>>>>> REPLACE

