        def _verify_decomposition(U, V, W, n=3, m=5, p=5):
            U = np.array(U, dtype=float)
            V = np.array(V, dtype=float)
            W = np.array(W, dtype=float)
            rank = U.shape[1]
            
            # Build target tensor T of shape (n*m, m*p, p*n)
            T = np.zeros((n * m, m * p, p * n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        I = i * m + j
                        J = j * p + k
                        K = k * n + i
                        T[I, J, K] = 1.0
            
            # Compute decomposition
            # T[I, J, K] = sum_r U[I, r] V[J, r] W[K, r]
            # Reshape for einsum?
            # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
            # We want result[I, J, K] = sum_r U[I, r] V[J, r] W[K, r]
            # This is a 3-way contraction.
            # Can do: temp[I, J] = sum_r U[I, r] V[J, r] -> (n*m, m*p)
            # Then result[I, J, K] = sum_J' ? No.
            # result[I, J, K] = sum_r U[I, r] V[J, r] W[K, r]
            # This is not a standard matrix multiply chain directly on the flattened indices without reshaping to 3D.
            # U3: (n, m, rank)
            # V3: (m, p, rank)
            # W3: (p, n, rank)
            # T[i, j, k] = sum_r U3[i, j, r] V3[j, k, r] W3[k, i, r]
            
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            
            # einsum: 'ijr,jkr,kir->ijk'
            result_3d = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            
            # Flatten result_3d to match T's shape
            # T is indexed by I=i*m+j, J=j*p+k, K=k*n+i
            # result_3d is indexed by i, j, k
            # We need to check if result_3d[i, j, k] == T[i*m+j, j*p+k, k*n+i]
            
            # Construct a flattened version of result_3d to compare with T
            # Or just loop
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        I = i * m + j
                        J = j * p + k
                        K = k * n + i
                        if not np.isclose(result_3d[i, j, k], T[I, J, K], atol=1e-10):
                            return False
            return True
        