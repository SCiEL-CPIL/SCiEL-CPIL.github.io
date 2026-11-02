    # The product sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should reconstruct the multiplication.
    #
    # For a fixed (i,k), C[i,k] = sum_j A[i,j] B[j,k].
    #
    # In the decomposition: C[i,k] = sum_c (sum_j A[i,j] U[i*m+j, c]) * (sum_j' B[j',k] V[j'*p+k, c]) * W[k*n+i, c]
    #
    # For the schoolbook, when c corresponds to (i0,j0,k0):
    # U[i*m+j, c] = 1 if i=i0 and j=j0, else 0
    # V[j*p+k, c] = 1 if j=j0 and k=k0, else 0
    # W[k*n+i, c] = 1 if k=k0 and i=i0, else 0
    #
    # So the term is: W[k*n+i, c] * A[i0,j0] * B[j0,k0] if i=i0 and k=k0 and j=j0... 
    # Wait, let me be more careful.
