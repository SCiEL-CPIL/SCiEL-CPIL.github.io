# Replace the 18-term loop with a single term
add_term(
    {0*m+2: 1, 1*m+2: 1},  # A[0,2] and A[1,2]
    {2*p+k for k in range(p)},  # B[2,k] for all k
    {k*n+i for k in range(p) for i in range(n)}  # C[i,k] for all i,k
)
