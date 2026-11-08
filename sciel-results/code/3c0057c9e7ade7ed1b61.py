    # Terms for k=8 (2x2x1, rank 4)
    k8 = 8
    # C00 = a00*b08
    add_term([(0*m+0, 1)], [(0*p+k8, 1)], [(k8*n+0, 1)], c)
    c += 1
    # C01 = a01*b18 ? No. C_{i,k} = sum_j A_{ij}B_{jk}.
    # ... comments ...
    # Correcting k=8 terms:
    # Term 1: a00 * b08
    add_term([(0*m+0, 1)], [(0*p+k8, 1)], [(k8*n+0, 1)], c)
    c += 1
   