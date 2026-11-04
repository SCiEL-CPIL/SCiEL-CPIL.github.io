<<<<<<< SEARCH
    # C11 = A11*B11 + A12*B21
    # A11*B11: 2x2 * 2x2, use Strassen
    strassen_2x2(0, 0, 0)  # A base 0, B base 0, C base 0
    # A12*B21: 2x2 * 2x2, use Strassen
    strassen_2x2(2, 8, 0)  # A base 2, B base 8, C base 0
    
    # C12 = A11*B12 + A12*B22
    # A11*B12: 2x2 * 2x2, use Strassen
    strassen_2x2(0, 2, 6)  # A base 0, B base 2, C base 6
    # A12*B22: 2x2 * 2x2, use Strassen
    strassen_2x2(2, 10, 6)  # A base 2, B base 10, C base 6
    
    # C21 = A21*B11 + A22*B21
    # A21*B11: 1x2 * 2x2
    simple_1x2_x_2x2(8, 0, 2)  # A at 8, B base 0, C base 2
    # A22*B21: 1x2 * 2x2
    simple_1x2_x_2x2(10, 8, 2)  # A at 10, B base 8, C base 2
    
    # C22 = A21*B12 + A22*B22
    # A21*B12: 1x2 * 2x2
    simple_1x2_x_2x2(8, 2, 8)  # A at 8, B base 2, C base 8
    # A22*B22: 1x2 * 2x2
    simple_1x2_x_2x2(10, 10, 8)  # A at 10, B base 10, C base 8
=======
    # C11 = A11*B11 + A12*B21
    # A11: i0=0,j0=0; B11: j0=0,k0=0; C11: k0=0,i0=0
    strassen_2x2(0, 0, 0, 0, 0, 0)
    # A12: i0=0,j0=2; B21: j0=2,k0=0; C11: k0=0,i0=0
    strassen_2x2(0, 2, 2, 0, 0, 0)
    
    # C12 = A11*B12 + A12*B22
    # A11: i0=0,j0=0; B12: j0=0,k0=2; C12: k0=2,i0=0
    strassen_2x2(0, 0, 0, 2, 2, 0)
    # A12: i0=0,j0=2; B22: j0=2,k0=2; C12: k0=2,i0=0
    strassen_2x2(0, 2, 2, 2, 2, 0)
    
    # C21 = A21*B11 + A22*B21
    # A21: row i=2, cols j=0,1; B11: j0=0,k0=0; C21: k0=0,1; i=2
    simple_1x2_x_2x2(8, 0, 2)
    # A22: row i=2, cols j=2,3; B21: j0=2,k0=0; C21: k0=0,1; i=2
    simple_1x2_x_2x2(10, 8, 2)
    
    # C22 = A21*B12 + A22*B22
    # A21: row i=2, cols j=0,1; B12: j0=0,k0=2; C22: k0=2,3; i=2
    simple_1x2_x_2x2(8, 2, 8)
    # A22: row i=2, cols j=2,3; B22: j0=2,k0=2; C22: k0=2,3; i=2
    simple_1x2_x_2x2(10, 10, 8)
>>>>>>> REPLACE

Now let me also fix the `simple_1x2_x_2x2` function to use proper indexing:

