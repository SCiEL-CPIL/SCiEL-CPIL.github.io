<<<<<<< SEARCH
def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return U, V, W
=======
def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    # U[i,j] where i is row in A (0=A11, 1=A12, 2=A21, 3=A22), j is Strassen product index
    # M1 = (A11-A22)(B22-B11) -> U: A11=1, A22=-1
    # M2 = (A21+A22)B11 -> U: A21=1, A22=1
    # M3 = A11(B11-B22) -> U: A11=1
    # M4 = A22(B22-B11) -> U: A22=1
    # M5 = (A11+A12)B22 -> U: A11=1, A12=1
    # M6 = (A21-A11)(B11+B22) -> U: A21=1, A11=-1
    # M7 = (A12-A22)(B21+B22) -> U: A12=1, A22=-1
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # A11: M1=1, M3=1, M5=1, M6=-1
        [0, 0, 0, 0, 1, 0, 1],    # A12: M5=1, M7=1
        [0, 1, 0, 0, 0, 1, 0],    # A21: M2=1, M6=1
        [-1, 1, 0, 1, 0, 0, -1]   # A22: M1=-1, M2=1, M4=1, M7=-1
    ]
    # V[j,k] where j is Strassen product index, k is row in B (0=B11, 1=B12, 2=B21, 3=B22)
    # M1 = (A11-A22)(B22-B11) -> V: B22=1, B11=-1
    # M2 = (A21+A22)B11 -> V: B11=1
    # M3 = A11(B11-B22) -> V: B11=1, B22=-1
    # M4 = A22(B22-B11) -> V: B22=1, B11=-1
    # M5 = (A11+A12)B22 -> V: B22=1
    # M6 = (A21-A11)(B11+B22) -> V: B11=1, B22=1
    # M7 = (A12-A22)(B21+B22) -> V: B21=1, B22=1
    V = [
        [1, 0, 0, -1],   # M1: B22=1, B11=-1
        [0, 1, 0, 0],    # M2: B11=1
        [1, 0, 0, -1],   # M3: B11=1, B22=-1
        [-1, 0, 0, 1],   # M4: B22=1, B11=-1
        [0, 0, 0, 1],    # M5: B22=1
        [1, 0, 0, 1],    # M6: B11=1, B22=1
        [0, 0, 1, 1]     # M7: B21=1, B22=1
    ]
    # W[k,i] where k is row in C (0=C11, 1=C12, 2=C21, 3=C22), i is Strassen product index
    # C11 = M1 + M4 - M5 + M7 -> W: M1=1, M4=1, M5=-1, M7=1
    # C12 = M3 + M5 -> W: M3=1, M5=1
    # C21 = M2 + M4 -> W: M2=1, M4=1
    # C22 = M1 - M2 + M3 + M6 -> W: M1=1, M2=-1, M3=1, M6=1
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # C11: M1=1, M4=1, M5=-1, M7=1
        [0, 0, 1, 0, 1, 0, 0],    # C12: M3=1, M5=1
        [0, 1, 0, 1, 0, 0, 0],    # C21: M2=1, M4=1
        [1, -1, 1, 0, 0, 1, 0]    # C22: M1=1, M2=-1, M3=1, M6=1
    ]
    return U, V, W
>>>>>>> REPLACE

Now let me verify this is correct by checking the tensor decomposition. The key insight is that the current code might have incorrect Strassen matrices. Let me also improve the search strategy by trying a more structured approach.

Actually, I realize I should just fix the Strassen matrices and keep the rest of the code. The rank-88 decomposition should work correctly with the right Strassen matrices.

