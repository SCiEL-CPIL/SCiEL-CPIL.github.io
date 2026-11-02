"""Find rank-minimal exact tensor decomposition for 4x4 matrix multiplication."""
from fractions import Fraction
import time

DIMENSIONS = [4, 4, 4]

def _strassen_forms():
    """
    Strassen's 7 bilinear forms for 2x2 matrix multiplication.
    Returns (forms, output_coeffs) where:
    - forms: list of 7 tuples (left_dict, right_dict)
    - output_coeffs: list of 4 dicts {form_idx: coeff} for C[00], C[01], C[10], C[11]
    """
    F = Fraction
    forms = [
        ({0: F(1), 3: F(1)}, {0: F(1), 3: F(1)}),           # M1=(a00+a11)(b00+b11)
        ({2: F(1), 3: F(1)}, {0: F(1)}),                     # M2=(a10+a11)b00
        ({0: F(1)}, {1: F(1), 3: F(-1)}),                    # M3=a00(b01-b11)
        ({3: F(1)}, {2: F(1), 0: F(-1)}),                    # M4=a11(b10-b00)
        ({0: F(1), 1: F(1)}, {3: F(1)}),                     # M5=(a00+a01)b11
        ({2: F(1), 0: F(-1)}, {0: F(1), 1: F(1)}),           # M6=(a10-a00)(b00+b01)
        ({1: F(1), 3: F(-1)}, {2: F(1), 3: F(1)}),           # M7=(a01-a11)(b10+b11)
    ]
    output = [
        {0: F(1), 3: F(1), 4: F(-1), 6: F(1)},  # C00 = M1+M4-M5+M7
        {2: F(1), 4: F(1)},                      # C01 = M3+M5
        {1: F(1), 3: F(1)},                      # C10 = M2+M4
        {0: F(1), 1: F(-1), 2: F(1), 5: F(1)},  # C11 = M1-M2+M3+M6
    ]
    return forms, output

def _build_rank49():
    """Build rank-49 decomposition via recursive Strassen."""
    F = Fraction
    outer_forms, outer_output = _strassen_forms()
    inner_forms, inner_output = _strassen_forms()
    
    rank = 49
    U = [[F(0)] * rank for _ in range(16)]
    V = [[F(0)] * rank for _ in range(16)]
    W = [[F(0)] * rank for _ in range(16)]
    
    for f in range(7):
        left_blocks, right_blocks = outer_forms[f]
        for g in range(7):
            c = f * 7 + g
            inner_left, inner_right = inner_forms[g]
            
            # U: coefficient of A[i,j] in left factor
            for i in range(4):
                for j in range(4):
                    bi, li = divmod(i, 2)
                    bj, lj = divmod(j, 2)
                    lb = bi * 2 + bj
                    if lb in left_blocks:
                        pos = li * 2 + lj
                        if pos in inner_left:
                            U[i*4+j][c] += left_blocks[lb] * inner_left[pos]
            
            # V: coefficient of B[j,k] in right factor
            for j in range(4):
                for k in range(4):
                    bj, lj = divmod(j, 2)
                    bk, lk = divmod(k, 2)
                    rb = bj * 2 + bk
                    if rb in right_blocks:
                        pos = lj * 2 + lk
                        if pos in inner_right:
                            V[j*4+k][c] += right_blocks[rb] * inner_right[pos]
            
            # W: coefficient of C[i,k] in output
            for ci in range(2):
                for cj in range(2):
                    out_pos = ci * 2 + cj
                    outer_coeff = outer_output[out_pos].get(f, F(0))
                    if outer_coeff == 0:
                        continue
                    for li in range(2):
                        for lk in range(2):
                            in_pos = li * 2 + lk
                            inner_coeff = inner_output[in_pos].get(g, F(0))
                            if inner_coeff == 0:
                                continue
                            i = 2*ci + li
                            k = 2*cj + lk
                            W[k*4+i][c] += outer_coeff * inner_coeff
    
    return U, V, W

def _verify(U, V, W, trials=10):
    """Verify decomposition correctness."""
    import random
    random.seed(42)
    for _ in range(trials):
        A = [[random.randint(-3,3) for _ in range(4)] for _ in range(4)]
        B = [[random.randint(-3,3) for _ in range(4)] for _ in range(4)]
        C = [[sum(A[i][j]*B[j][k] for j in range(4)) for k in range(4)] for i in range(4)]
        for i in range(4):
            for k in range(4):
                s = Fraction(0)
                for r in range(len(U[0])):
                    inner = Fraction(0)
                    for j in range(4):
                        inner += Fraction(U[i*4+j][r]) * Fraction(V[j*4+k][r])
                    s += inner * Fraction(W[k*4+i][r])
                if s != Fraction(C[i][k]):
                    return False
    return True

def construct():
    start = time.time()
    deadline = start + 135
    
    U, V, W = _build_rank49()
    
    if not _verify(U, V, W):
        # Fallback to schoolbook rank 64
        rank = 64
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        c = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j][c] = 1
                    V[j*4+k][c] = 1
                    W[k*4+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    # Try rank reduction via Gaussian elimination on the 3-tensor
    # This is complex; for now return rank 49
    U_out = [[int(x) for x in row] for row in U]
    V_out = [[int(x) for x in row] for row in V]
    W_out = [[int(x) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
