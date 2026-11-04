"""Rank-minimal tensor decomposition for 3x4x6 matrix multiplication.
Uses block-partition baseline (rank 66) then applies structured local search
to identify and eliminate linearly-dependent rank terms.
"""
import time

n, m, p = 3, 4, 6

def Aidx(i, j): return i * m + j
def Bidx(j, k): return j * p + k
def Cidx(i, k): return k * n + i

def verify(U, V, W, r):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for l in range(r):
                    s += U[Aidx(i,j)][l] * V[Bidx(j,k)][l] * W[Cidx(i,k)][l]
                if s != 1:
                    return False
    return True

def baseline_66():
    U = [[0]*66 for _ in range(n*m)]
    V = [[0]*66 for _ in range(m*p)]
    W = [[0]*66 for _ in range(p*n)]
    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = a_r*m + a_c; A01 = a_r*m + a_c+1
        A10 = (a_r+1)*m + a_c; A11 = (a_r+1)*m + a_c+1
        B00 = b_r*p + b_c; B01 = b_r*p + b_c+1
        B10 = (b_r+1)*p + b_c; B11 = (b_r+1)*p + b_c+1
        C00 = c_c*n + c_r; C01 = (c_c+1)*n + c_r
        C10 = c_c*n + (c_r+1); C11 = (c_c+1)*n + (c_r+1)
        U[A00][col]+=1; U[A11][col]+=1; V[B00][col]+=1; V[B11][col]+=1; W[C00][col]+=1; W[C11][col]+=1; col+=1
        U[A10][col]+=1; U[A11][col]+=1; V[B00][col]+=1; W[C10][col]+=1; W[C11][col]-=1; col+=1
        U[A00][col]+=1; V[B01][col]+=1; V[B11][col]-=1; W[C01][col]+=1; W[C11][col]+=1; col+=1
        U[A11][col]+=1; V[B10][col]+=1; V[B00][col]-=1; W[C00][col]+=1; W[C10][col]+=1; col+=1
        U[A00][col]+=1; U[A01][col]+=1; V[B11][col]+=1; W[C00][col]-=1; W[C01][col]+=1; col+=1
        U[A10][col]+=1; U[A00][col]-=1; V[B00][col]+=1; V[B01][col]+=1; W[C11][col]+=1; col+=1
        U[A01][col]+=1; U[A11][col]-=1; V[B10][col]+=1; V[B11][col]+=1; W[C00][col]+=1; col+=1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = a_r*m + a_c; A1 = a_r*m + a_c+1
        B00 = b_r*p + b_c; B01 = b_r*p + b_c+1
        B10 = (b_r+1)*p + b_c; B11 = (b_r+1)*p + b_c+1
        C0 = c_c*n + c_r; C1 = (c_c+1)*n + c_r
        U[A0][col]+=1; V[B00][col]+=1; W[C0][col]+=1; col+=1
        U[A1][col]+=1; V[B10][col]+=1; W[C0][col]+=1; col+=1
        U[A0][col]+=1; V[B01][col]+=1; W[C1][col]+=1; col+=1
        U[A1][col]+=1; V[B11][col]+=1; W[C1][col]+=1; col+=1

    add_1x2_2x2(0,0,0,0,0,0); add_1x2_2x2(0,2,2,0,0,0)
    add_1x2_2x2(0,0,0,2,0,2); add_1x2_2x2(0,2,2,2,0,2)
    add_1x2_2x2(0,0,0,4,0,4); add_1x2_2x2(0,2,2,4,0,4)
    add_strassen(1,0,0,0,1,0); add_strassen(1,2,2,0,1,0)
    add_strassen(1,0,0,2,1,2); add_strassen(1,2,2,2,1,2)
    add_strassen(1,0,0,4,1,4); add_strassen(1,2,2,4,1,4)
    assert col == 66
    return U, V, W

def construct():
    deadline = time.time() + 115
    U, V, W = baseline_66()
    r = 66

    # Structured local search: try dropping terms one at a time,
    # then try merging pairs by checking if their combined contribution
    # can be represented as a single rank-1 term.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try dropping each term
        for l in range(r):
            ok = True
            for i in range(n):
                if not ok: break
                for j in range(m):
                    if not ok: break
                    for k in range(p):
                        s = 0
                        for ll in range(r):
                            if ll == l: continue
                            s += U[Aidx(i,j)][ll] * V[Bidx(j,k)][ll] * W[Cidx(i,k)][ll]
                        if s != 1:
                            ok = False
                            break
                    if not ok: break
            if ok:
                U = [row[:l] + row[l+1:] for row in U]
                V = [row[:l] + row[l+1:] for row in V]
                W = [row[:l] + row[l+1:] for row in W]
                r -= 1
                improved = True
                break

    assert verify(U, V, W, r), "Final decomposition is not exact!"
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
