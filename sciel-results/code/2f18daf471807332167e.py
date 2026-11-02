import time
import numpy as np

_DIMS = (2, 2, 7)
_T0 = time.time()
_LIMIT = _T0 + 120.0

def _left():
    return _LIMIT - time.time()

def construct():
    n, m, p = _DIMS
    R = 25
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))

    def strassen_block(cs, ks):
        # M1 = (A00+A11)*(B0k + B1,k+1)
        U[0, cs+0] += 1; U[3, cs+0] += 1
        V[ks, cs+0] += 1; V[7+ks+1, cs+0] += 1
        W[2*ks, cs+0] += 1; W[2*(ks+1)+1, cs+0] += 1

        # M2 = (A10+A11)*B0k
        U[2, cs+1] += 1; U[3, cs+1] += 1
        V[ks, cs+1] += 1
        W[2*ks+1, cs+1] += 1; W[2*(ks+1)+1, cs+1] -= 1

        # M3 = A00*(B0,k+1 - B1,k+1)
        U[0, cs+2] += 1
        V[7+ks+1, cs+2] += 1; V[14+ks+1, cs+2] -= 1
        W[2*(ks+1), cs+2] += 1; W[2*(ks+1)+1, cs+2] += 1

        # M4 = A11*(B1k - B0k)
        U[3, cs+3] += 1
        V[7+ks, cs+3] += 1; V[ks, cs+3] -= 1
        W[2*ks, cs+3] += 1; W[2*ks+1, cs+3] += 1

        # M5 = (A00+A01)*B1,k+1
        U[0, cs+4] += 1; U[1, cs+4] += 1
        V[14+ks+1, cs+4] += 1
        W[2*ks, cs+4] -= 1; W[2*(ks+1), cs+4] += 1

        # M6 = (A10-A00)*(B0k + B0,k+1)
        U[2, cs+5] += 1; U[0, cs+5] -= 1
        V[ks, cs+5] += 1; V[7+ks+1, cs+5] += 1
        W[2*(ks+1)+1, cs+5] += 1

        # M7 = (A01-A11)*(B1k + B1,k+1)
        U[1, cs+6] += 1; U[3, cs+6] -= 1
        V[7+ks, cs+6] += 1; V[14+ks+1, cs+6] += 1
        W[2*ks, cs+6] += 1

    strassen_block(0, 0)
    strassen_block(7, 2)
    strassen_block(14, 4)

    cs = 21
    U[0, cs+0] += 1; V[42, cs+0] += 1; W[12, cs+0] += 1
    U[1, cs+1] += 1; V[49, cs+1] += 1; W[12, cs+1] += 1
    U[2, cs+2] += 1; V[42, cs+2] += 1; W[13, cs+2] += 1
    U[3, cs+3] += 1; V[49, cs+3] += 1; W[13, cs+3] += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook():
    n, m, p = _DIMS
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, t] = 1.0
                V[j*p+k, t] = 1.0
                W[k*n+i, t] = 1.0
                t += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    r = construct()
    U = np.array(r["U"]); V = np.array(r["V"]); W = np.array(r["W"])
    A = np.random.rand(2, 2); B = np.random.rand(2, 7)
    Ce = A @ B
    Cw = np.zeros(14)
    for c in range(U.shape[1]):
        sa = float(np.dot(U[:, c], A.ravel()))
        sb = float(np.dot(V[:, c], B.ravel()))
        Cw += W[:, c] * sa * sb
    Cm = np.zeros((2, 7))
    for k in range(7):
        for i in range(2):
            Cm[i, k] = Cw[2*k+i]
    print("OK" if np.allclose(Cm, Ce) else "FAIL")
