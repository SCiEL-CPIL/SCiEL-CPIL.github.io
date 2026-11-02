"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # Try to construct rank-49 decomposition
    try:
        # 2x2 Strassen base
        # U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
        U2 = [[Fraction(0)]*7 for _ in range(4)]
        V2 = [[Fraction(0)]*7 for _ in range(4)]
        W2 = [[Fraction(0)]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11) -> C00:+1, C11:+1
        U2[0][0]=1; U2[3][0]=1; V2[0][0]=1; V2[3][0]=1; W2[0][0]=1; W2[3][0]=1
        # M2 = (A10+A11)B00 -> C10:+1, C11:-1
        U2[2][1]=1; U2[3][1]=1; V2[0][1]=1; W2[2][1]=1; W2[3][1]=-1
        # M3 = A00(B10-B11) -> C00:-1, C10:+1
        U2[0][2]=1; V2[2][2]=1; V2[3][2]=-1; W2[0][2]=-1; W2[2][2]=1
        # M4 = A11(B01-B00) -> C01:+1, C11:+1
        U2[3][3]=1; V2[1][3]=1; V2[0][3]=-1; W2[1][3]=1; W2[3][3]=1
        # M5 = (A00+A01)B11 -> C00:-1, C01:+1
        U2[0][4]=1; U2[1][4]=1; V2[3][4]=1; W2[0][4]=-1; W2[1][4]=1
        # M6 = (A10-A00)(B00+B10) -> C11:-1
        U2[2][5]=1; U2[0][5]=-1; V2[0][5]=1; V2[2][5]=1; W2[3][5]=-1
        # M7 = (A01-A11)(B01+B11) -> C00:+1
        U2[1][6]=1; U2[3][6]=-1; V2[1][6]=1; V2[3][6]=1; W2[0][6]=1
        
        # Verify 2x2
        ok = True
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = sum(U2[i*2+j][r]*V2[j*2+k][r]*W2[k*2+i][r] for r in range(7))
                    if s != 1:
                        ok = False
                        break
                if not ok: break
            if not ok: break
        
        if not ok:
            raise ValueError("2x2 base failed")
        
        # Build 4x4 rank-49
        U4 = [[Fraction(0)]*49 for _ in range(16)]
        V4 = [[Fraction(0)]*49 for _ in range(16)]
        W4 = [[Fraction(0)]*49 for _ in range(16)]
        
        # Block coefficients
        bA = [
            {(0,0):1,(1,1):1},
            {(1,0):1,(1,1):1},
            {(0,0):1},
            {(1,1):1},
            {(0,0):1,(0,1):1},
            {(1,0):1,(0,0):-1},
            {(0,1):1,(1,1):-1},
        ]
        bB = [
            {(0,0):1,(1,1):1},
            {(0,0):1},
            {(1,0):1,(1,1):-1},
            {(0,1):1,(0,0):-1},
            {(1,1):1},
            {(0,0):1,(1,0):1},
            {(0,1):1,(1,1):1},
        ]
        bC = [
            {(0,0):1,(1,1):1,(0,1):-1,(1,0):-1},
            {(1,0):1,(1,1):-1},
            {(0,0):-1,(1,0):1},
            {(0,1):1,(1,1):1},
            {(0,0):-1,(0,1):1},
            {(1,1):-1},
            {(0,0):1},
        ]
        
        for t in range(7):
            for r2 in range(7):
                col = t*7+r2
                for (a,b),ca in bA[t].items():
                    for ip in range(2):
                        for jp in range(2):
                            uv = U2[ip*2+jp][r2]
                            if uv:
                                U4[(2*a+ip)*4+(2*b+jp)][col] += ca*uv
                for (c,d),cb in bB[t].items():
                    for jp in range(2):
                        for kp in range(2):
                            vv = V2[jp*2+kp][r2]
                            if vv:
                                V4[(2*c+jp)*4+(2*d+kp)][col] += cb*vv
                for (e,f),cc in bC[t].items():
                    for kp in range(2):
                        for ip in range(2):
                            wv = W2[kp*2+ip][r2]
                            if wv:
                                W4[(2*f+kp)*4+(2*e+ip)][col] += cc*wv
        
        # Verify 4x4
        ok = True
        for i in range(4):
            if time.time() > deadline: break
            for j in range(4):
                for k in range(4):
                    s = sum(U4[i*4+j][r]*V4[j*4+k][r]*W4[k*4+i][r] for r in range(49))
                    if s != 1:
                        ok = False
                        break
                if not ok: break
            if not ok: break
        
        if ok:
            return {
                "U": [[float(x) for x in row] for row in U4],
                "V": [[float(x) for x in row] for row in V4],
                "W": [[float(x) for x in row] for row in W4],
            }
        else:
            raise ValueError("4x4 verification failed")
    
    except Exception:
        # Fallback to schoolbook
        rank = 64
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j][col] = 1
                    V[j*4+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
