"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time

def _strassen_2x2x2():
    """Strassen 2x2x2, rank 7. U:(4,7), V:(4,7), W:(4,7)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1=(a+d)(e+h)
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[1][0]=1
    W[0][0]=1; W[3][0]=1
    # M2=(c+d)e
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[1][1]=1
    # M3=a(f-h)
    U[0][2]=1
    V[1][2]=1; V[3][2]=-1
    W[2][2]=1; W[3][2]=1
    # M4=d(g-e)
    U[3][3]=1
    V[2][3]=1; V[0][3]=-1
    W[0][3]=1; W[1][3]=1
    # M5=(a+b)h
    U[0][4]=1; U[1][4]=1
    V[3][4]=1
    W[0][4]=-1; W[2][4]=1
    # M6=(c-a)(e+f)
    U[2][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1
    W[3][5]=1
    # M7=(b-d)(g+h)
    U[1][6]=1; U[3][6]=-1
    V[2][6]=1; V[3][6]=1
    W[0][6]=1
    return U, V, W

def _schoolbook_2x2x1():
    """Schoolbook 2x2x1, rank 4. U:(4,4), V:(2,4), W:(2,4)."""
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    for j in range(2):
        for i in range(2):
            U[i*2+j][j] = 1
        V[j][j] = 1
        for i in range(2):
            W[i][j] = 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Build 2x2x7 (rank 25) = 2x2x4 (rank 14) + 2x2x3 (rank 11)
    # 2x2x4 = 2x Strassen 2x2x2
    # 2x2x3 = Strassen 2x2x2 + schoolbook 2x2x1
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # 2x2x4: k={0,1} and k={2,3}
    for k_start in [0, 2]:
        Us, Vs, Ws = _strassen_2x2x2()
        Vrem = [[0]*7 for _ in range(4)]
        for j in range(2):
            for k in range(2):
                Vrem[j*2 + k_start + k] = Vs[j*2 + k]
        Wrem = [[0]*7 for _ in range(4)]
        for k in range(2):
            for i in range(2):
                Wrem[(k_start + k)*2 + i] = Ws[k*2 + i]
        U_parts.append(Us)
        V_parts.append(Vrem)
        W_parts.append(Wrem)
    
    # 2x2x3: k={4,5} Strassen + k={6} schoolbook
    Us, Vs, Ws = _strassen_2x2x2()
    Vrem = [[0]*7 for _ in range(4)]
    for j in range(2):
        for k in range(2):
            Vrem[j*2 + 4 + k] = Vs[j*2 + k]
    Wrem = [[0]*7 for _ in range(4)]
    for k in range(2):
        for i in range(2):
            Wrem[(4 + k)*2 + i] = Ws[k*2 + i]
    U_parts.append(Us)
    V_parts.append(Vrem)
    W_parts.append(Wrem)
    
    # Schoolbook 2x2x1 for k=6
    Usb, Vsb, Wsb = _schoolbook_2x2x1()
    U_parts.append(Usb)
    Vexp = [[0]*4 for _ in range(14)]
    for j in range(2):
        for c in range(4):
            Vexp[j*2 + 6][c] = Vsb[j][c]
    V_parts.append(Vexp)
    Wexp = [[0]*4 for _ in range(14)]
    for i in range(2):
        for c in range(4):
            Wexp[6*2 + i][c] = Wsb[i][c]
    W_parts.append(Wexp)
    
    # Combine to 2x2x7
    rank_227 = sum(len(vp) for vp in V_parts)  # 7+7+7+4 = 25
    U227 = [[0]*rank_227 for _ in range(4)]
    V227 = [[0]*rank_227 for _ in range(14)]
    W227 = [[0]*rank_227 for _ in range(14)]
    off = 0
    for Ub, Vb, Wb in zip(U_parts, V_parts, W_parts):
        rb = len(Vb)
        for i in range(4):
            for r in range(rb):
                U227[i][off+r] = Ub[i][r]
        for j in range(14):
            for r in range(rb):
                V227[j][off+r] = Vb[j][r]
        for k in range(14):
            for r in range(rb):
                W227[k][off+r] = Wb[k][r]
        off += rb
    
    # Build 4x4x7 from 4 blocks of 2x2x7
    U447_parts = []
    V447_parts = []
    W447_parts = []
    
    for ib in range(2):
        for jb in range(2):
            Ub = [[0]*25 for _ in range(16)]
            for i in range(2):
                for j in range(2):
                    for r in range(25):
                        Ub[(ib*2+i)*4 + (jb*2+j)][r] = U227[i*2+j][r]
            Vb = [[0]*25 for _ in range(28)]
            for j in range(2):
                for k in range(7):
                    for r in range(25):
                        Vb[(jb*2+j)*7 + k][r] = V227[j*2+k][r]
            Wb = [[0]*25 for _ in range(28)]
            for k in range(7):
                for i in range(2):
                    for r in range(25):
                        Wb[k*4 + (ib*2+i)][r] = W227[k*2+i][r]
            U447_parts.append(Ub)
            V447_parts.append(Vb)
            W447_parts.append(Wb)
    
    total_rank = 100
    U_final = [[0]*total_rank for _ in range(16)]
    V_final = [[0]*total_rank for _ in range(28)]
    W_final = [[0]*total_rank for _ in range(28)]
    off = 0
    for Ub, Vb, Wb in zip(U447_parts, V447_parts, W447_parts):
        for i in range(16):
            for r in range(25):
                U_final[i][off+r] = Ub[i][r]
        for j in range(28):
            for r in range(25):
                V_final[j][off+r] = Vb[j][r]
        for k in range(28):
            for r in range(25):
                W_final[k][off+r] = Wb[k][r]
        off += 25
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
