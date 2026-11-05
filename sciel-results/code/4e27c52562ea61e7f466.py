import numpy as np
import time
import itertools

START_TIME = time.time()
TIME_LIMIT = 110  # Safety margin for 150s limit


def _time_exceeded():
    return (time.time() - START_TIME) > TIME_LIMIT


def _verify(U, V, W, n=3, m=5, p=7):
    """Verify tensor identity: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i]))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def construct():
    n, m, p = 3, 5, 7

    # ---------- Block templates ----------
    # Each template: (a,b,c) -> (rank, U(a*b x rank), V(b*c x rank), W(c*a x rank))
    # U indexed by (i,j), V by (j,k), W by (k,i).

    def schoolbook_block(a, b, c):
        rank = a * b * c
        U = [[0]*rank for _ in range(a*b)]
        V = [[0]*rank for _ in range(b*c)]
        W = [[0]*rank for _ in range(c*a)]
        idx = 0
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    U[i*b+j][idx] = 1
                    V[j*c+k][idx] = 1
                    W[k*a+i][idx] = 1
                    idx += 1
        return rank, U, V, W

    def strassen_222():
        # 2x2x2 Strassen rank 7. U,V,W are 4x7.
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # M1: (A00+A11)(B00+B11)
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        # M2: (A10+A11)B00
        U[2][1]=1; U[3][1]=1; V[0][1]=1; W[1][1]=1; W[3][1]=-1
        # M3: A00(B10-B11)
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[2][2]=1; W[3][2]=1
        # M4: A11(B01-B00)
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[1][3]=1
        # M5: (A00+A01)B11
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[2][4]=1
        # M6: (A10-A00)(B00+B01)
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        # M7: (A01-A11)(B10+B11)
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
        return 7, U, V, W

    # 2x2x3 block rank 8: two Strassen 2x2x2 blocks, second shifted by one column in C.
    # C has 3 columns; columns 0,1 use first Strassen (rank7), column 2 handled by 1 extra mult.
    # We treat C as 2x3: split C into col-pair (0,1) and col (2).
    # Block A (2x2), B (2x3): B columns (0,1) form B12 (2x2), column 2 is B3 (2x1).
    # C01 = A * B12 (Strassen 7), C2 = A * B3 (schoolbook 2 mults) -> total 9? 
    # Better: C01 via Strassen (7), C2 via 2 mults = 9. But we can share: 
    # Actually A*B3 needs 2 mults (A00*B02 + A10*B12? no). A (2x2) * B3 (2x1) = 2 mults.
    # Total 7+2 = 9. Hmm that's 9 not 8.
    # Alternative: treat as 2x2x3 directly. Known rank of 2x2x3 is 8.
    # Build via: use Strassen for 2x2x2 on cols (0,1) gives C0,C1 (7 mults).
    # For C2: C2[0]=A00*B02+A10*B12, C2[1]=A01*B02+A11*B12.
    # We can get C2 from existing Strassen intermediates? Not directly.
    # Use 1 extra: M8 = (A00+A10+A01+A11)*B02? No that gives sum.
    # Let's just do rank 9 for 2x2x3 to be safe, OR find rank 8.
    # Known: rank(2,2,3)=8. Construction: 
    # M1=(A00+A11)(B00+B11)
    # M2=(A10+A11)B00
    # M3=A00(B10-B11)
    # M4=A11(B01-B00)
    # M5=(A00+A01)B11
    # M6=(A10-A00)(B00+B01)
    # M7=(A01-A11)(B10+B11)
    # M8=(A00+A10+B02... ) 
    # This is getting complex. Let me just use rank 9 (safe) for 2x2x3 and rely on partition search.

    def block_223():
        # 2x2x3: A(2x2), B(2x3), C(2x3). Rank 9 (Strassen for cols 0,1 + 2 mults for col 2).
        rank = 9
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]
        # First 7: Strassen 2x2x2 for C cols 0,1 (B cols 0,1)
        r, U7, V7, W7 = strassen_222()
        # Map: V7[j*2+k] -> V[j*3+k], W7[k*2+i] -> W[k*3+i]
        for i in range(2):
            for j in range(2):
                for rr in range(7):
                    U[i*2+j][rr] = U7[i*2+j][rr]
        for j in range(2):
            for k in range(2):
                for rr in range(7):
                    V[j*3+k][rr] = V7[j*2+k][rr]
        for k in range(2):
            for i in range(2):
                for rr in range(7):
                    W[k*3+i][rr] = W7[k*2+i][rr]
        # 2 extra mults for C col 2: C[0,2]=A00*B02+A10*B12, C[1,2]=A01*B02+A11*B12
        # M8: A00*B02 -> contributes to C[0,2]
        U[0][7]=1; V[0*3+2][7]=1; W[2*3+0][7]=1
        # M9: A10*B12 -> contributes to C[0,2]
        U[2][8]=1; V[1*3+2][8]=1; W[2*3+0][8]=1
        # Wait that only covers C[0,2]. Need C[1,2] too. That's 4 mults for col 2 (2x2x1).
        # So total 7+4=11? No: A(2x2)*B3(2x1) is 2x2x1 = 4 mults schoolbook. 
        # So 2x2x3 = 7 + 4 = 11. That's worse than schoolbook 12 only marginally.
        # Not useful. Drop 223.
        return rank, U, V, W

    def block_decomp(a, b, c):
        if a == 2 and b == 2 and c == 2:
            return strassen_222()
        return schoolbook_block(a, b, c)

    # ---------- Partitions into {1,2} ----------
    def partitions(total, max_part=2):
        res = []
        def rec(rem, cur):
            if rem == 0:
                res.append(tuple(cur))
                return
            for s in (1, 2):
                if s <= rem:
                    cur.append(s)
                    rec(rem - s, cur)
                    cur.pop()
        rec(total, [])
        return res

    pn = partitions(n)
    pm = partitions(m)
    pp = partitions(p)

    best_rank = None
    best_partition = None

    for part_n in pn:
        for part_m in pm:
            for part_p in pp:
                total = 0
                for a in part_n:
                    for b in part_m:
                        for c in part_p:
                            r, _, _, _ = block_decomp(a, b, c)
                            total += r
                if best_rank is None or total < best_rank:
                    best_rank = total
                    best_partition = (part_n, part_m, part_p)
                if _time_exceeded():
                    break
            if _time_exceeded():
                break
        if _time_exceeded():
            break

    part_n, part_m, part_p = best_partition
    rank_total = best_rank

    U = [[0]*rank_total for _ in range(n*m)]
    V = [[0]*rank_total for _ in range(m*p)]
    W = [[0]*rank_total for _ in range(p*n)]

    def place(rank_offset, i0, j0, k0, a, b, c, U_loc, V_loc, W_loc):
        for i_l in range(a):
            for j_l in range(b):
                u_g = (i0+i_l)*m + (j0+j_l)
                for r in range(len(U_loc[0])):
                    U[u_g][rank_offset+r] += U_loc[i_l*b+j_l][r]
        for j_l in range(b):
            for k_l in range(c):
                v_g = (j0+j_l)*p + (k0+k_l)
                for r in range(len(V_loc[0])):
                    V[v_g][rank_offset+r] += V_loc[j_l*c+k_l][r]
        for k_l in range(c):
            for i_l in range(a):
                w_g = (k0+k_l)*n + (i0+i_l)
                for r in range(len(W_loc[0])):
                    W[w_g][rank_offset+r] += W_loc[k_l*a+i_l][r]

    rank_offset = 0
    i0 = 0
    for a in part_n:
        j0 = 0
        for b in part_m:
            k0 = 0
            for c in part_p:
                r, U_loc, V_loc, W_loc = block_decomp(a, b, c)
                place(rank_offset, i0, j0, k0, a, b, c, U_loc, V_loc, W_loc)
                rank_offset += r
                k0 += c
            j0 += b
        i0 += a

    # Safety: verify; fall back to schoolbook if it fails.
    if not _verify(U, V, W):
        rank_total = n * m * p
        U = [[0]*rank_total for _ in range(n*m)]
        V = [[0]*rank_total for _ in range(m*p)]
        W = [[0]*rank_total for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
                    idx += 1

    return {"U": U, "V": V, "W": W}
