"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    def check_decomposition_exact(U, V, W, rank):
        """Check using exact arithmetic with fractions."""
        U_frac = [[Fraction(x) for x in row] for row in U]
        V_frac = [[Fraction(x) for x in row] for row in V]
        W_frac = [[Fraction(x) for x in row] for row in W]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for c in range(rank):
                        total += U_frac[i * m + j][c] * V_frac[j * p + k][c] * W_frac[k * n + i][c]
                    if total != 1:
                        return False
        return True

    def check_decomposition_float(U, V, W, rank, tol=1e-9):
        """Quick float check."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Build schoolbook rank-18 as fallback
    schoolbook_rank = n * m * p
    sb_U = [[0] * schoolbook_rank for _ in range(nm)]
    sb_V = [[0] * schoolbook_rank for _ in range(mp)]
    sb_W = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                sb_U[i * m + j][col] = 1
                sb_V[j * p + k][col] = 1
                sb_W[k * n + i][col] = 1
                col += 1

    # Try to construct rank-17 using Strassen-like embedding
    # For (2,3,3): use Strassen on A[:,:2]*B[:2,:2] (7 terms) + remaining (10 terms) = 17
    def try_rank17():
        """Construct a rank-17 decomposition using Strassen embedding."""
        rank = 17
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]

        # Strassen terms for A[:,:2] * B[:2,:2] -> C[:,:2]
        # A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
        # B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
        # p1 = (a+b)(e+g), p2 = (c+d)e, p3 = a(f+h), p4 = d(g-e)
        # p5 = (a+d)(e+h), p6 = (b-d)(f+g), p7 = (a+c)(h-e)
        # C[0,0] = p1 + p4 - p5 + p7
        # C[0,1] = p3 + p5
        # C[1,0] = p2 + p4
        # C[1,1] = p1 - p2 + p3 + p6

        # Term 0: p1 = (A[0,0]+A[0,1]) * (B[0,0]+B[1,0])
        U[0*3+0][0] += 1  # A[0,0]
        U[0*3+1][0] += 1  # A[0,1]
        V[0*3+0][0] += 1  # B[0,0]
        V[1*3+0][0] += 1  # B[1,0]
        W[0*2+0][0] += 1  # C[0,0]
        W[1*2+0][0] += 1  # C[1,0]

        # Term 1: p2 = (A[1,0]+A[1,1]) * B[0,0]
        U[1*3+0][1] += 1  # A[1,0]
        U[1*3+1][1] += 1  # A[1,1]
        V[0*3+0][1] += 1  # B[0,0]
        W[0*2+0][1] += 1  # C[0,0]
        W[1*2+0][1] += 1  # C[1,0]

        # Term 2: p3 = A[0,0] * (B[0,1]+B[1,1])
        U[0*3+0][2] += 1  # A[0,0]
        V[0*3+1][2] += 1  # B[0,1]
        V[1*3+1][2] += 1  # B[1,1]
        W[1*2+0][2] += 1  # C[0,1]
        W[1*2+1][2] += 1  # C[1,1]

        # Term 3: p4 = A[1,1] * (B[1,0]-B[0,0])
        U[1*3+1][3] += 1  # A[1,1]
        V[1*3+0][3] += 1  # B[1,0]
        V[0*3+0][3] -= 1  # B[0,0]
        W[0*2+0][3] += 1  # C[0,0]
        W[1*2+0][3] += 1  # C[1,0]

        # Term 4: p5 = (A[0,0]+A[1,0]) * (B[0,0]+B[1,1])
        U[0*3+0][4] += 1  # A[0,0]
        U[1*3+0][4] += 1  # A[1,0]
        V[0*3+0][4] += 1  # B[0,0]
        V[1*3+1][4] += 1  # B[1,1]
        W[0*2+0][4] -= 1  # C[0,0] (negative in formula)
        W[1*2+0][4] += 1  # C[1,0]
        W[1*2+0][4] += 0  # placeholder
        # Actually C[0,1] = p3 + p5, so W for C[0,1] should get +1
        # Let me redo: W[k*n+i, c] where k is column, i is row
        # C[0,0]: k=0,i=0 -> W[0, c]
        # C[0,1]: k=1,i=0 -> W[1, c]
        # C[1,0]: k=0,i=1 -> W[2, c]
        # C[1,1]: k=1,i=1 -> W[3, c]
        # C[0,2]: k=2,i=0 -> W[4, c]
        # C[1,2]: k=2,i=1 -> W[5, c]
        # I had the W indexing wrong. Let me restart.
        return None  # Placeholder, will rebuild

    # Restart rank-17 construction with correct W indexing
    # W row index: k*n + i, where k in [0,p), i in [0,n)
    # So W[0] = (k=0,i=0) -> C[0,0]
    #    W[1] = (k=0,i=1) -> C[1,0]
    #    W[2] = (k=1,i=0) -> C[0,1]
    #    W[3] = (k=1,i=1) -> C[1,1]
    #    W[4] = (k=2,i=0) -> C[0,2]
    #    W[5] = (k=2,i=1) -> C[1,2]

    def build_rank17():
        rank = 17
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]

        # U row: i*m + j -> U[i*3+j]
        # V row: j*p + k -> V[j*3+k]
        # W row: k*n + i -> W[k*2+i]

        # Strassen for A[:,:2] * B[:2,:2] -> C[:,:2]
        # p1 = (A[0,0]+A[0,1])*(B[0,0]+B[1,0])
        # p2 = (A[1,0]+A[1,1])*B[0,0]
        # p3 = A[0,0]*(B[0,1]+B[1,1])
        # p4 = A[1,1]*(B[1,0]-B[0,0])
        # p5 = (A[0,0]+A[1,0])*(B[0,0]+B[1,1])
        # p6 = (A[0,1]-A[1,1])*(B[1,0]+B[0,1])
        # p7 = (A[0,0]+A[1,0])*(B[1,1]-B[0,0])
        #
        # C[0,0] = p1 + p4 - p5 + p7
        # C[0,1] = p3 + p5
        # C[1,0] = p2 + p4
        # C[1,1] = p1 - p2 + p3 + p6

        # Term 0: p1
        U[0][0] += 1; U[1][0] += 1  # A[0,0], A[0,1]
        V[0][0] += 1; V[3][0] += 1  # B[0,0], B[1,0]
        W[0][0] += 1; W[2][0] += 1  # C[0,0], C[0,1]... wait
        # C[0,0]: k=0,i=0 -> W[0]
        # C[1,0]: k=0,i=1 -> W[1]
        # p1 contributes to C[0,0] and C[1,0] (from formula C[0,0]=p1+..., C[1,0]=p2+p4, no p1 in C[1,0])
        # Actually C[0,0] = p1 + p4 - p5 + p7, C[1,1] = p1 - p2 + p3 + p6
        # So p1 contributes to C[0,0] and C[1,1]
        W[0][0] += 1   # C[0,0]
        W[3][0] += 1   # C[1,1]

        # Term 1: p2
        U[3][1] += 1; U[4][1] += 1  # A[1,0], A[1,1]
        V[0][1] += 1  # B[0,0]
        W[1][1] += 1   # C[1,0]
        W[3][1] -= 1   # C[1,1] (negative)

        # Term 2: p3
        U[0][2] += 1  # A[0,0]
        V[1][2] += 1; V[4][2] += 1  # B[0,1], B[1,1]
        W[2][2] += 1   # C[0,1]
        W[3][2] += 1   # C[1,1]

        # Term 3: p4
        U[4][3] += 1  # A[1,1]
        V[3][3] += 1; V[0][3] -= 1  # B[1,0], -B[0,0]
        W[0][3] += 1   # C[0,0]
        W[1][3] += 1   # C[1,0]

        # Term 4: p5
        U[0][4] += 1; U[3][4] += 1  # A[0,0], A[1,0]
        V[0][4] += 1; V[4][4] += 1  # B[0,0], B[1,1]
        W[0][4] -= 1   # C[0,0] (negative)
        W[2][4] += 1   # C[0,1]

        # Term 5: p6
        U[1][5] += 1; U[4][5] -= 1  # A[0,1], -A[1,1]
        V[3][5] += 1; V[1][5] += 1  # B[1,0], B[0,1]
        W[3][5] += 1   # C[1,1]

        # Term 6: p7
        U[0][6] += 1; U[3][6] += 1  # A[0,0], A[1,0]
        V[4][6] += 1; V[0][6] -= 1  # B[1,1], -B[0,0]
        W[0][6] += 1   # C[0,0]

        # Now remaining terms for the rest:
        # Need C[:,2] = A[:,0]*B[0,2] + A[:,1]*B[1,2] + A[:,2]*B[2,2]
        # And C[:,0] += A[:,2]*B[2,0]
        # And C[:,1] += A[:,2]*B[2,1]
        # That's 2+2+2+2+2 = 10 more terms (terms 7-16)

        # Term 7: A[0,0]*B[0,2] -> C[0,2]
        U[0][7] += 1
        V[2][7] += 1
        W[4][7] += 1

        # Term 8: A[1,0]*B[0,2] -> C[1,2]
        U[3][8] += 1
        V[2][8] += 1
        W[5][8] += 1

        # Term 9: A[0,1]*B[1,2] -> C[0,2]
        U[1][9] += 1
        V[5][9] += 1
        W[4][9] += 1

        # Term 10: A[1,1]*B[1,2] -> C[1,2]
        U[4][10] += 1
        V[5][10] += 1
        W[5][10] += 1

        # Term 11: A[0,2]*B[2,0] -> C[0,0]
        U[2][11] += 1
        V[6][11] += 1
        W[0][11] += 1

        # Term 12: A[1,2]*B[2,0] -> C[1,0]
        U[5][12] += 1
        V[6][12] += 1
        W[1][12] += 1

        # Term 13: A[0,2]*B[2,1] -> C[0,1]
        U[2][13] += 1
        V[7][13] += 1
        W[2][13] += 1

        # Term 14: A[1,2]*B[2,1] -> C[1,1]
        U[5][14] += 1
        V[7][14] += 1
        W[3][14] += 1

        # Term 15: A[0,2]*B[2,2] -> C[0,2]
        U[2][15] += 1
        V[8][15] += 1
        W[4][15] += 1

        # Term 16: A[1,2]*B[2,2] -> C[1,2]
        U[5][16] += 1
        V[8][16] += 1
        W[5][16] += 1

        return U, V, W

    U17, V17, W17 = build_rank17()
    if check_decomposition_exact(U17, V17, W17, 17):
        best_U, best_V, best_W = U17, V17, W17
        best_rank = 17
    else:
        # Fall back to schoolbook
        best_U, best_V, best_W = sb_U, sb_V, sb_W
        best_rank = schoolbook_rank

    deadline = time.time() + 110

    # Search for rank-16
    def search_rank16(timeout=100):
        """Structured search for rank-16 decomposition."""
        start = time.time()
        rank = 16

        # Strategy: start from rank-17 solution and try to merge/eliminate one term
        # Or use random restarts with the Strassen structure
        for attempt in range(200):
            if time.time() - start > timeout:
                break

            # Random perturbation of the rank-17 solution, dropping one column
            drop_col = random.randint(0, 16)
            U = [[U17[r][c] for c in range(17) if c != drop_col] for r in range(nm)]
            V = [[V17[r][c] for c in range(17) if c != drop_col] for r in range(mp)]
            W = [[W17[r][c] for c in range(17) if c != drop_col] for r in range(pn)]

            # Add small random perturbations
            for _ in range(5):
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    r, c = random.randint(0, nm-1), random.randint(0, rank-1)
                    U[r][c] += random.choice([-1, 0.5, 0.5, 1])
                elif mat == 'V':
                    r, c = random.randint(0, mp-1), random.randint(0, rank-1)
                    V[r][c] += random.choice([-1, 0.5, 0.5, 1])
                else:
                    r, c = random.randint(0, pn-1), random.randint(0, rank-1)
                    W[r][c] += random.choice([-1, 0.5, 0.5, 1])

            if check_decomposition_exact(U, V, W, rank):
                return U, V, W

            # Local search
            for it in range(200):
                if time.time() - start > timeout:
                    break
                if check_decomposition_exact(U, V, W, rank):
                    return U, V, W

                # Find worst (i,j,k)
                worst_err = 0
                worst_ijk = (0, 0, 0)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = Fraction(0)
                            for c in range(rank):
                                total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                            err = abs(total - 1)
                            if err > worst_err:
                                worst_err = err
                                worst_ijk = (i, j, k)

                if worst_err == 0:
                    return U, V, W

                # Try to fix by modifying entries
                i, j, k = worst_ijk
                improved = False
                for mat in ['U', 'V', 'W']:
                    for r in range(nm if mat=='U' else mp if mat=='V' else pn):
                        for c in range(rank):
                            old = (U if mat=='U' else V if mat=='V' else W)[r][c]
                            for delta in [-1, -0.5, 0.5, 1]:
                                new = old + delta
                                if mat == 'U':
                                    U[r][c] = new
                                elif mat == 'V':
                                    V[r][c] = new
                                else:
                                    W[r][c] = new

                                # Check total error
                                total_err = 0
                                for i2 in range(n):
                                    for j2 in range(m):
                                        for k2 in range(p):
                                            total = 0
                                            for c2 in range(rank):
                                                total += U[i2*m+j2][c2] * V[j2*p+k2][c2] * W[k2*n+i2][c2]
                                            total_err += abs(total - 1.0)

                                # Check if improved
                                if total_err < 0.5:
                                    improved = True
                                    break
                                else:
                                    if mat == 'U':
                                        U[r][c] = old
                                    elif mat == 'V':
                                        V[r][c] = old
                                    else:
                                        W[r][c] = old
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if not improved:
                    break

        return None

    result16 = search_rank16(timeout=100)
    if result16 is not None:
        best_U, best_V, best_W = result16
        best_rank = 16

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
