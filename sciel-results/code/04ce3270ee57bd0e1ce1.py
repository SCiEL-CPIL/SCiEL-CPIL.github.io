"""Matrix multiplication 2,2,4 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8

    # Build the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(nm):
        for j in range(mp):
            for k in range(pn):
                if (i % m == j // p) and (k % n == i // m):
                    T[i, j, k] = 1.0

    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        # Check all tensor entries
        for i in range(nm):
            for j in range(mp):
                for k in range(pn):
                    expected = T[i, j, k]
                    s = 0.0
                    for r in range(R):
                        s += U[i][r] * V[j][r] * W[k][r]
                    if abs(s - expected) > 1e-9:
                        return False
        return True

    def build_rank14():
        """Standard rank-14: two independent Strassen 2x2 blocks."""
        R = 14
        U = [[0]*R for _ in range(nm)]
        V = [[0]*R for _ in range(mp)]
        W = [[0]*R for _ in range(pn)]
        
        # Left block (B cols 0,1): Strassen terms r=0..6
        # A: 0=a00, 1=a01, 2=a10, 3=a11
        # B: 0=b00, 1=b01, 4=b10, 5=b11
        # C: 0=C00, 1=C10, 2=C01, 3=C11
        
        # M1 = (a00+a11)(b00+b11)
        U[0][0]=1; U[3][0]=1
        V[0][0]=1; V[5][0]=1
        W[0][0]=1; W[3][0]=1
        # M2 = (a10+a11)b00
        U[2][1]=1; U[3][1]=1
        V[0][1]=1
        W[1][1]=1; W[3][1]=-1
        # M3 = a00(b01-b11)
        U[0][2]=1
        V[1][2]=1; V[5][2]=-1
        W[2][2]=1; W[3][2]=1
        # M4 = a11(b10-b00)
        U[3][3]=1
        V[4][3]=1; V[0][3]=-1
        W[0][3]=1; W[1][3]=1
        # M5 = (a00+a01)b11
        U[0][4]=1; U[1][4]=1
        V[5][4]=1
        W[0][4]=-1; W[2][4]=1
        # M6 = (a10-a00)(b00+b01)
        U[2][5]=1; U[0][5]=-1
        V[0][5]=1; V[1][5]=1
        W[3][5]=1
        # M7 = (a01-a11)(b10+b11)
        U[1][6]=1; U[3][6]=-1
        V[4][6]=1; V[5][6]=1
        W[0][6]=1

        # Right block (B cols 2,3): Strassen terms r=7..13
        # B: 2=b02, 3=b03, 6=b12, 7=b13
        # C: 4=C02, 5=C12, 6=C03, 7=C13
        
        # N1 = (a00+a11)(b02+b13)
        U[0][7]=1; U[3][7]=1
        V[2][7]=1; V[7][7]=1
        W[4][7]=1; W[7][7]=1
        # N2 = (a10+a11)b02
        U[2][8]=1; U[3][8]=1
        V[2][8]=1
        W[5][8]=1; W[7][8]=-1
        # N3 = a00(b03-b13)
        U[0][9]=1
        V[3][9]=1; V[7][9]=-1
        W[6][9]=1; W[7][9]=1
        # N4 = a11(b12-b02)
        U[3][10]=1
        V[6][10]=1; V[2][10]=-1
        W[4][10]=1; W[5][10]=1
        # N5 = (a00+a01)b13
        U[0][11]=1; U[1][11]=1
        V[7][11]=1
        W[4][11]=-1; W[6][11]=1
        # N6 = (a10-a00)(b02+b03)
        U[2][12]=1; U[0][12]=-1
        V[2][12]=1; V[3][12]=1
        W[7][12]=1
        # N7 = (a01-a11)(b12+b13)
        U[1][13]=1; U[3][13]=-1
        V[6][13]=1; V[7][13]=1
        W[4][13]=1

        return {"U": U, "V": V, "W": W}

    def try_rank13_optimization():
        """
        Search for rank-13 decomposition using least squares optimization.
        """
        if time.time() > deadline - 30:
            return None

        R = 13
        # Flatten T to 1D
        T_flat = T.flatten()

        def residuals(x):
            # x is flattened [U (4*13), V (8*13), W (8*13)]
            U_flat = x[:4*R]
            V_flat = x[4*R:4*R+8*R]
            W_flat = x[4*R+8*R:]
            
            U_mat = U_flat.reshape(4, R)
            V_mat = V_flat.reshape(8, R)
            W_mat = W_flat.reshape(8, R)
            
            # Compute T_approx
            # T_approx[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
            # Use einsum for speed
            T_approx = np.einsum('ir,jr,kr->ijk', U_mat, V_mat, W_mat)
            return (T_approx - T).flatten()

        # Try multiple random starts
        np.random.seed(42)
        for attempt in range(50):
            if time.time() > deadline - 5:
                break

            # Initialize with random values in [-1, 1]
            # Or use the rank-14 solution truncated?
            # Let's try random sparse initialization
            U_init = np.random.choice([-1, 0, 1], size=(4, R))
            V_init = np.random.choice([-1, 0, 1], size=(8, R))
            W_init = np.random.choice([-1, 0, 1], size=(8, R))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=1000)
            except Exception:
                continue

            if result.cost < 1e-10:
                # Check if solution is close to integers
                U_sol = result.x[:4*R].reshape(4, R)
                V_sol = result.x[4*R:4*R+8*R].reshape(8, R)
                W_sol = result.x[4*R+8*R:].reshape(8, R)
                
                # Round to nearest integer (or half-integer if needed, but integers are expected)
                U_int = np.round(U_sol).astype(int)
                V_int = np.round(V_sol).astype(int)
                W_int = np.round(W_sol).astype(int)
                
                # Verify exact integer solution
                if verify(U_int.tolist(), V_int.tolist(), W_int.tolist()):
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
        
        return None

    # Try rank-13 first
    result = try_rank13_optimization()
    if result is not None:
        return result
    
    # Fall back to rank-14
    return build_rank14()
# EVOLVE-BLOCK-END
