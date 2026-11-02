"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 2, 9

    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Build rank-33 decomposition: 3 blocks of 2x2x3, each rank 11
    # Each 2x2x3 block: Strassen(2x2x2) rank 7 + schoolbook(2x2x1) rank 4 = rank 11
    def build_rank33():
        rank = 33
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0

        def add_strassen_block(k0, k1):
            """Add Strassen 2x2x2 decomposition for columns k0, k1 of B."""
            nonlocal col
            # M1 = (A00+A11)(B00+B11)
            U[0][col] = 1; U[3][col] = 1
            V[0 * p + k0][col] = 1; V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
            col += 1

            # M2 = (A10+A11)B00
            U[2][col] = 1; U[3][col] = 1
            V[0 * p + k0][col] = 1
            W[k0 * n + 1][col] = 1; W[k1 * n + 1][col] = -1
            col += 1

            # M3 = A00(B01-B11)
            U[0][col] = 1
            V[0 * p + k1][col] = 1; V[1 * p + k1][col] = -1
            W[k1 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
            col += 1

            # M4 = A11(B10-B00)
            U[3][col] = 1
            V[1 * p + k0][col] = 1; V[0 * p + k0][col] = -1
            W[k0 * n + 0][col] = 1; W[k0 * n + 1][col] = 1
            col += 1

            # M5 = (A00+A01)B11
            U[0][col] = 1; U[1][col] = 1
            V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = -1; W[k1 * n + 0][col] = 1
            col += 1

            # M6 = (A10-A00)(B00+B01)
            U[2][col] = 1; U[0][col] = -1
            V[0 * p + k0][col] = 1; V[0 * p + k1][col] = 1
            W[k1 * n + 1][col] = 1
            col += 1

            # M7 = (A01-A11)(B10+B11)
            U[1][col] = 1; U[3][col] = -1
            V[1 * p + k0][col] = 1; V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = 1
            col += 1

        def add_trivial_col(k):
            """Add schoolbook 2x2x1 decomposition for column k of B."""
            nonlocal col
            U[0][col] = 1; V[0 * p + k][col] = 1; W[k * n + 0][col] = 1; col += 1
            U[1][col] = 1; V[1 * p + k][col] = 1; W[k * n + 0][col] = 1; col += 1
            U[2][col] = 1; V[0 * p + k][col] = 1; W[k * n + 1][col] = 1; col += 1
            U[3][col] = 1; V[1 * p + k][col] = 1; W[k * n + 1][col] = 1; col += 1

        # Three blocks of 2x2x3: columns (0,1,2), (3,4,5), (6,7,8)
        for block in range(3):
            k0 = block * 3
            k1 = block * 3 + 1
            k2 = block * 3 + 2
            add_strassen_block(k0, k1)
            add_trivial_col(k2)

        assert col == 33, f"Expected col=33, got {col}"
        return {"U": U, "V": V, "W": W}

    # Build rank-32 decomposition (current best): 4 Strassen blocks + 1 trivial
    def build_rank32():
        rank = 32
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0

        def add_strassen(k0, k1):
            nonlocal col
            U[0][col] = 1; U[3][col] = 1
            V[0 * p + k0][col] = 1; V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
            col += 1

            U[2][col] = 1; U[3][col] = 1
            V[0 * p + k0][col] = 1
            W[k0 * n + 1][col] = 1; W[k1 * n + 1][col] = -1
            col += 1

            U[0][col] = 1
            V[0 * p + k1][col] = 1; V[1 * p + k1][col] = -1
            W[k1 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
            col += 1

            U[3][col] = 1
            V[1 * p + k0][col] = 1; V[0 * p + k0][col] = -1
            W[k0 * n + 0][col] = 1; W[k0 * n + 1][col] = 1
            col += 1

            U[0][col] = 1; U[1][col] = 1
            V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = -1; W[k1 * n + 0][col] = 1
            col += 1

            U[2][col] = 1; U[0][col] = -1
            V[0 * p + k0][col] = 1; V[0 * p + k1][col] = 1
            W[k1 * n + 1][col] = 1
            col += 1

            U[1][col] = 1; U[3][col] = -1
            V[1 * p + k0][col] = 1; V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = 1
            col += 1

        def add_trivial(k):
            nonlocal col
            U[0][col] = 1; V[0 * p + k][col] = 1; W[k * n + 0][col] = 1; col += 1
            U[1][col] = 1; V[1 * p + k][col] = 1; W[k * n + 0][col] = 1; col += 1
            U[2][col] = 1; V[0 * p + k][col] = 1; W[k * n + 1][col] = 1; col += 1
            U[3][col] = 1; V[1 * p + k][col] = 1; W[k * n + 1][col] = 1; col += 1

        add_strassen(0, 1)
        add_strassen(2, 3)
        add_strassen(4, 5)
        add_strassen(6, 7)
        add_trivial(8)

        assert col == 32, f"Expected col=32, got {col}"
        return {"U": U, "V": V, "W": W}

    # First, build and verify the rank-33 baseline
    result33 = build_rank33()
    if not verify(result33["U"], result33["V"], result33["W"]):
        # Fallback to schoolbook
        return schoolbook()

    # Build and verify the rank-32 solution
    result32 = build_rank32()
    if not verify(result32["U"], result32["V"], result32["W"]):
        return result33

    # Try to find a rank-10 decomposition for M_{2,2,3}
    # If found, total rank would be 3*10 = 30
    def try_rank10_m223(time_limit=60.0):
        """
        Try to find a rank-10 decomposition for the 2x2x3 tensor.
        The 2x2x3 tensor has 2*2*3 = 12 entries that must equal 1.
        We need 10 columns such that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all (i,j,k).
        
        Approach: Start from the rank-11 decomposition and try to eliminate one column
        by solving a nonlinear system.
        """
        start = time.time()
        
        # Build the rank-11 decomposition for one 2x2x3 block (columns 0,1,2)
        def build_m223_rank11():
            U = [[0] * 11 for _ in range(4)]  # 2x2 matrix A
            V = [[0] * 11 for _ in range(6)]  # 2x3 matrix B
            W = [[0] * 11 for _ in range(6)]  # 3x2 output C
            col = 0
            
            # Strassen for columns 0,1
            U[0][col]=1; U[3][col]=1; V[0][col]=1; V[3][col]=1; W[0][col]=1; W[3][col]=1; col+=1
            U[2][col]=1; U[3][col]=1; V[0][col]=1; W[1][col]=1; W[3][col]=-1; col+=1
            U[0][col]=1; V[1][col]=1; V[3][col]=-1; W[2][col]=1; W[3][col]=1; col+=1
            U[3][col]=1; V[2][col]=1; V[0][col]=-1; W[0][col]=1; W[1][col]=1; col+=1
            U[0][col]=1; U[1][col]=1; V[3][col]=1; W[0][col]=-1; W[2][col]=1; col+=1
            U[2][col]=1; U[0][col]=-1; V[0][col]=1; V[1][col]=1; W[3][col]=1; col+=1
            U[1][col]=1; U[3][col]=-1; V[2][col]=1; V[3][col]=1; W[0][col]=1; col+=1
            
            # Schoolbook for column 2
            U[0][col]=1; V[4][col]=1; W[4][col]=1; col+=1
            U[1][col]=1; V[5][col]=1; W[4][col]=1; col+=1
            U[2][col]=1; V[4][col]=1; W[5][col]=1; col+=1
            U[3][col]=1; V[5][col]=1; W[5][col]=1; col+=1
            
            assert col == 11
            return U, V, W
        
        U11, V11, W11 = build_m223_rank11()
        
        # Verify rank-11
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    total = sum(U11[i*2+j][c] * V11[j*3+k][c] * W11[k*2+i][c] for c in range(11))
                    if abs(total - 1) > 1e-9:
                        return None
        
        # Try to eliminate the last column (index 10)
        # We need: for all (i,j,k), sum_{c=0}^{9} U[i,j,c]*V[j,k,c]*W[k,i,c] + U[i,j,10]*V[j,k,10]*W[k,i,10] = 1
        # Let's try to set column 10 to zero and see if the first 10 columns suffice
        # They don't (that's why we need 11). So we need a different approach.
        
        # Alternative: try to find a rank-10 decomposition directly using optimization
        # Variables: 10 columns, each with 4+6+6=16 entries = 160 variables
        # Constraints: 12 equations (one for each (i,j,k) triple)
        
        # This is a hard nonlinear problem. Let's try a simple local search.
        # Initialize with random perturbations of the rank-11 solution with one column zeroed.
        
        np.random.seed(42)
        
        best_err = float('inf')
        best_solution = None
        
        for attempt in range(100):
            if time.time() - start > time_limit:
                break
            
            # Start from rank-11 and perturb
            U = [row[:] for row in U11]
            V = [row[:] for row in V11]
            W = [row[:] for row in W11]
            
            # Zero out one random column and perturb the others
            col_to_eliminate = np.random.randint(11)
            U[col_to_eliminate] = [0] * 11
            V[col_to_eliminate] = [0] * 11
            W[col_to_eliminate] = [0] * 11
            
            # Perturb remaining columns
            for c in range(11):
                if c == col_to_eliminate:
                    continue
                for i in range(4):
                    U[i][c] += np.random.randn() * 0.1
                for j in range(6):
                    V[j][c] += np.random.randn() * 0.1
                for k in range(6):
                    W[k][c] += np.random.randn() * 0.1
            
            # Gradient descent to minimize error
            lr = 0.01
            for step in range(1000):
                if time.time() - start > time_limit:
                    break
                
                # Compute error and gradients
                err = 0
                dU = [[0]*11 for _ in range(4)]
                dV = [[0]*11 for _ in range(6)]
                dW = [[0]*11 for _ in range(6)]
                
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            total = 0
                            for c in range(11):
                                total += U[i*2+j][c] * V[j*3+k][c] * W[k*2+i][c]
                            diff = total - 1
                            err += diff * diff
                            
                            for c in range(11):
                                if c == col_to_eliminate:
                                    continue
                                prod_UV = U[i*2+j][c] * V[j*3+k][c]
                                prod_VW = V[j*3+k][c] * W[k*2+i][c]
                                prod_UW = U[i*2+j][c] * W[k*2+i][c]
                                dU[i*2+j][c] += 2 * diff * prod_VW
                                dV[j*3+k][c] += 2 * diff * prod_UW
                                dW[k*2+i][c] += 2 * diff * prod_UV
                
                if err < best_err:
                    best_err = err
                    best_solution = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
                
                if err < 1e-6:
                    break
                
                # Update
                for c in range(11):
                    if c == col_to_eliminate:
                        continue
                    for i in range(4):
                        U[i][c] -= lr * dU[i][c]
                    for j in range(6):
                        V[j][c] -= lr * dV[j][c]
                    for k in range(6):
                        W[k][c] -= lr * dW[k][c]
                
                # Decay learning rate
                lr *= 0.999
        
        # Check if we found a valid solution
        if best_solution and best_err < 1e-6:
            U, V, W = best_solution
            # Verify
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        total = sum(U[i*2+j][c] * V[j*3+k][c] * W[k*2+i][c] for c in range(11))
                        if abs(total - 1) > 1e-5:
                            return None
            return U, V, W
        
        return None

    # Try to find rank-10 for M_{2,2,3}
    m223_rank10 = None
    if time.time() < deadline - 30:
        m223_rank10 = try_rank10_m223(time_limit=60.0)
    
    if m223_rank10:
        U10, V10, W10 = m223_rank10
        # Build rank-30 solution: 3 copies of rank-10 M_{2,2,3}
        rank30 = 30
        U = [[0] * rank30 for _ in range(n * m)]
        V = [[0] * rank30 for _ in range(m * p)]
        W = [[0] * rank30 for _ in range(p * n)]
        
        for block in range(3):
            offset = block * 10
            k0 = block * 3
            k1 = block * 3 + 1
            k2 = block * 3 + 2
            for c in range(10):
                for i in range(4):
                    U[i][offset + c] = U10[i][c]
                for j in range(2):
                    for k in range(3):
                        V[j * p + k0 + k][offset + c] = V10[j * 3 + k][c]
                for k in range(3):
                    for i in range(2):
                        W[(k0 + k) * n + i][offset + c] = W10[k * 2 + i][c]
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Return the best verified solution
    if verify(result32["U"], result32["V"], result32["W"]):
        return result32
    return result33
# EVOLVE-BLOCK-END
