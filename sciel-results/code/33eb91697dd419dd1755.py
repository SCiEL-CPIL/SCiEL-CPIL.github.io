"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # The inspiration algorithm achieves rank 55
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Iterate over 5 blocks of 2 columns (k0, k1)
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1: (A00 + A11) * (B0,k0 + B1,k1)
        # U: A00(1), A11(1) -> indices 0, 4
        # V: B0,k0(1), B1,k1(1) -> indices k0, 10+k1
        # W: C0,k0(1), C1,k1(1) -> indices 2*k0, 2*k1+1
        U[0 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M2: (A10 + A11) * B0,k0
        # U: A10(1), A11(1) -> indices 3, 4
        # V: B0,k0(1) -> index k0
        # W: C1,k0(1), C1,k1(-1) -> indices 2*k0+1, 2*k1+1
        U[1 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1
        
        # M3: A00 * (B0,k1 - B1,k1)
        # U: A00(1) -> index 0
        # V: B0,k1(1), B1,k1(-1) -> indices 10+k1, 10+k1
        # W: C0,k1(1), C1,k1(1) -> indices 2*k1, 2*k1+1
        U[0 * 3 + 0][col] += 1
        V[0 * 10 + k1][col] += 1
        V[1 * 10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M4: A11 * (B1,k0 - B0,k0)
        # U: A11(1) -> index 4
        # V: B1,k0(1), B0,k0(-1) -> indices 10+k0, k0
        # W: C0,k0(1), C1,k0(1) -> indices 2*k0, 2*k0+1
        U[1 * 3 + 1][col] += 1
        V[1 * 10 + k0][col] += 1
        V[0 * 10 + k0][col] -= 1
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # M5: (A00 + A01) * B1,k1
        # U: A00(1), A01(1) -> indices 0, 1
        # V: B1,k1(1) -> index 10+k1
        # W: C0,k0(-1), C0,k1(1) -> indices 2*k0, 2*k1
        U[0 * 3 + 0][col] += 1
        U[0 * 3 + 1][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # M6: (A10 - A00) * (B0,k0 + B0,k1)
        # U: A10(1), A00(-1) -> indices 3, 0
        # V: B0,k0(1), B0,k1(1) -> indices k0, k1
        # W: C1,k1(1) -> index 2*k1+1
        U[1 * 3 + 0][col] += 1
        U[0 * 3 + 0][col] -= 1
        V[0 * 10 + k0][col] += 1
        V[0 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        # M7: (A01 - A11) * (B1,k0 + B1,k1)
        # U: A01(1), A11(-1) -> indices 1, 4
        # V: B1,k0(1), B1,k1(1) -> indices 10+k0, 10+k1
        # W: C0,k0(1) -> index 2*k0
        U[0 * 3 + 1][col] += 1
        U[1 * 3 + 1][col] -= 1
        V[1 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        # W00: A02 * B2,k0
        # U: A02(1) -> index 2
        # V: B2,k0(1) -> index 20+k0
        # W: C0,k0(1) -> index 2*k0
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        # W01: A02 * B2,k1
        # U: A02(1) -> index 2
        # V: B2,k1(1) -> index 20+k1
        # W: C0,k1(1) -> index 2*k1
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # W10: A12 * B2,k0
        # U: A12(1) -> index 5
        # V: B2,k0(1) -> index 20+k0
        # W: C1,k0(1) -> index 2*k0+1
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # W11: A12 * B2,k1
        # U: A12(1) -> index 5
        # V: B2,k1(1) -> index 20+k1
        # W: C1,k1(1) -> index 2*k1+1
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
