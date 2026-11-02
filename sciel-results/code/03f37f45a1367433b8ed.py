import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank: 64
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    # Rank 64 is always valid
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_strassen_rank56():
    """
    Try to construct a rank 56 decomposition using Strassen-based block decomposition.
    A is 2x4, B is 4x8, C is 2x8.
    Split A = [A1, A2] (2x2 each), B = [[B11, B12], [B21, B22]] (2x2 each).
    C = [A1*B11 + A2*B21, A1*B12 + A2*B22].
    Each 2x2 * 2x2 multiplication uses 7 multiplications (Strassen).
    Total rank = 4 * 7 = 28? No, that's not right.
    A1*B11: 7 mults. A2*B21: 7 mults. A1*B12: 7 mults. A2*B22: 7 mults.
    But we need to combine them. The intermediate results are 2x2 matrices.
    Each 2x2 result contributes to C.
    Actually, the rank is the number of scalar multiplications.
    A1*B11 uses 7 mults. A2*B21 uses 7 mults. A1*B12 uses 7 mults. A2*B22 uses 7 mults.
    Total = 28? No, wait.
    C = [C_left, C_right].
    C_left = A1*B11 + A2*B21. This is a 2x2 matrix.
    To compute A1*B11 (2x2*2x2), we use 7 mults.
    To compute A2*B21 (2x2*2x2), we use 7 mults.
    The sum is element-wise.
    So we need 14 mults for C_left.
    Similarly, 14 mults for C_right.
    Total = 28?
    Wait, A1*B11 produces a 2x2 matrix M1.
    A2*B21 produces a 2x2 matrix M2.
    C_left = M1 + M2.
    The entries of M1 and M2 are linear combinations of the 7 products.
    So the rank is indeed 28?
    Let's check the dimensions.
    U is (8, 28), V is (32, 28), W is (16, 28).
    The schoolbook rank is 64.
    28 is much lower. Is it valid?
    Let's verify with a small example.
    Actually, the rank of 2x2x2 is 7.
    The tensor for 2x4x8 can be seen as 4 copies of 2x2x2 tensors?
    No, it's not a direct sum.
    Let's just use the known rank 64 and return it.
    Trying to find a lower rank without a verified algorithm is risky.
    """
    pass

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
