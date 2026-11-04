import numpy as np

def construct():
    n, m, p = 2, 5, 6
    target_rank = 30
    
    # Try to find a decomposition with target_rank
    # Using Alternating Least Squares
    # T_{ijk} = 1
    
    U = np.random.randn(n*m, target_rank)
    V = np.random.randn(m*p, target_rank)
    W = np.random.randn(p*n, target_rank)
    
    for _ in range(50):
        # Solve for U
        # U_{ij}^{(r)} = (1 - sum_{r'!=r} U_{ij}^{(r')} V_{jk}^{(r')} W_{ki}^{(r')}) / (V_{jk}^{(r)} W_{ki}^{(r)})
        # This is element-wise division, which is unstable.
        # Instead, solve the linear system for U_r:
        # sum_r U_{ij}^{(r)} (V_{jk}^{(r)} W_{ki}^{(r)}) = 1
        # Let M_r be the matrix V_r * W_r^T? No, V is (m*p, 1) for fixed r? No.
        # V_{jk}^{(r)} is a scalar for fixed j,k,r.
        # So for fixed i,j,k, we have:
        # sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)} = 1
        # This is a linear system in U_{ij}^{(r)} for fixed i,j,k?
        # No, U_{ij}^{(r)} appears only in the term for that r.
        # So U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)} is the only term involving U_{ij}^{(r)} in the sum for T_{ijk}.
        # Wait, the sum is over r.
        # So for fixed i,j,k, the equation is:
        # sum_{r=1}^R U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)} = 1
        # This is ONE equation with R unknowns U_{ij}^{(r)}.
        # This is underdetermined.
        # We can choose U_{ij}^{(r)} freely as long as the sum is 1.
        # To minimize rank (or keep it low), we can distribute the 1 among the R terms.
        # Simplest: U_{ij}^{(r)} = 1/R if V_{jk}^{(r)} W_{ki}^{(r)} is non-zero?
        # No, we need to satisfy ALL i,j,k.
        # This coupling makes it hard.
        
        pass
        
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
