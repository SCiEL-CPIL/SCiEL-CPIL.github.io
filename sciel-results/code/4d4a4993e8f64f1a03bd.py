import numpy as np

def compute_laderman_W():
    """Compute W coefficients for Laderman's 3x3x3 decomposition."""
    # Define the 23 products as (U_coeffs, V_coeffs)
    # U_coeffs: list of (i, j, coeff)
    # V_coeffs: list of (j, k, coeff)
    
    products = [
        ([(0,0,1)], [(0,0,1)]),  # P1
        ([(0,1,1)], [(1,0,1)]),  # P2
        ([(0,2,1)], [(2,0,1)]),  # P3
        ([(1,0,1)], [(0,1,1)]),  # P4
        ([(1,1,1)], [(1,1,1)]),  # P5
        ([(1,2,1)], [(2,1,1)]),  # P6
        ([(2,0,1)], [(0,2,1)]),  # P7
        ([(2,1,1)], [(1,2,1)]),  # P8
        ([(2,2,1)], [(2,2,1)]),  # P9
        ([(0,0,1),(0,1,1)], [(1,0,1),(1,1,1)]),  # P10
        ([(0,1,1),(0,2,1)], [(2,1,1),(2,2,1)]),  # P11
        ([(0,2,1),(0,0,1)], [(0,2,1),(0,0,1)]),  # P12
        ([(1,0,1),(1,1,1)], [(1,0,1),(1,1,1)]),  # P13
        ([(1,1,1),(1,2,1)], [(2,1,1),(2,2,1)]),  # P14
        ([(1,2,1),(1,0,1)], [(0,2,1),(0,0,1)]),  # P15
        ([(2,0,1),(2,1,1)], [(1,0,1),(1,1,1)]),  # P16
        ([(2,1,1),(2,2,1)], [(2,1,1),(2,2,1)]),  # P17
        ([(2,2,1),(2,0,1)], [(0,2,1),(0,0,1)]),  # P18
        ([(0,0,1),(1,1,1)], [(1,1,1),(0,0,1)]),  # P19
        ([(0,1,1),(2,2,1)], [(2,2,1),(1,0,1)]),  # P20
        ([(0,2,1),(1,0,1)], [(0,1,1),(2,0,1)]),  # P21
        ([(0,0,1),(1,1,-1),(2,2,1)], [(0,0,1),(1,1,-1),(2,2,1)]),  # P22
        ([(0,1,1),(1,2,-1),(2,0,1)], [(1,0,1),(2,1,-1),(0,2,1)]),  # P23
    ]
    
    # For each C[i,k], we need: C[i,k] = sum_j A[i,j]*B[j,k] = sum_r w_r[i,k] * M_r
    # M_r = (sum_{i',j'} u_r[i',j'] * A[i',j']) * (sum_{j',k'} v_r[j',k'] * B[j',k'])
    #     = sum_{i',j',j'',k'} u_r[i',j'] * v_r[j'',k''] * A[i',j'] * B[j'',k'']
    # For this to contribute to C[i,k] = A[i,j]*B[j,k], we need i'=i, j''=k, and j'=j''=j
    # So the coefficient of A[i,j]*B[j,k] in M_r is: sum_j u_r[i,j] * v_r[j,k]
    
    # We need: for all i,k: sum_j delta(i,j)*delta(j,k) = sum_r w_r[i,k] * (sum_j u_r[i,j] * v_r[j,k])
    # i.e., 1 (for the specific j) = sum_r w_r[i,k] * (sum_j u_r[i,j] * v_r[j,k])
    
    # Actually, C[i,k] = sum_j A[i,j]*B[j,k], so the coefficient of A[i,j]*B[j,k] in C is 1 for each j.
    # In M_r, the coefficient of A[i,j]*B[j,k] is u_r[i,j] * v_r[j,k].
    # So we need: for all i,k,j: 1 = sum_r w_r[i,k] * u_r[i,j] * v_r[j,k]
    
    # This gives us 3*3*3 = 27 equations with 3*3*23 = 207 unknowns (w_r[i,k] for r=1..23, i=0..2, k=0..2)
    # But actually, w_r[i,k] is the same for all j, so we have 9*23 = 207 unknowns and 27*... 
    
    # Wait, let me re-derive. C[i,k] = sum_j A[i,j] * B[j,k]
    # M_r = (sum_{i',j'} u_r[i',j'] A[i',j']) * (sum_{j'',k'} v_r[j'',k''] B[j'',k''])
    #     = sum_{i',j',j'',k'} u_r[i',j'] v_r[j'',k''] A[i',j'] B[j'',k'']
    # The term A[i,j] B[j,k] appears with coefficient u_r[i,j] * v_r[j,k]
    # So C[i,k] = sum_j A[i,j] B[j,k] = sum_r w_r[i,k] M_r
    # The coefficient of A[i,j] B[j,k] on the LHS is 1 (for each j)
    # On the RHS, it's w_r[i,k] * u_r[i,j] * v_r[j,k]
    # So: 1 = sum_r w_r[i,k] * u_r[i,j] * v_r[j,k] for all i,j,k
    
    # This gives 27 equations (one for each i,j,k) with 23*9 = 207 unknowns.
    # The system is underdetermined, but we just need one solution.
    
    # Let's set up the system. Unknowns: w_r[i,k] for r=0..22, i=0..2, k=0..2
    # Equation for (i,j,k): sum_r w_r[i,k] * u_r[i,j] * v_r[j,k] = 1
    
    # We can solve this as a linear system. Let's vectorize.
    # Let x be the vector of all w_r[i,k] (207 elements)
    # Each equation gives a row of the matrix A
    
    # Actually, the equations separate by (i,k): for each (i,k), we have 3 equations (one for each j)
    # and 23 unknowns (w_0[i,k], ..., w_22[i,k])
    
    # So we can solve 9 independent 3x23 systems!
    
    W = np.zeros((3, 3, 23))  # W[i,k,r]
    
    for i in range(3):
        for k in range(3):
            # Build 3x23 system
            A_sys = np.zeros((3, 23))
            b_sys = np.ones(3)
            for j in range(3):
                for r in range(23):
                    u_coeff = 0
                    for (ii, jj, cc) in products[r][0]:
                        if ii == i and jj == j:
                            u_coeff += cc
                    v_coeff = 0
                    for (jj, kk, cc) in products[r][1]:
                        if jj == j and kk == k:
                            v_coeff += cc
                    A_sys[j, r] = u_coeff * v_coeff
            # Solve A_sys @ w = b_sys (underdetermined, use least squares)
            w, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
            W[i, k, :] = w
    
    return W
