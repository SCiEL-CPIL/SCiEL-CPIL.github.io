"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Known rank for 2x3x6 matrix multiplication is 20
    rank = 20
    
    # We'll solve for U, V, W using the following approach:
    # Fix U to have a specific sparse structure, then solve for V and W
    
    # Let's use a constructive approach based on the following idea:
    # For each k, we have 3 j-values and 2 i-values.
    # We can use 3 terms per k for a "base" decomposition (9 terms total for 3 k-groups)
    # plus 11 correction terms.
    
    # Actually, let me try to find a valid rank-20 decomposition numerically.
    # Set up the system: for each (i,j,k), we need sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1
    
    # Let's parameterize: fix U to be a 6x20 matrix with specific entries,
    # then for each c, v_c and w_c must satisfy linear constraints.
    
    # Alternative: use the fact that if we fix U and V, then W is determined by:
    # For each c and each (k,i): w_c[k*n+i] = sum over... no, it's not linear in W alone.
    
    # Let me try a different parameterization. Fix U and W, solve for V.
    # For each c: v_c[j*p+k] appears in equations for all i.
    # sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1
    
    # For fixed j,k: sum_c (u_c[i*m+j] * w_c[k*n+i]) * v_c[j*p+k] = 1 for all i
    # This is a system in v_c[j*p+k] for c=0..19, with 2 equations (i=0,1) and 20 unknowns.
    # Underdetermined, so we can choose freely.
    
    # Let me just try to construct a valid decomposition by trial.
    
    # Here's a known construction for rank 20:
    # Use 6 terms of the form: for each k, one term where U has 1s in positions (0,j) and (1,j) for all j,
    # V has 1 in position (j,k) for all j, and W has 1 in positions (k,0) and (k,1).
    # But that gives the sum, not individual products.
    
    # Let me just use a numerical solver. I'll set up the bilinear system and solve it.
    
    # Variables: U (6x20), V (18x20), W (12x20) = 6*20 + 18*20 + 12*20 = 120 + 360 + 240 = 720 variables
    # Equations: 2*3*6 = 36 equations
    # The equations are cubic in the variables, so this is nonlinear.
    
    # Let me try a simpler approach: fix U to be sparse, then the equations become bilinear in V and W.
    
    # Let U be a 6x20 matrix where each row has at most 2 non-zeros.
    # Let's try U with the following structure:
    # Rows 0-2 (i=0, j=0,1,2): non-zeros in columns 0,1,2,3
    # Rows 3-5 (i=1, j=0,1,2): non-zeros in columns 4,5,6,7
    # Plus 12 more columns for corrections.
    
    # This is getting too complex. Let me just verify that rank 20 is achievable
    # by providing a known valid decomposition.
    
    # Based on literature, the rank of 2x3x6 is 20. Let me construct it explicitly.
    
    # Construction using the following 20 terms:
    # Terms 0-5: For each k in 0..5, term that contributes to C[0,k] and C[1,k]
    #   U: row i*m+j = 1 for all i,j (i.e., all 6 entries are 1)
    #   V: row j*p+k = 1 for all j (i.e., V[0*6+k]=V[1*6+k]=V[2*6+k]=1)
    #   W: row k*n+i = 1 for all i (i.e., W[k*2+0]=W[k*2+1]=1)
    #   This gives sum_j A[i,j]*B[j,k] for both i=0 and i=1, but with the same coefficient.
    #   Actually this gives the correct product! But it uses the same V and W for both i values.
    #   Wait, this means C[0,k] = sum_j A[0,j]*B[j,k] and C[1,k] = sum_j A[1,j]*B[j,k]
    #   which is exactly what we want! So 6 terms suffice? No, because U[i*m+j] is the same for both i.
    #   U[0*m+j] = U[1*m+j] = 1, so the term contributes A[0,j]*B[j,k] to C[0,k] and A[1,j]*B[j,k] to C[1,k].
    #   That's correct! So 6 terms work?
    
    # Let me verify: for term c=k, U[i*3+j] = 1 for all i,j, V[j*6+k] = 1 for all j, W[k*2+i] = 1 for all i.
    # Contribution to C[i',k'] = sum_j sum_c U[i'*3+j,c] * V[j*6+k',c] * W[k'*2+i',c]
    # = sum_j U[i'*3+j,k'] * V[j*6+k',k'] * W[k'*2+i',k']
    # = sum_j 1 * 1 * 1 = 3 (for k'=k)
    # That's 3, not 1. So we need to scale by 1/3, but we need integer/half-integer entries.
    
    # So that doesn't work directly. Let me think again.
    
    # The issue is that with U[i*m+j] = 1 for all i,j, we're summing over j without distinguishing which j.
    # We need U[i*m+j] to depend on j properly.
    
    # Let me go back to the schoolbook and see if I can reduce it.
    # Schoolbook: 36 terms, one per (i,j,k).
    # Can I combine terms that share the same (j,k) but different i?
    # For fixed (j,k), I have two terms: one for i=0, one for i=1.
    # Term for i=0: U[0*m+j]=1, V[j*p+k]=1, W[k*n+0]=1
    # Term for i=1: U[1*m+j]=1, V[j*p+k]=1, W[k*n+1]=1
    # Can I replace these with fewer terms? I need:
    #   sum_c' U'[0*m+j,c'] * V'[j*p+k,c'] * W'[k*n+0,c'] = 1
    #   sum_c' U'[1*m+j,c'] * V'[j*p+k,c'] * W'[k*n+1,c'] = 1
    #   And for other (i',k'): contributions should be 0.
    
    # If I use one term with U'[0*m+j]=a, U'[1*m+j]=b, V'[j*p+k]=c, W'[k*n+0]=d, W'[k*n+1]=e,
    # then I need a*c*d = 1 and b*c*e = 1, and for other combinations, the product should be 0
    # when summed with other terms.
    
    # This is complex. Let me just provide the schoolbook decomposition which is correct.
    
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
# EVOLVE-BLOCK-END
