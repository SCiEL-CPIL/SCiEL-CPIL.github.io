"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-7 tensor decomposition for 2x2 matrix multiplication via Strassen.
    
    Returns factor matrices U (4x7), V (4x7), W (4x7) with integer entries
    such that for all i,j,k in {0,1}:
      sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] == 1
    
    The rank 7 is provably optimal (Bini et al., 1979).
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    # Define the 7 rank-1 tensors as sparse coefficient lists
    # Each entry: (u_indices, v_indices, w_indices, coeffs)
    # U index: i*m+j, V index: j*p+k, W index: k*n+i
    #
    # Seven products:
    #  P0 = (a00+a11)(b00+b11)
    #  P1 = (a10+a11)(b00)
    #  P2 = (a00)(b01-b11)
    #  P3 = (a11)(b10-b00)
    #  P4 = (a00+a01)(b11)
    #  P5 = (a10-a00)(b00+b01)
    #  P6 = (a01-a11)(b10+b11)
    #
    # C reconstruction:
    #  c00 = P0 - P4 + P3 + P6
    #  c01 = P2 + P4
    #  c10 = P1 + P3
    #  c11 = P0 - P1 + P2 + P5
    
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # P0: u={0,3}, v={0,3}, w={0,3}
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=1
    
    # P1: u={2,3}, v={0}, w={1,3}
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[1][1]=1; W[3][1]=-1
    
    # P2: u={0}, v={1,3}, w={2,3}
    U[0][2]=1
    V[1][2]=1; V[3][2]=-1
    W[2][2]=1; W[3][2]=1
    
    # P3: u={3}, v={0,2}, w={0,1}
    U[3][3]=1
    V[0][3]=-1; V[2][3]=1
    W[0][3]=1; W[1][3]=1
    
    # P4: u={0,1}, v={3}, w={0,2}
    U[0][4]=1; U[1][4]=1
    V[3][4]=1
    W[0][4]=-1; W[2][4]=1
    
    # P5: u={0,2}, v={0,1}, w={3}
    U[0][5]=-1; U[2][5]=1
    V[0][5]=1; V[1][5]=1
    W[3][5]=1
    
    # P6: u={1,3}, v={2,3}, w={0}
    U[1][6]=1; U[3][6]=-1
    V[2][6]=1; V[3][6]=1
    W[0][6]=1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
