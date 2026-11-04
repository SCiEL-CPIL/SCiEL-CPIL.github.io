import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = DIMENSIONS  # 4, 4, 8
    nm = n * m   # 16
    mp = m * p   # 32
    pn = p * n   # 32
    
    # Build rank-49 decomposition for 4x4x4
    # U2, V2, W2 for 2x2x2 Strassen
    U2 = np.zeros((4, 7)); V2 = np.zeros((4, 7)); W2 = np.zeros((4, 7))
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[2,1]=1; W2[3,1]=-1
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[1,2]=1; W2[3,2]=1
    U2[3,3]=1; V2[2,3]=1; V2[0,3]=-1; W2[0,3]=1; W2[2,3]=1
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1; W2[1,4]=1
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1; W2[3,5]=1
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1

    A_specs = [
        [(1,0,0),(1,1,1)],
        [(1,1,0),(1,1,1)],
        [(1,0,0)],
        [(1,1,1)],
        [(1,0,0),(1,0,1)],
        [(1,1,0),(-1,0,0)],
        [(1,0,1),(-1,1,1)],
    ]
    B_specs = [
        [(1,0,0),(1,1,1)],
        [(1,0,0)],
        [(1,0,1),(-1,1,1)],
        [(1,1,0),(-1,0,0)],
        [(1,1,1)],
        [(1,0,0),(1,0,1)],
        [(1,1,0),(1,1,1)],
    ]
    C_specs = [
        {(0,0):1,(1,1):1},
        {(1,0):1,(1,1):-1},
        {(0,1):1,(1,1):1},
        {(0,0):1,(1,0):1},
        {(0,0):-1,(0,1):1},
        {(1,1):1},
        {(0,0):1},
    ]

    # Build U4, V4, W4 for 4x4x4
    U4_list = []
    V4_list = []
    W4_list = []
    
    for kp in range(7):
        for ms in range(7):
            u_dict = {}
            v_dict = {}
            w_dict = {}
            for (coeff, br, bc) in A_specs[kp]:
                for r in range(2):
                    for c in range(2):
                        if U2[r*2+c, ms] != 0:
                            gi = br * 2 + r
                            gj = bc * 2 + c
                            u_dict[(gi, gj)] = u_dict.get((gi, gj), 0) + coeff * U2[r*2+c, ms]
            for (coeff, br, bc) in B_specs[kp]:
                for r in range(2):
                    for c in range(2):
                        if V2[r*2+c, ms] != 0:
                            gj = br * 2 + r
                            gk = bc * 2 + c
                            v_dict[(gj, gk)] = v_dict.get((gj, gk), 0) + coeff * V2[r*2+c, ms]
            for (cbr, cbc), coeff_c in C_specs[kp].items():
                for r in range(2):
                    for c in range(2):
                        if W2[r*2+c, ms] != 0:
                            gi = cbr * 2 + r
                            gk = cbc * 2 + c
                            w_dict[(gk, gi)] = w_dict.get((gk, gi), 0) + coeff_c * W2[r*2+c, ms]
            
            U4_row = np.zeros(16)
            V4_row = np.zeros(16)
            W4_row = np.zeros(16)
            
            for (i, j), val in u_dict.items():
                U4_row[i*4 + j] = val
            for (j, k), val in v_dict.items():
                V4_row[j*4 + k] = val
            for (k, i), val in w_dict.items():
                W4_row[k*4 + i] = val
                
            U4_list.append(U4_row)
            V4_list.append(V4_row)
            W4_list.append(W4_row)

    U4 = np.array(U4_list).T # Shape (16, 49)
    V4 = np.array(V4_list).T # Shape (16, 49)
    W4 = np.array(W4_list).T # Shape (16, 49)
    
    rank = 49
    
    # Construct U, V, W for 4x4x8
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    U[:] = U4
    
    # V: index j*8 + k
    # For k=0..3: V[j, k] = V4[j, k] (V4 index j*4+k)
    # For k=4..7: V[j, 4+l] = V4[j, l] (V4 index j*4+l)
    for j in range(4):
        for k in range(4):
            V[j*8 + k, :] = V4[j*4 + k, :]
            V[j*8 + 4 + k, :] = V4[j*4 + k, :]
            
    # W: index k*4 + i
    # For k=0..3: W[k, i] = W4[k, i] (W4 index k*4+i)
    # For k=4..7: W[4+l, i] = W4[l, i] (W4 index l*4+i)
    for k in range(4):
        for i in range(4):
            W[k*4 + i, :] = W4[k*4 + i, :]
            W[(4+k)*4 + i, :] = W4[k*4 + i, :]
            
    # Verify
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # Expected T:
    # T[i,j,k] = 1 if i == k mod 4? No.
    # A is identity? No, A is arbitrary.
    # The tensor T represents the multiplication A*B.
    # T[i,j,k] = delta_{i, k} is WRONG.
    # T[i,j,k] is the coefficient of A_{ij} B_{jk} in C_{ik}.
    # So T[i,j,k] = 1 if the indices match the multiplication structure.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # So T[i,j,k] = 1 for all i,j,k? No.
    # The tensor is T_{ijk} = 1 if it contributes to C_{ik}.
    # Actually, the tensor M_{ijk} = 1 for all i,j,k is NOT the multiplication tensor.
    # The multiplication tensor is T_{ijk} = 1 if i,j,k are valid indices?
    # No, the tensor is defined by T(i,j,k) = 1 if we are computing C_{ik} from A_{ij} and B_{jk}.
    # Wait, the standard multiplication tensor for n x n x n is T_{ijk} = 1 if i,j,k are in range.
    # No, that's the "all ones" tensor.
    # The multiplication tensor is T_{ijk} = \delta_{i, ?} \delta_{j, ?}?
    # Let's check the definition in the task: "exactly decompose the matrix multiplication tensor".
    # The multiplication tensor for A (n x m) * B (m x p) -> C (n x p) is:
    # T_{i, j, k} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # This is ALWAYS true for valid i,j,k.
    # So T_{i,j,k} = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # Wait, if T is all ones, then sum U V W = 1 for all i,j,k.
    # Let's verify with 2x2x2 Strassen.
    # T_{ijk} = 1 for all i,j,k in 0..1.
    # sum_r U_{ir} V_{jr} W_{kr} = 1.
    # Let's check r=0: U_{00}=1, V_{00}=1, W_{00}=1.
    # If i=0, j=0, k=0: 1*1*1 = 1.
    # If i=0, j=0, k=1: 1*1*0 = 0.
    # Sum over r must be 1.
    # So yes, the target tensor is ALL ONES.
    
    error = np.max(np.abs(T - 1.0))
    
    if error > 1e-8:
        # Fallback to schoolbook
        U_school = np.zeros((nm, 128))
        V_school = np.zeros((mp, 128))
        W_school = np.zeros((pn, 128))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m + j, col] = 1
                    V_school[j*p + k, col] = 1
                    W_school[k*n + i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
