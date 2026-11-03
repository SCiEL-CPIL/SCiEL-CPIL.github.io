import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Strassen for (2,2,2)
    # U1: 4x7, V1: 4x7, W1: 4x7
    U1 = np.zeros((4, 7))
    V1 = np.zeros((4, 7))
    W1 = np.zeros((4, 7))
    
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # W indices (k*2+i): (0,0)=0, (1,0)=1, (0,1)=2, (1,1)=3
    
    # M1: (a+d)(e+h)
    U1[0,0]=1; U1[3,0]=1
    V1[0,0]=1; V1[3,0]=1
    W1[0,0]=1; W1[3,0]=1
    
    # M2: (c+d)e
    U1[2,1]=1; U1[3,1]=1
    V1[0,1]=1
    W1[1,1]=1
    
    # M3: a(f-h)
    U1[0,2]=1
    V1[1,2]=1; V1[3,2]=-1
    W1[2,2]=1
    
    # M4: d(g-e)
    U1[3,3]=1
    V1[2,3]=1; V1[0,3]=-1
    W1[0,3]=1; W1[1,3]=1
    
    # M5: (a+b)h
    U1[0,4]=1; U1[1,4]=1
    V1[3,4]=1
    W1[0,4]=-1; W1[2,4]=1
    
    # M6: (c-a)(e+f)
    U1[2,5]=1; U1[0,5]=-1
    V1[0,5]=1; V1[1,5]=1
    W1[3,5]=1
    
    # M7: (b-d)(g+h)
    U1[1,6]=1; U1[3,6]=-1
    V1[2,6]=1; V1[3,6]=1
    W1[0,6]=1
    
    # Schoolbook for (1,2,3)
    # U2: 2x6, V2: 6x6, W2: 3x6
    U2 = np.zeros((2, 6))
    V2 = np.zeros((6, 6))
    W2 = np.zeros((3, 6))
    
    # A2 indices: (0,0)=0, (0,1)=1
    # B2 indices: (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5
    # W2 indices (k*1+0): k=0->0, k=1->1, k=2->2
    
    # P1: a*e
    U2[0,0]=1
    V2[0,0]=1
    W2[0,0]=1
    
    # P2: a*f
    U2[0,1]=1
    V2[1,1]=1
    W2[1,1]=1
    
    # P3: a*g
    U2[0,2]=1
    V2[2,2]=1
    W2[2,2]=1
    
    # P4: b*h
    U2[1,3]=1
    V2[3,3]=1
    W2[0,3]=1
    
    # P5: b*i
    U2[1,4]=1
    V2[4,4]=1
    W2[1,4]=1
    
    # P6: b*j
    U2[1,5]=1
    V2[5,5]=1
    W2[2,5]=1
    
    # Kronecker product
    U = np.kron(U1, U2)  # (4*2) x (7*6) = 8 x 42
    V = np.kron(V1, V2)  # (4*6) x (7*6) = 24 x 42
    W = np.kron(W1, W2)  # (4*3) x (7*6) = 12 x 42
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
