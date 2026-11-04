import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin from 150s limit

    n, m, p = DIMENSIONS
    
    # We will construct the decomposition by combining sub-decompositions.
    # The final U, V, W matrices will be of size (n*m, rank), (m*p, rank), (p*n, rank)
    # where rank is the total number of terms.
    
    U_list = []
    V_list = []
    W_list = []

    # Helper to add a schoolbook term for specific indices
    # i: row of A (0..n-1), j: col of A / row of B (0..m-1), k: col of B (0..p-1)
    # This corresponds to C[i,k] += A[i,j] * B[j,k]
    # U row index: i*m + j
    # V row index: j*p + k
    # W row index: k*n + i
    def add_schoolbook(i, j, k):
        U_list.append([0] * (n*m))
        U_list[-1][i*m + j] = 1
        V_list.append([0] * (m*p))
        V_list[-1][j*p + k] = 1
        W_list.append([0] * (p*n))
        W_list[-1][k*n + i] = 1

    # Helper to add a Strassen term for a 2x2x2 block
    # We need to map local 2x2x2 indices to global indices
    # i_off, j_off, k_off: offsets in A, B, C
    # Strassen factors for 2x2x2:
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    # M2 = (A21+A22)B11 -> C21 += M2
    # M3 = A11(B12-B22) -> C12 += M3
    # M4 = A22(B21-B11) -> C11 += M4
    # M5 = (A11+A12)B22 -> C12 -= M5, C21 -= M5
    # M6 = (A21-A11)(B11+B12) -> C22 += M6
    # M7 = (A12-A22)(B21+B22) -> C11 -= M7, C22 -= M7
    
    # Local indices for 2x2 block:
    # A: (0,0), (0,1), (1,0), (1,1) -> Global (i0+j0, i0+j1), etc.
    # B: (0,0), (0,1), (1,0), (1,1) -> Global (j0+k0, j0+k1), etc.
    # C: (0,0), (0,1), (1,0), (1,1) -> Global (i0+k0, i0+k1), etc.
    
    def add_strassen(i_off, j_off, k_off):
        # i0, i1 = i_off, i_off+1
        # j0, j1 = j_off, j_off+1
        # k0, k1 = k_off, k_off+1
        
        i0, i1 = i_off, i_off + 1
        j0, j1 = j_off, j_off + 1
        k0, k1 = k_off, k_off + 1
        
        # We add 7 terms. Each term is a linear combination of basis tensors.
        # Instead of expanding fully into U,V,W rows, we can just add the specific contributions.
        # However, the interface requires U,V,W to be dense vectors of length rank.
        # Each "term" in the final rank decomposition is a vector in U, V, W.
        # Strassen's algorithm is a sum of 7 rank-1 tensors.
        # Rank-1 tensor T_{ijk} = u_i v_j w_k.
        # For Strassen, the "u" part is a linear combination of A elements, "v" of B, "w" of C.
        # But our decomposition is C_{ik} = sum_t U_{i,j_t} V_{j_t,k} W_{k,i}?
        # No, the standard tensor decomposition for matrix mult is:
        # C_{i,k} = sum_{j, t} A_{i,j} B_{j,k} (delta)
        # General rank decomposition:
        # C_{i,k} = sum_t U^{(t)}_{i, *} V^{(t)}_{*, k} ... 
        # Actually, the standard form for matrix multiplication tensor is:
        # T_{i,j,k} = delta_{ij} delta_{jk} ? No.
        # T_{i,j,k} corresponds to A_{i,j} B_{j,k} -> C_{i,k}.
        # The tensor has indices i (row A), j (col A / row B), k (col B).
        # Value is 1 if it's a valid multiplication step.
        # Rank decomposition: T_{i,j,k} = sum_t U_{i,j}^{(t)} V_{j,k}^{(t)} W_{k,i}^{(t)}?
        # The task says: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
        # This implies U is indexed by (i,j), V by (j,k), W by (k,i).
        # The decomposition is: 
        # sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t} = delta? No.
        # It must reconstruct the multiplication.
        # C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
        # If we have factors U,V,W such that:
        # sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t} ... this doesn't look like C.
        
        # Let's re-read the task carefully.
        # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
        # "exactly decompose the matrix multiplication tensor"
        # The matrix multiplication tensor $T$ has components $T_{i,j,k}$?
        # Usually, the tensor for $n \times m \times p$ is defined on $[n] \times [m] \times [p]$.
        # $T_{i,j,k} = 1$ if the term $A_{i,j} B_{j,k}$ contributes to $C_{i,k}$.
        # Actually, the tensor is $T_{i,j,k} = 1$ for all $i,j,k$? No.
        # The standard "multiplication tensor" is the identity of the algebra?
        # No, for matrix multiplication, the tensor is $\sum_{i,j,k} E_{i,j} \otimes E_{j,k} \otimes E_{k,i}$?
        # Let's look at the seed code:
        # U[i*m+j, col] = 1
        # V[j*p+k, col] = 1
        # W[k*n+i, col] = 1
        # This produces a sum over columns t:
        # sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t}
        # In the seed, for a specific (i,j,k), there is exactly one column t where all three are 1.
        # So sum_t ... = 1 for that (i,j,k).
        # This means the tensor being decomposed is simply the all-ones tensor on the valid indices?
        # Or is it that the "value" of the tensor at (i,j,k) is 1?
        # Yes, the "schoolbook decomposition" sets 1s such that for each (i,j,k) tuple,
        # there is a rank-1 component that is 1 at that spot.
        # This implies the target tensor is $T_{i,j,k} = 1$ for all $i,j,k$.
        # Wait, does $\sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t}$ equal 1?
        # Yes, if the columns are orthogonal in a specific way or just disjoint supports.
        # In the seed, the supports are disjoint. Each (i,j,k) has a unique column.
        # So the target tensor is indeed the "all ones" tensor of shape $n \times m \times p$.
        
        # So, we are decomposing the tensor $T_{i,j,k} = 1$.
        # We can use any rank decomposition of the all-ones tensor.
        # Strassen decomposes the *multiplication algorithm*, which is a sum of rank-1 tensors
        # that, when contracted, yield the product.
        # But here we are just decomposing the constant 1 tensor?
        # If so, we can use the rank-1 decomposition of the all-ones tensor:
        # $T_{i,j,k} = 1$.
        # Rank 1? $U_{i,j}=1, V_{j,k}=1, W_{k,i}=1$.
        # Then sum_t ... = 1 * 1 * 1 = 1.
        # Rank 1 is possible!
        
        # BUT, the task says "exactly decompose the matrix multiplication tensor".
        # Usually, "Matrix Multiplication Tensor" refers to the structure of the multiplication,
        # i.e., $C = AB$.
        # If the evaluator checks if the decomposition allows computing $C=AB$,
        # then the "tensor" is the one defined by the multiplication rule.
        # However, the seed code produces a decomposition where $U,V,W$ are sparse 0/1 matrices.
        # If we use Rank 1 (all ones), does it compute $AB$?
        # $C'_{i,k} = \sum_j (\sum_t U_{i,j,t} A_{i,j}) (\sum_t V_{j,k,t} B_{j,k}) ...$?
        # No, the standard way to use such a decomposition is:
        # Compute $M_t = \sum_{i,j} U_{i,j,t} A_{i,j}$
        # Compute $N_t = \sum_{j,k} V_{j,k,t} B_{j,k}$
        # Compute $O_t = \sum_{k,i} W_{k,i,t} M_t N_t$?
        # Then $C_{i,k} = \sum_t O_t$?
        
        # Let's check the seed's validity.
        # Seed: Rank 30.
        # If the target was just "all ones", Rank 1 would be valid.
        # Since the seed is Rank 30 (n*m*p), it implies that we cannot simply use Rank 1.
        # Why? Because the decomposition must support the *linear combination* to form $C$.
        # The "Matrix Multiplication Tensor" is likely defined as:
        # $T_{i,j,k}$ such that $\langle T, A \otimes B \rangle = \text{something related to } C$.
        # Actually, the standard definition of the rank of matrix multiplication $R(n,m,p)$
        # is the minimal number of scalar multiplications needed.
        # This corresponds to the rank of the tensor $\mathcal{M}_{n,m,p}$ defined by
        # $\mathcal{M}_{n,m,p} = \sum_{i,j,k} E_{i,j} \otimes E_{j,k} \otimes E_{k,i}$.
        # Note the indices: $E_{i,j}$ (A), $E_{j,k}$ (B), $E_{k,i}$ (C).
        # The seed code uses:
        # U index: $i*m+j$ (corresponds to $E_{i,j}$)
        # V index: $j*p+k$ (corresponds to $E_{j,k}$)
        # W index: $k*n+i$ (corresponds to $E_{k,i}$)
        # And it sets them to 1 for the matching $(i,j,k)$.
        # This means the seed is decomposing the tensor:
        # $T = \sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}$.
        # This is EXACTLY the matrix multiplication tensor.
        
        # So, we need to decompose $T = \sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}$.
        # The seed does this with rank $nmp$ (one term per $(i,j,k)$).
        # We can do better using Strassen-like block decompositions.
        # Strassen reduces the number of terms for $2 \times 2 \times 2$ from 8 to 7.
        # The "terms" in Strassen are linear combinations of the basis tensors.
        # Specifically, for $2 \times 2 \times 2$:
        # The 7 terms correspond to the 7 rank-1 tensors in the Strassen algorithm.
        # We need to generate the U, V, W rows for these 7 terms.
        
        # Let's implement the Strassen terms explicitly for a $2 \times 2 \times 2$ block.
        # Local indices $u,v,w \in \{0,1\}$.
        # Global indices:
        # A: $i = i\_off + u, j = j\_off + v$
        # B: $j = j\_off + v, k = k\_off + w$
        # C: $k = k\_off + w, i = i\_off + u$
        
        # The 7 Strassen products $M_1 \dots M_7$ are linear combinations of $A$ and $B$ elements.
        # $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$
        # This expands to $A_{00}B_{00} + A_{00}B_{11} + A_{11}B_{00} + A_{11}B_{11}$.
        # In the tensor decomposition, each product $A_{i,j}B_{j',k}$ contributes to $C_{i,k}$ via $W$.
        # Wait, the tensor $T$ has fixed indices.
        # The term $e_{i,j} \otimes e_{j,k} \otimes e_{k,i}$ corresponds to the scalar multiplication $A_{i,j} B_{j,k}$.
        # Strassen groups these.
        # For example, the term for $M_1$ in the decomposition of $T$ is not a single rank-1 tensor.
        # It is a linear combination of 4 rank-1 tensors.
        # But Strassen says we only need 7 scalar multiplications.
        # This means the rank of the tensor is 7.
        # The decomposition consists of 7 rank-1 tensors $R_1 \dots R_7$.
        # Each $R_t$ is of the form $u_t \otimes v_t \otimes w_t$.
        # We need to find $u_t, v_t, w_t$ for the 7 terms.
        
        # Standard Strassen Rank-1 Factors for 2x2:
        # Let $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$, $B = \begin{pmatrix} e & f \\ g & h \end{pmatrix}$.
        # $M_1 = (a+d)(e+h)$. 
        #   $u_1 = e_{0,0} + e_{1,1}$ (coeffs: $a:1, d:1$)
        #   $v_1 = e_{0,0} + e_{0,1}$? No. $B$ indices are $(0,0),(0,1),(1,0),(1,1)$.
        #   $v_1$ corresponds to $B$ part: $e+h = B_{0,0} + B_{1,1}$.
        #   So $v_1$ has 1 at $(0,0)$ and $(1,1)$.
        #   $w_1$ corresponds to $C$ part: $C_{0,0} + C_{1,1}$.
        #   So $w_1$ has 1 at $(0,0)$ and $(1,1)$.
        
        # $M_2 = (c+d)e$.
        #   $u_2 = c+d \rightarrow e_{1,0} + e_{1,1}$.
        #   $v_2 = e \rightarrow e_{0,0}$.
        #   $w_2 = C_{1,0} \rightarrow e_{0,1}$ (since $C_{k,i}$, $k=0, i=1 \rightarrow$ index $0 \times 2 + 1 = 1$? 
        #   Wait, $W$ index is $k \times n + i$.
        #   For $C_{1,0}$ ($i=1, k=0$), index is $0 \times 2 + 1 = 1$.
        
        # We will generate these 7 terms programmatically for any $2 \times 2 \times 2$ block.
        
        def add_strassen_2x2(i_off, j_off, k_off):
            # Returns 7 tuples of (u_vec, v_vec, w_vec)
            terms = []
            
            # 1. (a+d)(e+h) -> C00, C11
            u = [0]*4; u[i_off*2+j_off] = 1; u[i_off*2+j_off+2] = 1 # a, d (assuming row major 2x2: 00,01,10,11)
            # Wait, U size is n*m. For 2x2 block, local size 4.
            # We are appending to global lists, so we need global indices.
            # Let's define local-to-global mapping helpers inside the function scope or pass them.
            
            # It's easier to construct the local vectors and then scatter them.
            # But we are adding to global U, V, W.
            # We will create sparse-like local vectors and then add to global.
            
            def make_global(vec_local, offset, dim0, dim1):
                # vec_local is list of length dim0*dim1
                # offset is (start_i, start_j) for A, (start_j, start_k) for B, etc.
                # We return a list of indices and values?
                # No, we just append to U_list, V_list, W_list.
                # We need to add to the existing lists.
                pass
            
            # Let's just hardcode the 7 terms for 2x2.
            # Local indices for U (A): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Local indices for V (B): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Local indices for W (C): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -- Note: W index is k*n+i.
            # For 2x2, k,n=2.
            # W local 0: k=0, i=0 -> (0,0)
            # W local 1: k=0, i=1 -> (0,1) in C? No, C index is (i,k).
            # Tensor index is (i,j,k). W component is $W_{k,i}$.
            # So $W_{0,0}$ corresponds to $C_{0,0}$.
            # $W_{0,1}$ corresponds to $C_{1,0}$?
            # $W_{k \cdot n + i}$.
            # If $k=0, i=1$, index 1. This is $C_{1,0}$.
            # If $k=1, i=0$, index 2. This is $C_{0,1}$.
            # If $k=1, i=1$, index 3. This is $C_{1,1}$.
            
            # Strassen Terms:
            # 1. M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
            #    U1: A00, A11 (1,0,0,1)
            #    V1: B00, B11 (1,0,0,1)
            #    W1: C00, C11 (1,0,0,1)
            U1 = [0]*4; U1[0]=1; U1[3]=1
            V1 = [0]*4; V1[0]=1; V1[3]=1
            W1 = [0]*4; W1[0]=1; W1[3]=1
            terms.append((U1, V1, W1))
            
            # 2. M2 = (A10+A11)B00 -> C10 += M2
            #    U2: A10, A11 (0,0,1,1)
            #    V2: B00 (1,0,0,0)
            #    W2: C10 (W index for k=0, i=1 is 1) -> (0,0,1,0)
            U2 = [0]*4; U2[2]=1; U2[3]=1
            V2 = [0]*4; V2[0]=1
            W2 = [0]*4; W2[1]=1
            terms.append((U2, V2, W2))
            
            # 3. M3 = A00(B01-B11) -> C01 += M3
            #    U3: A00 (1,0,0,0)
            #    V3: B01, B11 (0,1,0,-1)
            #    W3: C01 (W index for k=1, i=0 is 2) -> (0,0,1,0)
            U3 = [0]*4; U3[0]=1
            V3 = [0]*4; V3[1]=1; V3[3]=-1
            W3 = [0]*4; W3[2]=1
            terms.append((U3, V3, W3))
            
            # 4. M4 = A11(B10-B00) -> C00 += M4
            #    U4: A11 (0,0,0,1)
            #    V4: B10, B00 (0,0,1,-1)
            #    W4: C00 (1,0,0,0)
            U4 = [0]*4; U4[3]=1
            V4 = [0]*4; V4[2]=1; V4[0]=-1
            W4 = [0]*4; W4[0]=1
            terms.append((U4, V4, W4))
            
            # 5. M5 = (A00+A01)B11 -> C01 -= M5, C10 -= M5
            #    U5: A00, A01 (1,1,0,0)
            #    V5: B11 (0,0,0,1)
            #    W5: C01 (2), C10 (1) -> (0,-1,1,0)
            U5 = [0]*4; U5[0]=1; U5[1]=1
            V5 = [0]*4; V5[3]=1
            W5 = [0]*4; W5[1]=-1; W5[2]=1
            terms.append((U5, V5, W5))
            
            # 6. M6 = (A10-A00)(B00+B01) -> C11 += M6
            #    U6: A10, A00 (0,0,1,-1)
            #    V6: B00, B01 (1,1,0,0)
            #    W6: C11 (0,0,0,1)
            U6 = [0]*4; U6[2]=1; U6[0]=-1
            V6 = [0]*4; V6[0]=1; V6[1]=1
            W6 = [0]*4; W6[3]=1
            terms.append((U6, V6, W6))
            
            # 7. M7 = (A01-A11)(B10+B11) -> C00 -= M7, C11 -= M7
            #    U7: A01, A11 (0,1,0,-1)
            #    V7: B10, B11 (0,0,1,1)
            #    W7: C00, C11 (0,-1,0,-1)
            U7 = [0]*4; U7[1]=1; U7[3]=-1
            V7 = [0]*4; V7[2]=1; V7[3]=1
            W7 = [0]*4; W7[0]=-1; W7[3]=-1
            terms.append((U7, V7, W7))
            
            # Scatter these local vectors to global U, V, W lists
            for u_l, v_l, w_l in terms:
                u_g = [0] * (n*m)
                v_g = [0] * (m*p)
                w_g = [0] * (p*n)
                
                for idx_u, val in enumerate(u_l):
                    if val != 0:
                        i_loc = idx_u // 2
                        j_loc = idx_u % 2
                        i_glob = i_off + i_loc
                        j_glob = j_off + j_loc
                        u_g[i_glob * m + j_glob] = val
                
                for idx_v, val in enumerate(v_l):
                    if val != 0:
                        j_loc = idx_v // 2
                        k_loc = idx_v % 2
                        j_glob = j_off + j_loc
                        k_glob = k_off + k_loc
                        v_g[j_glob * p + k_glob] = val
                
                for idx_w, val in enumerate(w_l):
                    if val != 0:
                        k_loc = idx_w // 2
                        i_loc = idx_w % 2
                        k_glob = k_off + k_loc
                        i_glob = i_off + i_loc
                        w_g[k_glob * n + i_glob] = val
                
                U_list.append(u_g)
                V_list.append(v_g)
                W_list.append(w_g)

    # Strategy:
    # Split B (3x5) into B_L (3x2) and B_R (3x3).
    # A (2x3) is split into A_L (2x2) and A_R (2x1).
    
    # 1. A_L (2x2) * B_L_L (2x2) part of B_L.
    #    B_L is 3x2. Split B_L into B_LL (2x2) and B_LR (1x2).
    #    A_L (2x2) * B_LL (2x2) -> Strassen.
    #    i_off=0, j_off=0, k_off=0.
    if time.time() < deadline:
        add_strassen_2x2(0, 0, 0)
        
    # 2. A_R (2x1) * B_LR (1x2).
    #    Schoolbook.
    #    A_R rows i=0,1. Col j=2.
    #    B_LR row j=2. Cols k=0,1.
    for i in range(2):
        for k in range(2):
            if time.time() < deadline:
                add_schoolbook(i, 2, k) # i=0,1; j=2; k=0,1
                
    # 3. A_L (2x2) * B_R_L (2x3) part of B_R.
    #    B_R is 3x3. Split B_R into B_RL (2x3) and B_RR (1x3).
    #    A_L (2x2) * B_RL (2x3).
    #    Split B_RL into B_RL_L (2x2) and B_RL_R (2x1).
    #    A_L (2x2) * B_RL_L (2x2) -> Strassen.
    #    i_off=0, j_off=0, k_off=2.
    if time.time() < deadline:
        add_strassen_2x2(0, 0, 2)
        
    # 4. A_L (2x2) * B_RL_R (2x1).
    #    Schoolbook.
    #    A_L i=0,1. j=0,1.
    #    B_RL_R j=0,1. k=4.
    for i in range(2):
        for j in range(2):
            if time.time() < deadline:
                add_schoolbook(i, j, 4)
                
    # 5. A_R (2x1) * B_RR (1x3).
    #    Schoolbook.
    #    A_R i=0,1. j=2.
    #    B_RR j=2. k=2,3,4.
    for i in range(2):
        for k in range(2, 5):
            if time.time() < deadline:
                add_schoolbook(i, 2, k)

    # Convert lists to numpy arrays or keep as lists?
    # Task says "integer/half-integer entries". Lists are fine.
    # Ensure shapes are correct.
    U = np.array(U_list).T if U_list else np.empty((n*m, 0))
    V = np.array(V_list).T if V_list else np.empty((m*p, 0))
    W = np.array(W_list).T if W_list else np.empty((p*n, 0))
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
