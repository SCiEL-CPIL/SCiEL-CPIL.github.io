import numpy as np

def construct():
    n, m, p = 3, 3, 7
    
    # Rank 23 decomposition for 3x3x3
    # We will build U3, V3, W3 of shape (9, 23), (9, 23), (9, 23)
    # U3 row index: i*3 + j
    # V3 row index: j*3 + k
    # W3 row index: k*3 + i
    
    U3 = np.zeros((9, 23))
    V3 = np.zeros((9, 23))
    W3 = np.zeros((9, 23))
    
    # 1. Strassen 7 terms for core 2x2 (indices 0,1)
    # A_core = [[a, b], [d, e]] -> indices (0,0),(0,1),(1,0),(1,1)
    # B_core = [[j, k], [m, n]] -> indices (0,0),(0,1),(1,0),(1,1)
    
    # M1 = (a+e)(j+n)
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (1,1)=1
    # W: (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1  -- Wait, Strassen output is sum of terms.
    # C00 = M1 - M4 - M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 - M5 + M6
    
    # Let's assign column indices 0-6 for these 7 multiplications.
    
    # M1
    U3[0, 0] = 1; U3[4, 0] = 1  # a, e
    V3[0, 0] = 1; V3[4, 0] = 1  # j, n
    W3[0, 0] = 1; W3[1, 0] = 1; W3[3, 0] = 1; W3[4, 0] = 1 # C00, C01, C10, C11
    
    # M2 = (d+e)j
    U3[3, 1] = 1; U3[4, 1] = 1  # d, e
    V3[0, 1] = 1                # j
    W3[3, 1] = 1                # C10
    
    # M3 = a(k-n)
    U3[0, 2] = 1                # a
    V3[1, 2] = 1; V3[4, 2] = -1 # k, -n
    W3[1, 2] = 1                # C01
    
    # M4 = e(m-j)
    U3[4, 3] = 1                # e
    V3[3, 3] = 1; V3[0, 3] = -1 # m, -j
    W3[0, 3] = -1; W3[3, 3] = 1 # -C00, C10
    
    # M5 = (a+b)n
    U3[0, 4] = 1; U3[1, 4] = 1  # a, b
    V3[4, 4] = 1                # n
    W3[1, 4] = 1; W3[4, 4] = -1 # C01, -C11
    
    # M6 = (d-a)(j+k)
    U3[3, 5] = 1; U3[0, 5] = -1 # d, -a
    V3[0, 5] = 1; V3[1, 5] = 1  # j, k
    W3[0, 5] = -1; W3[3, 5] = -1 # -C00, -C10
    
    # M7 = (b-e)(m+n)
    U3[1, 6] = 1; U3[4, 6] = -1 # b, -e
    V3[3, 6] = 1; V3[4, 6] = 1  # m, n
    W3[0, 6] = 1; W3[4, 6] = 1  # C00, C11

    # 2. Cross terms involving index 2
    # C02 = a*l + b*o + c*r
    # C12 = d*l + e*o + f*r
    # C20 = g*j + h*m + i*p
    # C21 = g*k + h*n + i*q
    # C22 = g*l + h*o + i*r
    
    # We need to decompose these 5 rows of C (each with 3 terms) into rank 1 tensors.
    # Total 15 terms?
    # C02: 3 terms
    # C12: 3 terms
    # C20: 3 terms
    # C21: 3 terms
    # C22: 3 terms
    # Total 15 terms.
    # 7 + 15 = 22 terms.
    # Wait, the rank is 23. Where is the 23rd term?
    # Maybe some terms can be merged or my count is off.
    # Actually, the outer products like $v s^T$ have rank 4, not 2.
    # $v = \begin{pmatrix} c \\ f \end{pmatrix}$, $s^T = \begin{pmatrix} p & q \end{pmatrix}$.
    # $v s^T = \begin{pmatrix} cp & cq \\ fp & fq \end{pmatrix}$.
    # This contributes to C00, C01, C10, C11.
    # This is 4 terms.
    # $A_{core} u$: $2 \times 2$ times $2 \times 1$.
    # $\begin{pmatrix} a & b \\ d & e \end{pmatrix} \begin{pmatrix} l \\ o \end{pmatrix} = \begin{pmatrix} al+bo \\ dl+eo \end{pmatrix}$.
    # 4 terms.
    # $w^T B_{core}$: $1 \times 2$ times $2 \times 2$.
    # $\begin{pmatrix} g & h \end{pmatrix} \begin{pmatrix} j & k \\ m & n \end{pmatrix} = \begin{pmatrix} gj+hm & gk+hn \end{pmatrix}$.
    # 4 terms.
    # $z s^T$: $i \begin{pmatrix} p & q \end{pmatrix} = \begin{pmatrix} ip & iq \end{pmatrix}$.
    # 2 terms.
    # $w^T u$: $\begin{pmatrix} g & h \end{pmatrix} \begin{pmatrix} l \\ o \end{pmatrix} = gl+ho$.
    # 1 term.
    # $z t$: $ir$.
    # 1 term.
    
    # Total cross terms: 4 + 4 + 4 + 2 + 1 + 1 = 16 terms.
    # 7 (Strassen) + 16 (Cross) = 23 terms.
    # Perfect.
    
    # Let's add these 16 terms to U3, V3, W3 starting at column 7.
    
    # v s^T (4 terms):
    # C00 += c*p
    U3[2, 7] = 1; V3[6, 7] = 1; W3[0, 7] = 1 # c, p, C00
    # C01 += c*q
    U3[2, 8] = 1; V3[7, 8] = 1; W3[1, 8] = 1 # c, q, C01
    # C10 += f*p
    U3[5, 9] = 1; V3[6, 9] = 1; W3[3, 9] = 1 # f, p, C10
    # C11 += f*q
    U3[5, 10] = 1; V3[7, 10] = 1; W3[4, 10] = 1 # f, q, C11
    
    # A_core u (4 terms):
    # C02 += a*l
    U3[0, 11] = 1; V3[2, 11] = 1; W3[6, 11] = 1 # a, l, C02
    # C02 += b*o
    U3[1, 12] = 1; V3[5, 12] = 1; W3[6, 12] = 1 # b, o, C02
    # C12 += d*l
    U3[3, 13] = 1; V3[2, 13] = 1; W3[7, 13] = 1 # d, l, C12
    # C12 += e*o
    U3[4, 14] = 1; V3[5, 14] = 1; W3[7, 14] = 1 # e, o, C12
    
    # w^T B_core (4 terms):
    # C20 += g*j
    U3[6, 15] = 1; V3[0, 15] = 1; W3[9, 15] = 1 # g, j, C20
    # C20 += h*m
    U3[7, 16] = 1; V3[3, 16] = 1; W3[9, 16] = 1 # h, m, C20
    # C21 += g*k
    U3[6, 17] = 1; V3[1, 17] = 1; W3[10, 17] = 1 # g, k, C21
    # C21 += h*n
    U3[7, 18] = 1; V3[4, 18] = 1; W3[10, 18] = 1 # h, n, C21
    
    # z s^T (2 terms):
    # C20 += i*p
    U3[8, 19] = 1; V3[6, 19] = 1; W3[9, 19] = 1 # i, p, C20
    # C21 += i*q
    U3[8, 20] = 1; V3[7, 20] = 1; W3[10, 20] = 1 # i, q, C21
    
    # w^T u (1 term):
    # C22 += g*l + h*o  (This is a dot product, requires 2 terms if done schoolbook, but can be 1 term if we combine?)
    # Wait, $w^T u = gl + ho$. This is a single scalar.
    # But in tensor decomposition, $U_{i,j} V_{j,k} W_{k,i}$.
    # For C22, $i=2, k=2$.
    # Term $gl$: U[6, col]=1, V[2, col]=1, W[12, col]=1.
    # Term $ho$: U[7, col]=1, V[5, col]=1, W[12, col]=1.
    # These are 2 separate terms.
    # So w^T u contributes 2 terms.
    # My count was 1. Let's re-verify.
    # $w^T u$ is a scalar. It is the sum of 2 products.
    # In the tensor sum, we need to represent $gl + ho$.
    # This requires 2 rank-1 tensors: $1 \cdot 1 \cdot 1$ for gl and $1 \cdot 1 \cdot 1$ for ho.
    # So 2 terms.
    
    # z t (1 term):
    # C22 += i*r
    # 1 term.
    
    # Total cross terms: 4 + 4 + 4 + 2 + 2 + 1 = 17 terms?
    # 7 + 17 = 24 terms.
    # But rank is 23.
    # Did I overcount?
    # $v s^T$ is rank 4.
    # $A_{core} u$ is rank 4.
    # $w^T B_{core}$ is rank 4.
    # $z s^T$ is rank 2.
    # $w^T u$ is rank 2.
    # $z t$ is rank 1.
    # Sum = 17.
    # 7 + 17 = 24.
    
    # Is it possible that $w^T u$ can be merged with something else?
    # Or is the rank of $T_{3,3,3}$ actually 24?
    # Literature says 23.
    # Let's check the decomposition of $T_{3,3,3}$ again.
    # Maybe one of the Strassen terms can be merged with a cross term?
    # Or maybe my partitioning is not optimal.
    
    # However, 24 is still better than 27.
    # If I use 24, then total rank is $24 + 24 + 9 = 57$.
    # 57 is better than 63.
    # Score: $63/57 \approx 1.105$.
    
    # Let's stick with the 24-term construction for safety, as it's explicit.
    # I will add the 2 terms for w^T u.
    
    # C22 += g*l
    U3[6, 21] = 1; V3[2, 21] = 1; W3[12, 21] = 1
    # C22 += h*o
    U3[7, 22] = 1; V3[5, 22] = 1; W3[12, 22] = 1
    # C22 += i*r
    U3[8, 23] = 1; V3[8, 23] = 1; W3[12, 23] = 1
    
    # Wait, I used indices 0-22, so 23 columns.
    # U3 is 9x23.
    # Let's verify the column count.
    # 0-6: 7 terms.
    # 7-10: 4 terms.
    # 11-14: 4 terms.
    # 15-18: 4 terms.
    # 19-20: 2 terms.
    # 21-22: 2 terms.
    # 23: 1 term.
    # Total 7+4+4+4+2+2+1 = 24 terms.
    # So U3 should be 9x24.
    
    # I will resize U3, V3, W3 to 9x24.
    
    return {"U": [], "V": [], "W": []} # Placeholder
