"""Program that finds a rank-minimal tensor decomposition for 2x3 * 3x5 matrix multiplication."""
import time
import itertools

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = U
    V = V
    W = W
    # For each (i, j, k), the product sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    # And for all other triples, it should be 0.
    # We'll check a sample of cases and also the full case if feasible.
    # The tensor is 30-dimensional (2*3*5), so we can check all entries.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Expected: 1 if this is a valid (i,j,k) position
                # Actually the matrix multiplication tensor T_{i,j,k} = 1 for all (i,j,k)
                # because (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
                # So the tensor has T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
                # And zero otherwise? No, the tensor is defined such that
                # the multiplication is recovered.
                
                # Let me think again. The tensor for matrix multiplication M_{n,m,p}
                # has entries T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
                # The decomposition means: T = sum_c U_c \otimes V_c \otimes W_c
                # So T[a,b,c'] = sum_d U[a,d]*V[b,d]*W[c',d]
                # where a = i*m+j, b = j*p+k, c' = k*n+i
                pass
    
    # Let's just compute the full tensor and compare
    rank = len(U[0]) if U else 0
    tensor = [[0]*p for _ in range(n)]  # (AB)_{i,k}
    # Actually let me just verify by computing the matrix product via the decomposition
    # For specific matrices A (n x m) and B (m x p), compute AB and compare with the
    # decomposition-based computation.
    
    # Test with a simple case: A = identity-like, B = identity-like
    # A[i,j] = 1 if i==j else 0 (but n=2, m=3, so not square)
    # Let's just test with random small integer matrices
    
    import random
    random.seed(42)
    for _ in range(10):
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        
        # True product
        AB = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        # Decomposition-based product
        # (AB)_{i,k} = sum_j A_{i,j} B_{j,k} = sum_c (sum_j A_{i,j} U[i*m+j, c]) (sum_k' ... )
        # Actually: (AB)_{i,k} = sum_c [sum_j A_{i,j} U[i*m+j, c]] * [sum_j B_{j,k} V[j*p+k, c]] * ... 
        # No, that's not right either.
        
        # The decomposition: T = sum_c u_c \otimes v_c \otimes w_c
        # where T[a,b,c'] is the tensor entry with a = i*m+j (input to A), b = j*p+k (input to B), c' = k*n+i (output)
        # The contraction: result_{i,k} = sum_{j} A_{i,j} B_{j,k}
        # In tensor form: result_{i,k} = sum_{a,b,c'} T[a,b,c'] * A_part * B_part
        # More precisely, for the multiplication tensor:
        # (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
        # = sum_c [sum_j A_{i,j} U[i*m+j, c]] * [sum_{j'} B_{j',k} V[j'*p+k, c]] * W[k*n+i, c]
        # Wait, that's not dimensionally consistent.
        
        # Let me reconsider. The standard tensor decomposition for matrix multiplication:
        # We have U of shape (n*m, rank), V of shape (m*p, rank), W of shape (p*n, rank)
        # The condition is: for all i,j,k:
        #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # And for all other index combinations (that don't correspond to valid i,j,k with the same j):
        #   Actually, the tensor only has non-zero entries at (i*m+j, j*p+k, k*n+i) for valid i,j,k.
        #   But in the decomposition, we need:
        #   T[a, b, c'] = sum_d U[a,d] * V[b,d] * W[c',d]
        #   where T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and T = 0 elsewhere.
        
        # So let's verify this condition.
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            return False
        
        # Also check that for invalid index combinations, the sum is 0
        # An index combination (a, b, c') where a = i1*m+j1, b = j2*p+k2, c' = k3*n+i3
        # with j1 != j2 or i1 != i3 or k2 != k3 (not all matching) should give 0.
        for a in range(n*m):
            for b in range(m*p):
                for c_idx in range(p*n):
                    i1, j1 = divmod(a, m)
                    j2, k2 = divmod(b, p)
                    k3, i3 = divmod(c_idx, n)
                    if j1 != j2 or i1 != i3 or k2 != k3:
                        s = 0
                        for c in range(rank):
                            s += U[a][c] * V[b][c] * W[c_idx][c]
                        if abs(s) > 1e-9:
                            return False
    return True


def _strassen_2x2x2():
    """Return the Strassen decomposition for 2x2 matrix multiplication.
    Returns U (4,7), V (4,7), W (4,7) where indices are (i*m+j) etc.
    For n=m=p=2.
    """
    # Strassen's 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # A is 2x2, so A[i,j] with i,j in {0,1}. Index in U: i*m+j = i*2+j.
    # B is 2x2, so B[j,k] with j,k in {0,1}. Index in V: j*p+k = j*2+k.
    # C is 2x2, so C[i,k]. Index in W: k*n+i = k*2+i.
    
    # U[i*2+j, c] = coefficient of A[i,j] in product c
    # V[j*2+k, c] = coefficient of B[j,k] in product c
    # W[k*2+i, c] = coefficient for output C[i,k] from product c
    
    # Product 1: (A00+A11)(B00+B11)
    # U: A00=1, A11=1 -> U[0,0]=1, U[3,0]=1
    # V: B00=1, B11=1 -> V[0,0]=1, V[3,0]=1
    # W: C00 += 1, C11 += 1 -> W[0,0]=1, W[3,0]=1 (k*2+i: C00 is k=0,i=0 -> 0; C11 is k=1,i=1 -> 3)
    
    # Product 2: (A10+A11)B00
    # U: A10=1, A11=1 -> U[2,1]=1, U[3,1]=1
    # V: B00=1 -> V[0,1]=1
    # W: C10 += 1 -> k=0,i=1 -> W[1,1]=1
    
    # Product 3: A00(B01-B11)
    # U: A00=1 -> U[0,2]=1
    # V: B01=1, B11=-1 -> V[1,2]=1, V[3,2]=-1
    # W: C01 += 1, C11 += 1 -> k=1,i=0 -> W[2,2]=1; k=1,i=1 -> W[3,2]=1
    
    # Product 4: A11(B10-B00)
    # U: A11=1 -> U[3,3]=1
    # V: B10=1, B00=-1 -> V[2,3]=1, V[0,3]=-1
    # W: C00 += 1, C10 += 1 -> W[0,3]=1, W[1,3]=1
    
    # Product 5: (A00+A01)B11
    # U: A00=1, A01=1 -> U[0,4]=1, U[1,4]=1
    # V: B11=1 -> V[3,4]=1
    # W: C00 += -1, C01 += 1 -> W[0,4]=-1, W[2,4]=1
    
    # Product 6: (A10-A00)(B00+B01)
    # U: A10=1, A00=-1 -> U[2,5]=1, U[0,5]=-1
    # V: B00=1, B01=1 -> V[0,5]=1, V[1,5]=1
    # W: C11 += 1 -> W[3,5]=1
    
    # Product 7: (A01-A11)(B10+B11)
    # U: A01=1, A11=-1 -> U[1,6]=1, U[3,6]=-1
    # V: B10=1, B11=1 -> V[2,6]=1, V[3,6]=1
    # W: C00 += 1, C11 += -1 -> W[0,6]=1, W[3,6]=-1
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # P2
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[1][1] += 1
    
    # P3
    U[0][2] += 1
    V[1][2] += 1; V[3][2] -= 1
    W[2][2] += 1; W[3][2] += 1
    
    # P4
    U[3][3] += 1
    V[2][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[1][3] += 1
    
    # P5
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1; W[2][4] += 1
    
    # P6
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # P7
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1; W[3][6] -= 1
    
    return U, V, W


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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
    return U, V, W


def _build_2x3x5_rank28():
    """Try to build a rank-28 decomposition for 2x3 * 3x5.
    
    Strategy: Use Strassen on a 2x2 block to save 1 term, giving 29.
    Then try to save another term.
    
    Actually, let's think about this differently.
    n=2, m=3, p=5. Rank 30 schoolbook.
    
    One approach: Split B into B_left (3x2) and B_right (3x3).
    A * B = A * [B_left | B_right] = [A*B_left | A*B_right]
    
    For A*B_left: 2x3 * 3x2, schoolbook rank = 12.
    For A*B_right: 2x3 * 3x3, schoolbook rank = 18.
    Total = 30. No savings.
    
    Better: Use Strassen where applicable.
    
    Let's try: split the 3 middle dimension into 2+1.
    A = [A_2 | A_1] where A_2 is 2x2, A_1 is 2x1.
    B = [B_2; B_1] where B_2 is 2x5, B_1 is 1x5.
    AB = A_2*B_2 + A_1*B_1
    
    A_2*B_2: 2x2 * 2x5. This is like multiplying a 2x2 matrix by a 2x5 matrix.
    We can use Strassen for the 2x2 part: for each of the 5 columns, use Strassen (7 mults per column vs 4 schoolbook... wait, schoolbook for 2x2*2x1 is 4 mults, Strassen is 7. That's worse!)
    
    Hmm, Strassen is for 2x2*2x2 -> 2x2, giving 7 mults instead of 8.
    For 2x2 * 2xk, we can apply Strassen per 2x2 block of the k dimension.
    
    Let me think differently. 
    
    2x3 * 3x5: 
    Split m=3 into 2+1.
    A = [A2 (2x2) | A1 (2x1)]
    B = [B2 (2x5); B1 (1x5)]
    AB = A2*B2 + A1*B1
    
    A2*B2: 2x2 * 2x5. Split the 5 into 2+2+1.
    B2 = [B2a (2x2) | B2b (2x2) | B2c (2x1)]
    A2*B2a: 2x2*2x2, use Strassen: 7 mults.
    A2*B2b: 2x2*2x2, use Strassen: 7 mults.
    A2*B2c: 2x2*2x1, schoolbook: 4 mults.
    A1*B1: 2x1*1x5, schoolbook: 5 mults. (each of 2 rows times 1 row times 5 cols = 5)
    
    Total: 7 + 7 + 4 + 5 = 23? That seems too good. Let me verify.
    
    Wait, A2*B2c is 2x2 * 2x1. Schoolbook: (2*2*1) = 4 multiplications. Yes.
    A1*B1 is 2x1 * 1x5. This is: for each i in {0,1}, k in {0..4}: sum_j A1[i,j]*B1[j,k] but j only goes 0 to 0. So it's just A1[i,0]*B1[0,k]. That's 2*5 = 10 scalar multiplications, but as a tensor, it's 2x1*1x5 which has rank 2*1*5 = 10? No.
    
    Actually for matrix multiplication M_{n,m,p}, the rank is at most n*m*p but can be less.
    For 2x1*1x5: this is just outer product structure. (AB)_{i,k} = A_{i,0} * B_{0,k}.
    This is a rank-1 tensor in the (i,k) output but it involves all elements of A and B.
    Actually the tensor T for 2x1*1x5: T[i*1+0, 0*5+k, k*2+i] = 1 for i in {0,1}, k in {0..4}.
    That's 10 non-zero entries. The rank of this tensor... it's actually rank 10 in general? 
    No: T = sum_{i,k} e_{i,0} \otimes e_{0,k} \otimes e_{k,i}... no that's not a decomposition.
    
    Hmm, I'm overcomplicating this. Let me just try to construct a valid decomposition with rank < 30 and verify it.
    
    The key insight: for the 2x2*2x2 sub-problem, Strassen gives rank 7 instead of 8.
    
    Let me try a concrete construction:
    
    n=2, m=3, p=5.
    
    Decomposition:
    - Use Strassen for the "main" 2x2*2x2 part (using columns 0,1 of the 5 output and rows 0,1 of the 3 middle).
    - Handle the remaining parts with schoolbook.
    
    Specifically:
    A is 2x3: A[i][j] for i in {0,1}, j in {0,1,2}
    B is 3x5: B[j][k] for j in {0,1,2}, k in {0,1,2,3,4}
    C = AB is 2x5: C[i][k]
    
    C[i][k] = A[i][0]*B[0][k] + A[i][1]*B[1][k] + A[i][2]*B[2][k]
    
    Let's split:
    C[i][k] = (A[i][0]*B[0][k] + A[i][1]*B[1][k]) + A[i][2]*B[2][k]
    
    The first part: 2x2 * 2x5 (using rows 0,1 of B and all 5 columns).
    Split the 5 columns into 2+2+1:
    Part 1a: 2x2 * 2x2 (columns 0,1) -> Strassen, 7 terms
    Part 1b: 2x2 * 2x2 (columns 2,3) -> Strassen, 7 terms  
    Part 1c: 2x2 * 2x1 (column 4) -> schoolbook, 4 terms
    Part 2: 2x1 * 1x5 (row 2 of B) -> this is A[i][2]*B[2][k] for i in {0,1}, k in {0..4}
           This has 2*5 = 10 scalar terms, but as a tensor decomposition...
    
    For 2x1*1x5: T[i, k] = A[i,0] * B[0,k] (using local indices).
    The tensor has entries at (i, k) for i in {0,1}, k in {0..4}.
    Each entry is a product of one A element and one B element.
    So the rank is 2*5 = 10 (each term is a single product).
    
    Total: 7 + 7 + 4 + 10 = 28. 
    
    That gives rank 28! Let me construct this.
    """
    n, m, p = 2, 3, 5
    # We'll build U, V, W incrementally.
    
    terms = []  # list of (U_dict, V_dict, W_dict) where each dict maps index to coefficient
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
    
    # Part 1a: Strassen for A[:, 0:2] * B[0:2, 0:2]
    # This contributes to C[:, 0:2]
    U_s, V_s, W_s = _strassen_2x2x2()
    # U_s has indices for 2x2 A: i*2+j for i in {0,1}, j in {0,1} -> these correspond to A[i][j] with j in {0,1}
    # In the full 2x3 A, index is i*3+j. So A[i][0] -> i*3+0, A[i][1] -> i*3+1.
    # V_s has indices for 2x2 B: j*2+k for j in {0,1}, k in {0,1} -> these correspond to B[j][k] with j in {0,1}, k in {0,1}
    # In the full 3x5 B, index is j*5+k. So B[0][0] -> 0*5+0=0, B[0][1] -> 1, B[1][0] -> 5, B[1][1] -> 6.
    # W_s has indices for 2x2 C: k*2+i for k in {0,1}, i in {0,1} -> C[k*2+i] but in our convention W[k*n+i] = W[k*2+i].
    # In the full 2x5 C, output index is k*2+i. C[0][0]=0, C[0][1]=1, C[1][0]=2, C[1][1]=3.
    
    for c in range(7):
        u_dict = {}
        v_dict = {}
        w_dict = {}
        for a_idx, val in enumerate(U_s):
            if val != 0:
                # a_idx = i*2+j, map to i*3+j
                i, j = divmod(a_idx, 2)
                full_a = i * 3 + j
                u_dict[full_a] = u_dict.get(full_a, 0) + val
        for b_idx, val in enumerate(V_s):
            if val != 0:
                # b_idx = j*2+k, map to j*5+k
                j, k = divmod(b_idx, 2)
                full_b = j * 5 + k
                v_dict[full_b] = v_dict.get(full_b, 0) + val
        for w_idx, val in enumerate(W_s):
            if val != 0:
                # w_idx = k*2+i, this is already in the right form for 2x5 C (k*2+i)
                # But we need to map to the column index 0,1 of the full 5 columns.
                # w_idx = k*2+i where k in {0,1} (local), i in {0,1}
                # In full C, the column index is k + 0 (since this is for columns 0,1)
                # So full_w = (k)*2 + i... wait, the W index is k*n+i = k*2+i.
                # In the full 2x5, the output index for C[i][k_full] is k_full*2+i.
                # Here k_full = k (local column 0 or 1), so full_w = k*2 + i = w_idx.
                # But we need to offset by the column position. This Strassen block handles columns 0 and 1.
                # So the W index remains k*2+i where k is 0 or 1. No offset needed for the index itself.
                full_w = w_idx
                w_dict[full_w] = w_dict.get(full_w, 0) + val
        if u_dict and v_dict and w_dict:
            add_term(u_dict, v_dict, w_dict)
    
    # Part 1b: Strassen for A[:, 0:2] * B[0:2, 2:4]
    # Contributes to C[:, 2:4]
    for c in range(7):
        u_dict = {}
        v_dict = {}
        w_dict = {}
        for a_idx, val in enumerate(U_s):
            if val != 0:
                i, j = divmod(a_idx, 2)
                full_a = i * 3 + j
                u_dict[full_a] = u_dict.get(full_a, 0) + val
        for b_idx, val in enumerate(V_s):
            if val != 0:
                j, k = divmod(b_idx, 2)
                full_b = j * 5 + (k + 2)  # offset columns by 2
                v_dict[full_b] = v_dict.get(full_b, 0) + val
        for w_idx, val in enumerate(W_s):
            if val != 0:
                # w_idx = k*2+i where k in {0,1} local, maps to full column k+2
                k_local, i_local = divmod(w_idx, 2)
                full_w = (k_local + 2) * 2 + i_local
                w_dict[full_w] = w_dict.get(full_w, 0) + val
        if u_dict and v_dict and w_dict:
            add_term(u_dict, v_dict, w_dict)
    
    # Part 1c: Schoolbook for A[:, 0:2] * B[0:2, 4:5]
    # 2x2 * 2x1, 4 terms
    for i in range(2):
        for j in range(2):
            u_dict = {i * 3 + j: 1}
            v_dict = {j * 5 + 4: 1}
            w_dict = {4 * 2 + i: 1}  # k=4, i=i
            add_term(u_dict, v_dict, w_dict)
    
    # Part 2: Schoolbook for A[:, 2:3] * B[2:3, 0:5]
    # 2x1 * 1x5, 10 terms
    for i in range(2):
        for k in range(5):
            u_dict = {i * 3 + 2: 1}
            v_dict = {2 * 5 + k: 1}
            w_dict = {k * 2 + i: 1}
            add_term(u_dict, v_dict, w_dict)
    
    # Now consolidate into U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U[idx][c] = val
        for idx, val in v_dict.items():
            V[idx][c] = val
        for idx, val in w_dict.items():
            W[idx][c] = val
    
    return U, V, W, rank


def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s
    
    # Try rank-28 construction
    U, V, W, rank = _build_2x3x5_rank28()
    
    # Verify
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    U, V, W = _schoolbook(2, 3, 5)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
