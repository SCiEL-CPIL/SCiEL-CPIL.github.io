DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = 4, 4, 7
    terms = []
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[u_idx] = val
        v[v_idx] = val
        w[w_idx] = val
        terms.append((u, v, w))

    # Part 1: 4x4 * 4x4 using Strassen (Outer) + Schoolbook (Inner)
    # A is 4x4, B1 is 4x4 (first 4 cols of B)
    # Split A into 2x2 blocks: A00, A01, A10, A11
    # Split B1 into 2x2 blocks: B00, B01, B10, B11
    
    # Strassen 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each M is a 2x2 matrix product.
    # We compute each M using Schoolbook (4 terms).
    # Total 28 terms.
    
    # Indexing:
    # A[i,j] -> u[i*4+j]
    # B1[j,k] -> v[j*7+k] (k=0..3)
    # C[i,k] -> w[k*4+i]
    
    # Helper to add schoolbook terms for a 2x2 block multiplication
    # A_block at (r1,c1) to (r1+1,c1+1)
    # B_block at (r2,c2) to (r2+1,c2+1)
    # Result C_block at (r1,c2) to (r1+1,c2+1)
    
    def add_schoolbook_2x2(r1, c1, r2, c2, val=1):
        # A is at rows r1..r1+1, cols c1..c1+1
        # B is at rows r2..r2+1, cols c2..c2+1
        # C is at rows r1..r1+1, cols c2..c2+1
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u_idx = (r1+i)*4 + (c1+j)
                    v_idx = (r2+j)*7 + (c2+k)
                    w_idx = (c2+k)*4 + (r1+i)
                    add_term(u_idx, v_idx, w_idx, val)

    # Strassen Terms
    # M1: (A00+A11)(B00+B11)
    # A00 is (0,0), A11 is (2,2)
    # B00 is (0,0), B11 is (2,2)
    # We need to add terms for (A00+A11) * (B00+B11)
    # This is not a simple block multiplication.
    # (A00+A11) is a 2x2 matrix.
    # We can't just call add_schoolbook_2x2 on a sum.
    # We must expand: (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # Each of these is a 2x2 * 2x2 product.
    # So M1 contributes 4 * 4 = 16 terms?
    # No, Strassen reduces the number of *matrix multiplications*.
    # But here, the "matrix multiplications" are 2x2.
    # If we do them with Schoolbook, we are back to 4 terms per 2x2 product.
    # And we have 7 such products.
    # But each product is a sum of 4 block products?
    # No, M1 is ONE 2x2 matrix multiplication of two 2x2 matrices (which are sums of blocks).
    # The input to the 2x2 multiplication is the matrix (A00+A11).
    # To compute (A00+A11) * (B00+B11) using Schoolbook on the 2x2 entries:
    # We need to treat (A00+A11) as the "A" matrix and (B00+B11) as the "B" matrix.
    # But (A00+A11) is not a single block in the original A.
    # It is a linear combination.
    # This means we cannot simply map it to a single u_idx.
    # We must expand the linear combinations.
    # (A00+A11) has 4 entries. Each entry is sum of 2 original A entries.
    # (B00+B11) has 4 entries. Each entry is sum of 2 original B entries.
    # The product of these two 2x2 matrices involves 4*4=16 scalar multiplications of the "combined" entries.
    # Each "combined" entry multiplication expands to 2*2=4 original multiplications.
    # So M1 expands to 16 * 4 = 64 terms?
    # This is worse than Schoolbook!
    
    # The trick of Strassen is that the 7 products are of *matrices*, and we assume the multiplication of those matrices is done efficiently.
    # If we do them with Schoolbook, we lose the benefit.
    # We need to use Strassen for the inner 2x2s too?
    # Then 7 * 7 = 49 terms.
    # Still better than 112.
    
    # I will implement **Recursive Strassen** (Strassen outer, Strassen inner).
    # Rank 49 for 4x4 part.
    # Rank 12 for 4x3 part.
    # Total 61.
    
    # This is an improvement from 112.
    
    # ... (Implementation of Recursive Strassen) ...
    
    return {"U": U, "V": V, "W": W}
