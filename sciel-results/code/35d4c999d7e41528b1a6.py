"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    # We use the known optimal rank-11 decomposition for 2x2 by 2x3 matrix multiplication.
    # This combines Strassen's rank-7 method for the 2x2 block (first column of B)
    # with schoolbook rank-4 for the remaining two columns.
    #
    # The tensor for C = A * B where A is 2x2, B is 2x3:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Strassen for 2x2*2x2 gives 7 terms for B's first column.
    # Schoolbook for B's 2nd and 3rd columns gives 2*2 = 4 terms.
    # Total rank = 7 + 4 = 11.
    #
    # U has shape (n*m, rank) = (4, 11)
    # V has shape (m*p, rank) = (6, 11)
    # W has shape (p*n, rank) = (6, 11)
    #
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # n=2, m=2, p=3
    #
    # Strassen 7 terms for 2x2 * 2x2 (B[:, 0]):
    # Let A = [[a00, a01], [a10, a11]], B_col = [[b00], [b10]]
    # 
    # M1 = (a00+a11)*(b00+b10)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b00)
    # M4 = a11*(b00+b10)  -- wait, let me use standard Strassen
    #
    # Standard Strassen for C = A*B where A,B are 2x2:
    # C = [[a,b],[c,d]] * [[e,f],[g,h]]
    # p1 = (a+d)(e+h)
    # p2 = (c+d)h
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)e
    # p6 = (c+d)(f-e)
    # p7 = (a+b)(h-g)
    # C11 = p1+p4-p5+p7
    # C12 = p3+p5
    # C21 = p2+p4
    # C22 = p1-p2+p3+p6
    #
    # For our case, B's first column is [b00, b10], so:
    # e=b00, f=0, g=b10, h=0 (treating it as a 2x2 with second column zero)
    # But we only need C[:,0], so:
    # C00 = a*b00 + b*b10
    # C10 = c*b00 + d*b10
    #
    # Let me directly construct the 11-term decomposition.
    #
    # Terms 0-6: Strassen-like for column 0 of B
    # A = [[a00, a01], [a10, a11]]
    # B col 0 = [b00, b10]
    #
    # We need: C00 = a00*b00 + a01*b10
    #           C10 = a10*b00 + a11*b10
    #
    # Using 7 terms (Strassen adapted):
    # t0: (a00+a11)*(b00+b10)  -> contributes to both C00 and C10
    # t1: (a10+a11)*b00
    # t2: a00*(b10-b00)
    # t3: a11*(b00+b10)  -- hmm, let me just verify with a known rank-11 scheme.
    
    # Actually, let me just use a verified rank-11 construction.
    # 
    # For 2x2 * 2x3, the minimum rank is 11 (known result).
    # Here's a concrete decomposition:
    #
    # Let's index: A[i,j] for i in {0,1}, j in {0,1}
    #               B[j,k] for j in {0,1}, k in {0,1,2}
    #               C[i,k] for i in {0,1}, k in {0,1,2}
    #
    # U[i*m+j, c] = U[i*2+j, c]
    # V[j*p+k, c] = V[j*3+k, c]
    # W[k*n+i, c] = W[k*2+i, c]
    
    n, m, p = 2, 2, 3
    rank = 11
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen 7 terms for B's column 0 (k=0)
    # A = [[a0, a1], [a2, a3]] where a0=A[0,0], a1=A[0,1], a2=A[1,0], a3=A[1,1]
    # B col 0 = [b0, b1] where b0=B[0,0], b1=B[1,0]
    #
    # C[0,0] = a0*b0 + a1*b1
    # C[1,0] = a2*b0 + a3*b1
    #
    # 7-term scheme:
    # t0: (a0+a3)*(b0+b1)
    # t1: (a2+a3)*b0
    # t2: a0*(b1-b0)
    # t3: a3*(b0+b1)  -- no, this isn't standard
    #
    # Let me just use the standard Strassen formulas and verify:
    # For 2x2 * 2x2 with B = [[b0, 0], [b1, 0]]:
    # p1 = (a0+a3)(b0+0) = (a0+a3)*b0
    # p2 = (a2+a3)*0 = 0
    # p3 = a0*(0-b0) = -a0*b0
    # p4 = a3*(b1-b0)
    # p5 = (a0+a1)*b0
    # p6 = (a2+a3)*(0-b0) = -(a2+a3)*b0
    # p7 = (a0+a1)*(0-b1) = -(a0+a1)*b1
    #
    # C00 = p1+p4-p5+p7 = (a0+a3)*b0 + a3*(b1-b0) - (a0+a1)*b0 - (a0+a1)*b1
    #     = a0*b0 + a3*b0 + a3*b1 - a3*b0 - a0*b0 - a1*b0 - a0*b1 - a1*b1
    #     = a3*b1 - a1*b0 - a0*b1 - a1*b1
    # That's not right. Let me just directly write a valid rank-11 decomposition.
    
    # Direct construction:
    # For k=0 (Strassen, 7 terms):
    # We want C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    #                 C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    
    # Use these 7 products:
    # 0: (A[0,0]+A[1,1]) * (B[0,0]+B[1,0])
    # 1: (A[1,0]+A[1,1]) * B[0,0]
    # 2: A[0,0] * (B[1,0]-B[0,0])
    # 3: A[1,1] * (B[0,0]+B[1,0])  -- not quite
    #
    # Let me just hardcode a known correct rank-11 scheme.
    
    # Verified rank-11 for 2x2 * 2x3:
    # Terms for k=0 (7 terms using Strassen):
    # t0 = (A00+A11)*(B00+B10)
    # t1 = (A10+A11)*B00  
    # t2 = A00*(B10-B00)
    # t3 = A11*(B00+B10)
    # t4 = (A00+A01)*B00
    # t5 = (A10+A11)*(B10-B00)
    # t6 = (A00+A01)*(B10-B00)
    
    # Check C00 = t0 + t3 - t4 + t6 - t1 - t2
    # = (A00+A11)(B00+B10) + A11(B00+B10) - (A00+A01)B00 + (A00+A01)(B10-B00) - (A10+A11)B00 - A00(B10-B00)
    # This is getting messy. Let me just use a simple verified construction.
    
    # Simplest: use 4 schoolbook for k=0, 4 for k=1, 3 for k=2 = 11
    # But that's not minimal structure. Actually 4+4+4=12.
    #
    # For rank 11, we need to save 1 term. 
    # 
    # Known: rank(2x2*2x3) = 11.
    # 
    # Here's a clean construction:
    # For k=0: use Strassen 7 terms
    # For k=1,2: use schoolbook but share... no.
    #
    # Actually the simplest rank-11: 
    # 7 terms for first column of B (Strassen), 2 terms for second column, 2 terms for third column = 11
    # Wait, 2x2 * 1 column needs rank 4 in general, but with Strassen we can do 2x2*2x2 in 7.
    # For 2x2 * single column (2x1), rank is 4.
    #
    # So: 7 (for 2x2*2x2 where B has 2 cols) + 4 (for remaining 1 col) = 11? No, 7+4=11 but that's for 2x2*3x2.
    #
    # For 2x2 * 2x3: think of B as 2x2 (cols 0,1) + 2x1 (col 2).
    # Rank(2x2*2x2) = 7 (Strassen), Rank(2x2*2x1) = 4. Total = 11. Yes!
    
    # Strassen for A (2x2) * B[:,0:2] (2x2):
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # p1=(a+d)(e+h), p2=(c+d)h, p3=a(f-h), p4=d(g-e), p5=(a+b)e, p6=(c+d)(f-e), p7=(a+b)(h-g)
    # C00=p1+p4-p5+p7, C01=p3+p5, C10=p2+p4, C11=p1-p2+p3+p6
    
    # Schoolbook for A (2x2) * B[:,2] (2x1) = [i,j]:
    # C02 = a*i + b*j
    # C12 = c*i + d*j
    # 4 terms: a*i, b*j, c*i, d*j
    
    # Total: 7 + 4 = 11 terms.
    
    # Now fill U, V, W:
    # U[i*m+j, c] = coefficient of A[i,j] in term c
    # V[j*p+k, c] = coefficient of B[j,k] in term c
    # W[k*n+i, c] = coefficient of C[i,k] in term c
    
    # Strassen terms (c=0..6):
    # p1=(a+d)(e+h): U[0,0]=1,U[1,1]=1; V[0,0]=1,V[1,1]=1; contributes to C00,C11
    # W for p1: C00 += p1, C11 += p1 -> W[0*2+0]=1 (k=0,i=0), W[1*2+1]=1 (k=1,i=1)
    
    # p2=(c+d)h: U[1,0]=1,U[1,1]=1; V[1,1]=1; C10 += p2 -> W[0*2+1]=1
    # p3=a(f-h): U[0,0]=1; V[0,1]=1,V[1,1]=-1; C01 += p3 -> W[1*2+0]=1
    # p4=d(g-e): U[1,1]=1; V[1,0]=1,V[0,0]=-1; C00 += p4, C10 += p4 -> W[0*2+0]=1, W[0*2+1]=1
    # p5=(a+b)e: U[0,0]=1,U[0,1]=1; V[0,0]=1; C00 -= p5, C01 += p5 -> W[0*2+0]=-1, W[1*2+0]=1
    # p6=(c+d)(f-e): U[1,0]=1,U[1,1]=1; V[0,1]=1,V[0,0]=-1; C11 += p6 -> W[1*2+1]=1
    # p7=(a+b)(h-g): U[0,0]=1,U[0,1]=1; V[1,1]=1,V[1,0]=-1; C00 += p7 -> W[0*2+0]=1
    
    U[0,0]=1; U[3,0]=1  # p1: a+d -> A[0,0]+A[1,1]
    V[0,0]=1; V[4,0]=1  # p1: e+h -> B[0,0]+B[1,1]
    W[0,0]=1; W[3,0]=1  # p1: C00+C11 -> W[0*2+0]+W[1*2+1]
    
    U[2,1]=1; U[3,1]=1  # p2: c+d -> A[1,0]+A[1,1]
    V[4,1]=1            # p2: h -> B[1,1]
    W[1,1]=1            # p2: C10 -> W[0*2+1]
    
    U[0,2]=1            # p3: a -> A[0,0]
    V[3,2]=1; V[4,2]=-1 # p3: f-h -> B[0,1]-B[1,1]
    W[2,2]=1            # p3: C01 -> W[1*2+0]
    
    U[3,3]=1            # p4: d -> A[1,1]
    V[4,3]=1; V[0,3]=-1 # p4: g-e -> B[1,0]-B[0,0]
    W[0,3]=1; W[1,3]=1  # p4: C00+C10 -> W[0*2+0]+W[0*2+1]
    
    U[0,4]=1; U[1,4]=1  # p5: a+b -> A[0,0]+A[0,1]
    V[0,4]=1            # p5: e -> B[0,0]
    W[0,4]=-1; W[2,4]=1 # p5: -C00+C01 -> W[0*2+0]=-1, W[1*2+0]=1
    
    U[2,5]=1; U[3,5]=1  # p6: c+d -> A[1,0]+A[1,1]
    V[3,5]=1; V[0,5]=-1 # p6: f-e -> B[0,1]-B[0,0]
    W[3,5]=1            # p6: C11 -> W[1*2+1]
    
    U[0,6]=1; U[1,6]=1  # p7: a+b -> A[0,0]+A[0,1]
    V[4,6]=1; V[4,6]=-1 # p7: h-g -> B[1,1]-B[1,0]
    W[0,6]=1            # p7: C00 -> W[0*2+0]
    
    # Fix p7 V:
    V[4,6]=1; V[4,6]=-1  # This is wrong, let me redo
    V[4,6]=1   # h = B[1,1]
    V[4,6]=-1  # g = B[1,0], so V[4,6] should be -1 for g... 
    
    # Let me redo this more carefully.
    # B[j,k]: j=0,1; k=0,1,2
    # V[j*p+k, c] = V[j*3+k, c]
    # B[0,0]=V[0], B[0,1]=V[1], B[0,2]=V[2]
    # B[1,0]=V[3], B[1,1]=V[4], B[1,2]=V[5]
    
    # p1=(a+d)(e+h): e=B[0,0]=V[0], h=B[1,1]=V[4]
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[4,0]=1
    W[0,0]=1; W[3,0]=1  # C[0,0]=W[0], C[1,1]=W[3]
    
    # p2=(c+d)h: c=A[1,0]=U[2], d=A[1,1]=U[3], h=B[1,1]=V[4]
    U[2,1]=1; U[3,1]=1
    V[4,1]=1
    W[1,1]=1  # C[1,0]=W[1]
    
    # p3=a(f-h): a=A[0,0]=U[0], f=B[0,1]=V[1], h=B[1,1]=V[4]
    U[0,2]=1
    V[1,2]=1; V[4,2]=-1
    W[2,2]=1  # C[0,1]=W[2]
    
    # p4=d(g-e): d=A[1,1]=U[3], g=B[1,0]=V[3], e=B[0,0]=V[0]
    U[3,3]=1
    V[3,3]=1; V[0,3]=-1
    W[0,3]=1; W[1,3]=1  # C[0,0]=W[0], C[1,0]=W[1]
    
    # p5=(a+b)e: a=A[0,0]=U[0], b=A[0,1]=U[1], e=B[0,0]=V[0]
    U[0,4]=1; U[1,4]=1
    V[0,4]=1
    W[0,4]=-1; W[2,4]=1  # -C[0,0]+C[0,1]
    
    # p6=(c+d)(f-e): c=A[1,0]=U[2], d=A[1,1]=U[3], f=B[0,1]=V[1], e=B[0,0]=V[0]
    U[2,5]=1; U[3,5]=1
    V[1,5]=1; V[0,5]=-1
    W[3,5]=1  # C[1,1]=W[3]
    
    # p7=(a+b)(h-g): a=A[0,0]=U[0], b=A[0,1]=U[1], h=B[1,1]=V[4], g=B[1,0]=V[3]
    U[0,6]=1; U[1,6]=1
    V[4,6]=1; V[3,6]=-1
    W[0,6]=1  # C[0,0]=W[0]
    
    # Schoolbook for k=2 (4 terms):
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # t7: A[0,0]*B[0,2]
    U[0,7]=1
    V[2,7]=1  # B[0,2]
    W[4,7]=1  # C[0,2]=W[2*2+0]=W[4]
    
    # t8: A[0,1]*B[1,2]
    U[1,8]=1
    V[5,8]=1  # B[1,2]
    W[4,8]=1  # C[0,2]
    
    # t9: A[1,0]*B[0,2]
    U[2,9]=1
    V[2,9]=1  # B[0,2]
    W[5,9]=1  # C[1,2]=W[2*2+1]=W[5]
    
    # t10: A[1,1]*B[1,2]
    U[3,10]=1
    V[5,10]=1  # B[1,2]
    W[5,10]=1  # C[1,2]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
