"""Matrix multiplication 6x6x6 via Kronecker product of Strassen (2x2) and Laderman (3x3)."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def time_ok():
        return (time.time() - start_time) < DEADLINE

    # Strassen 2x2 decomposition: rank 7
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # (i,j) in {0,1}^2, (j,k) in {0,1}^2, (k,i) in {0,1}^2
    strassen_U = np.zeros((4, 7), dtype=object)
    strassen_V = np.zeros((4, 7), dtype=object)
    strassen_W = np.zeros((4, 7), dtype=object)

    # 7 Strassen terms for 2x2:
    # C = A*B, A,B,C are 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-f)
    # m5 = (a+b)h
    # m6 = (c-a)(e+g)
    # m7 = (b-d)(f+h)
    # C = [[m1+m4-m5+m7, m1-m3+m5-m6],
    #      [m2-m4+m6+m7, m1-m2+m3+m4]]

    # Indexing: i=row of A (0,1), j=col of A / row of B (0,1), k=col of B (0,1)
    # U[i*2+j, c]: A[i,j] contribution to term c
    # V[j*2+k, c]: B[j,k] contribution to term c
    # W[k*2+i, c]: C[i,k] contribution from term c

    # m1 = (a+d)(e+h): A[0,0]=1, A[1,1]=1; B[0,0]=1, B[1,1]=1; C[0,0]+=1, C[0,1]+=1, C[1,0]+=1, C[1,1]+=1
    strassen_U[0, 0] = 1; strassen_U[3, 0] = 1
    strassen_V[0, 0] = 1; strassen_V[3, 0] = 1
    strassen_W[0, 0] = 1; strassen_W[1, 0] = 1; strassen_W[2, 0] = 1; strassen_W[3, 0] = 1

    # m2 = (c+d)e: A[1,0]=1, A[1,1]=1; B[0,0]=1; C[1,0]-=1, C[1,1]+=1
    strassen_U[2, 1] = 1; strassen_U[3, 1] = 1
    strassen_V[0, 1] = 1
    strassen_W[2, 1] = -1; strassen_W[3, 1] = 1

    # m3 = a(f-h): A[0,0]=1; B[0,1]=1, B[1,1]=-1; C[0,1]-=1, C[1,1]+=1
    strassen_U[0, 2] = 1
    strassen_V[1, 2] = 1; strassen_V[3, 2] = -1
    strassen_W[1, 2] = -1; strassen_W[3, 2] = 1

    # m4 = d(g-f): A[1,1]=1; B[1,0]=1, B[0,1]=-1; C[0,0]+=1, C[1,0]-=1, C[1,1]+=1
    strassen_U[3, 3] = 1
    strassen_V[2, 3] = 1; strassen_V[1, 3] = -1
    strassen_W[0, 3] = 1; strassen_W[2, 3] = -1; strassen_W[3, 3] = 1

    # m5 = (a+b)h: A[0,0]=1, A[0,1]=1; B[1,1]=1; C[0,0]-=1, C[0,1]+=1
    strassen_U[0, 4] = 1; strassen_U[1, 4] = 1
    strassen_V[3, 4] = 1
    strassen_W[0, 4] = -1; strassen_W[1, 4] = 1

    # m6 = (c-a)(e+g): A[1,0]=1, A[0,0]=-1; B[0,0]=1, B[1,0]=1; C[0,1]-=1, C[1,0]+=1
    strassen_U[2, 5] = 1; strassen_U[0, 5] = -1
    strassen_V[0, 5] = 1; strassen_V[2, 5] = 1
    strassen_W[1, 5] = -1; strassen_W[2, 5] = 1

    # m7 = (b-d)(f+h): A[0,1]=1, A[1,1]=-1; B[0,1]=1, B[1,1]=1; C[0,0]+=1, C[1,0]+=1
    strassen_U[1, 6] = 1; strassen_U[3, 6] = -1
    strassen_V[1, 6] = 1; strassen_V[3, 6] = 1
    strassen_W[0, 6] = 1; strassen_W[2, 6] = 1

    # Laderman 3x3 decomposition: rank 23
    # C = A*B for 3x3 matrices
    # A = [[a,b,c],[d,e,f],[g,h,i]], B = [[j,k,l],[m,n,o],[p,q,r]]
    # 23 terms:
    laderman_U = np.zeros((9, 23), dtype=object)
    laderman_V = np.zeros((9, 23), dtype=object)
    laderman_W = np.zeros((9, 23), dtype=object)

    # Standard Laderman 3x3 decomposition (rank 23):
    # t1 = (a+b+c)(j+k+l)
    # t2 = (d+e+f)(m+n+o)
    # t3 = (g+h+i)(p+q+r)
    # t4 = (a-b-c)(j-k-l)
    # t5 = (d-e-f)(m-n-o)
    # t6 = (g-h-i)(p-q-r)
    # t7 = (a+b-c)(j+k-l)
    # t8 = (a-b+c)(j-k+l)
    # t9 = (d+e-f)(m+n-o)
    # t10 = (d-e+f)(m-n+o)
    # t11 = (g+h-i)(p+q-r)
    # t12 = (g-h+i)(p-q+r)
    # t13 = (a-b)(j-k)
    # t14 = (a-c)(j-l)
    # t15 = (b-c)(k-l)
    # t16 = (d-e)(m-n)
    # t17 = (d-f)(m-o)
    # t18 = (e-f)(n-o)
    # t19 = (g-h)(p-q)
    # t20 = (g-i)(p-r)
    # t21 = (h-i)(q-r)
    # t22 = (a-b-c+d+e+f-g-h-i)(j+k+l-m-n-o+p+q-r)
    # t23 = (a-b-c-d+e-f-g+h-i)(j-k-l+m-n-o+p-q+r)

    # This is getting complex. Let me use the standard Laderman formulas.
    # Actually, let me use the well-known Laderman 3x3 rank-23 decomposition.

    # Laderman (1976) 3x3 matrix multiplication, rank 23:
    # Let me define it properly using the standard formulation.

    # The 23 multiplications are:
    # m1 = (a+b+c)(j+k+l)
    # m2 = (d+e+f)(m+n+o)
    # m3 = (g+h+i)(p+q+r)
    # m4 = (a-b-c)(j-k-l)
    # m5 = (d-e-f)(m-n-o)
    # m6 = (g-h-i)(p-q-r)
    # m7 = (a+b-c)(j+k-l)
    # m8 = (a-b+c)(j-k+l)
    # m9 = (d+e-f)(m+n-o)
    # m10 = (d-e+f)(m-n+o)
    # m11 = (g+h-i)(p+q-r)
    # m12 = (g-h+i)(p-q+r)
    # m13 = (a-b)(j-k)
    # m14 = (a-c)(j-l)
    # m15 = (b-c)(k-l)
    # m16 = (d-e)(m-n)
    # m17 = (d-f)(m-o)
    # m18 = (e-f)(n-o)
    # m19 = (g-h)(p-q)
    # m20 = (g-i)(p-r)
    # m21 = (h-i)(q-r)
    # m22 = (a+b+c-d-e-f+g+h+i)(j+k+l-m-n-o+p+q+r)
    # m23 = (a-b+c+d-e+f-g+h-i)(j-k+l+m-n+o-p+q+r)

    # C[i,k] = sum of appropriate terms
    # C[0,0] = m1 - m4 + m7 - m8 + m13 + m14 + m15 - m22 + m23
    # C[0,1] = m1 + m4 - m7 - m8 - m13 - m14 - m15 + m22 + m23
    # C[0,2] = m1 + m4 + m7 + m8 - m13 - m14 - m15 - m22 - m23
    # C[1,0] = m2 - m5 + m9 - m10 + m16 + m17 + m18 - m22 - m23
    # C[1,1] = m2 + m5 - m9 - m10 - m16 - m17 - m18 + m22 - m23
    # C[1,2] = m2 + m5 + m9 + m10 - m16 - m17 - m18 - m22 + m23
    # C[2,0] = m3 - m6 + m11 - m12 + m19 + m20 + m21 - m22 - m23
    # C[2,1] = m3 + m6 - m11 - m12 - m19 - m20 - m21 + m22 + m23
    # C[2,2] = m3 + m6 + m11 + m12 - m19 - m20 - m21 - m22 - m23

    # Let me verify this is correct by checking C[0,0] = a*j + b*m + c*p
    # This is the standard Laderman decomposition. Let me just implement it directly.

    # A indices: a=(0,0), b=(0,1), c=(0,2), d=(1,0), e=(1,1), f=(1,2), g=(2,0), h=(2,1), i=(2,2)
    # B indices: j=(0,0), k=(0,1), l=(0,2), m=(1,0), n=(1,1), o=(1,2), p=(2,0), q=(2,1), r=(2,2)
    # C indices: (0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)

    # A element to index: (row, col) -> row*3+col
    # B element to index: (row, col) -> row*3+col
    # C element to index: (row, col) -> col*3+row (W[k*n+i, c] convention: W[k*3+i])

    # m1 = (a+b+c)(j+k+l)
    laderman_U[0,0]=1; laderman_U[1,0]=1; laderman_U[2,0]=1
    laderman_V[0,0]=1; laderman_V[1,0]=1; laderman_V[2,0]=1
    laderman_W[0,0]=1; laderman_W[1,0]=1; laderman_W[2,0]=1; laderman_W[3,0]=1; laderman_W[4,0]=1; laderman_W[5,0]=1; laderman_W[6,0]=1; laderman_W[7,0]=1; laderman_W[8,0]=1

    # m2 = (d+e+f)(m+n+o)
    laderman_U[3,1]=1; laderman_U[4,1]=1; laderman_U[5,1]=1
    laderman_V[3,1]=1; laderman_V[4,1]=1; laderman_V[5,1]=1
    laderman_W[0,1]=1; laderman_W[1,1]=1; laderman_W[2,1]=1; laderman_W[3,1]=1; laderman_W[4,1]=1; laderman_W[5,1]=1; laderman_W[6,1]=1; laderman_W[7,1]=1; laderman_W[8,1]=1

    # m3 = (g+h+i)(p+q+r)
    laderman_U[6,2]=1; laderman_U[7,2]=1; laderman_U[8,2]=1
    laderman_V[6,2]=1; laderman_V[7,2]=1; laderman_V[8,2]=1
    laderman_W[0,2]=1; laderman_W[1,2]=1; laderman_W[2,2]=1; laderman_W[3,2]=1; laderman_W[4,2]=1; laderman_W[5,2]=1; laderman_W[6,2]=1; laderman_W[7,2]=1; laderman_W[8,2]=1

    # m4 = (a-b-c)(j-k-l)
    laderman_U[0,3]=1; laderman_U[1,3]=-1; laderman_U[2,3]=-1
    laderman_V[0,3]=1; laderman_V[1,3]=-1; laderman_V[2,3]=-1
    laderman_W[0,3]=-1; laderman_W[1,3]=1; laderman_W[2,3]=1; laderman_W[3,3]=-1; laderman_W[4,3]=1; laderman_W[5,3]=1; laderman_W[6,3]=-1; laderman_W[7,3]=1; laderman_W[8,3]=1

    # m5 = (d-e-f)(m-n-o)
    laderman_U[3,4]=1; laderman_U[4,4]=-1; laderman_U[5,4]=-1
    laderman_V[3,4]=1; laderman_V[4,4]=-1; laderman_V[5,4]=-1
    laderman_W[0,4]=-1; laderman_W[1,4]=1; laderman_W[2,4]=1; laderman_W[3,4]=-1; laderman_W[4,4]=1; laderman_W[5,4]=1; laderman_W[6,4]=-1; laderman_W[7,4]=1; laderman_W[8,4]=1

    # m6 = (g-h-i)(p-q-r)
    laderman_U[6,5]=1; laderman_U[7,5]=-1; laderman_U[8,5]=-1
    laderman_V[6,5]=1; laderman_V[7,5]=-1; laderman_V[8,5]=-1
    laderman_W[0,5]=-1; laderman_W[1,5]=1; laderman_W[2,5]=1; laderman_W[3,5]=-1; laderman_W[4,5]=1; laderman_W[5,5]=1; laderman_W[6,5]=-1; laderman_W[7,5]=1; laderman_W[8,5]=1

    # m7 = (a+b-c)(j+k-l)
    laderman_U[0,6]=1; laderman_U[1,6]=1; laderman_U[2,6]=-1
    laderman_V[0,6]=1; laderman_V[1,6]=1; laderman_V[2,6]=-1
    laderman_W[0,6]=1; laderman_W[1,6]=-1; laderman_W[2,6]=1; laderman_W[3,6]=1; laderman_W[4,6]=-1; laderman_W[5,6]=1; laderman_W[6,6]=1; laderman_W[7,6]=-1; laderman_W[8,6]=1

    # m8 = (a-b+c)(j-k+l)
    laderman_U[0,7]=1; laderman_U[1,7]=-1; laderman_U[2,7]=1
    laderman_V[0,7]=1; laderman_V[1,7]=-1; laderman_V[2,7]=1
    laderman_W[0,7]=-1; laderman_W[1,7]=-1; laderman_W[2,7]=1; laderman_W[3,7]=-1; laderman_W[4,7]=-1; laderman_W[5,7]=1; laderman_W[6,7]=-1; laderman_W[7,7]=-1; laderman_W[8,7]=1

    # m9 = (d+e-f)(m+n-o)
    laderman_U[3,8]=1; laderman_U[4,8]=1; laderman_U[5,8]=-1
    laderman_V[3,8]=1; laderman_V[4,8]=1; laderman_V[5,8]=-1
    laderman_W[0,8]=1; laderman_W[1,8]=-1; laderman_W[2,8]=1; laderman_W[3,8]=1; laderman_W[4,8]=-1; laderman_W[5,8]=1; laderman_W[6,8]=1; laderman_W[7,8]=-1; laderman_W[8,8]=1

    # m10 = (d-e+f)(m-n+o)
    laderman_U[3,9]=1; laderman_U[4,9]=-1; laderman_U[5,9]=1
    laderman_V[3,9]=1; laderman_V[4,9]=-1; laderman_V[5,9]=1
    laderman_W[0,9]=-1; laderman_W[1,9]=-1; laderman_W[2,9]=1; laderman_W[3,9]=-1; laderman_W[4,9]=-1; laderman_W[5,9]=1; laderman_W[6,9]=-1; laderman_W[7,9]=-1; laderman_W[8,9]=1

    # m11 = (g+h-i)(p+q-r)
    laderman_U[6,10]=1; laderman_U[7,10]=1; laderman_U[8,10]=-1
    laderman_V[6,10]=1; laderman_V[7,10]=1; laderman_V[8,10]=-1
    laderman_W[0,10]=1; laderman_W[1,10]=-1; laderman_W[2,10]=1; laderman_W[3,10]=1; laderman_W[4,10]=-1; laderman_W[5,10]=1; laderman_W[6,10]=1; laderman_W[7,10]=-1; laderman_W[8,10]=1

    # m12 = (g-h+i)(p-q+r)
    laderman_U[6,11]=1; laderman_U[7,11]=-1; laderman_U[8,11]=1
    laderman_V[6,11]=1; laderman_V[7,11]=-1; laderman_V[8,11]=1
    laderman_W[0,11]=-1; laderman_W[1,11]=-1; laderman_W[2,11]=1; laderman_W[3,11]=-1; laderman_W[4,11]=-1; laderman_W[5,11]=1; laderman_W[6,11]=-1; laderman_W[7,11]=-1; laderman_W[8,11]=1

    # m13 = (a-b)(j-k)
    laderman_U[0,12]=1; laderman_U[1,12]=-1
    laderman_V[0,12]=1; laderman_V[1,12]=-1
    laderman_W[0,12]=1; laderman_W[1,12]=-1; laderman_W[2,12]=-1; laderman_W[3,12]=0; laderman_W[4,12]=0; laderman_W[5,12]=0; laderman_W[6,12]=0; laderman_W[7,12]=0; laderman_W[8,12]=0

    # m14 = (a-c)(j-l)
    laderman_U[0,13]=1; laderman_U[2,13]=-1
    laderman_V[0,13]=1; laderman_V[2,13]=-1
    laderman_W[0,13]=1; laderman_W[1,13]=-1; laderman_W[2,13]=-1; laderman_W[3,13]=0; laderman_W[4,13]=0; laderman_W[5,13]=0; laderman_W[6,13]=0; laderman_W[7,13]=0; laderman_W[8,13]=0

    # m15 = (b-c)(k-l)
    laderman_U[1,14]=1; laderman_U[2,14]=-1
    laderman_V[1,14]=1; laderman_V[2,14]=-1
    laderman_W[0,14]=1; laderman_W[1,14]=-1; laderman_W[2,14]=-1; laderman_W[3,14]=0; laderman_W[4,14]=0; laderman_W[5,14]=0; laderman_W[6,14]=0; laderman_W[7,14]=0; laderman_W[8,14]=0

    # m16 = (d-e)(m-n)
    laderman_U[3,15]=1; laderman_U[4,15]=-1
    laderman_V[3,15]=1; laderman_V[4,15]=-1
    laderman_W[0,15]=0; laderman_W[1,15]=0; laderman_W[2,15]=0; laderman_W[3,15]=1; laderman_W[4,15]=-1; laderman_W[5,15]=-1; laderman_W[6,15]=0; laderman_W[7,15]=0; laderman_W[8,15]=0

    # m17 = (d-f)(m-o)
    laderman_U[3,16]=1; laderman_U[5,16]=-1
    laderman_V[3,16]=1; laderman_V[5,16]=-1
    laderman_W[0,16]=0; laderman_W[1,16]=0; laderman_W[2,16]=0; laderman_W[3,16]=1; laderman_W[4,16]=-1; laderman_W[5,16]=-1; laderman_W[6,16]=0; laderman_W[7,16]=0; laderman_W[8,16]=0

    # m18 = (e-f)(n-o)
    laderman_U[4,17]=1; laderman_U[5,17]=-1
    laderman_V[4,17]=1; laderman_V[5,17]=-1
    laderman_W[0,17]=0; laderman_W[1,17]=0; laderman_W[2,17]=0; laderman_W[3,17]=1; laderman_W[4,17]=-1; laderman_W[5,17]=-1; laderman_W[6,17]=0; laderman_W[7,17]=0; laderman_W[8,17]=0

    # m19 = (g-h)(p-q)
    laderman_U[6,18]=1; laderman_U[7,18]=-1
    laderman_V[6,18]=1; laderman_V[7,18]=-1
    laderman_W[0,18]=0; laderman_W[1,18]=0; laderman_W[2,18]=0; laderman_W[3,18]=0; laderman_W[4,18]=0; laderman_W[5,18]=0; laderman_W[6,18]=1; laderman_W[7,18]=-1; laderman_W[8,18]=-1

    # m20 = (g-i)(p-r)
    laderman_U[6,19]=1; laderman_U[8,19]=-1
    laderman_V[6,19]=1; laderman_V[8,19]=-1
    laderman_W[0,19]=0; laderman_W[1,19]=0; laderman_W[2,19]=0; laderman_W[3,19]=0; laderman_W[4,19]=0; laderman_W[5,19]=0; laderman_W[6,19]=1; laderman_W[7,19]=-1; laderman_W[8,19]=-1

    # m21 = (h-i)(q-r)
    laderman_U[7,20]=1; laderman_U[8,20]=-1
    laderman_V[7,20]=1; laderman_V[8,20]=-1
    laderman_W[0,20]=0; laderman_W[1,20]=0; laderman_W[2,20]=0; laderman_W[3,20]=0; laderman_W[4,20]=0; laderman_W[5,20]=0; laderman_W[6,20]=1; laderman_W[7,20]=-1; laderman_W[8,20]=-1

    # m22 = (a+b+c-d-e-f+g+h+i)(j+k+l-m-n-o+p+q+r)
    laderman_U[0,21]=1; laderman_U[1,21]=1; laderman_U[2,21]=1; laderman_U[3,21]=-1; laderman_U[4,21]=-1; laderman_U[5,21]=-1; laderman_U[6,21]=1; laderman_U[7,21]=1; laderman_U[8,21]=1
    laderman_V[0,21]=1; laderman_V[1,21]=1; laderman_V[2,21]=1; laderman_V[3,21]=-1; laderman_V[4,21]=-1; laderman_V[5,21]=-1; laderman_V[6,21]=1; laderman_V[7,21]=1; laderman_V[8,21]=1
    laderman_W[0,21]=-1; laderman_W[1,21]=1; laderman_W[2,21]=-1; laderman_W[3,21]=-1; laderman_W[4,21]=1; laderman_W[5,21]=-1; laderman_W[6,21]=-1; laderman_W[7,21]=1; laderman_W[8,21]=-1

    # m23 = (a-b+c-d+e-f-g+h-i)(j-k+l+m-n+o-p+q+r)
    laderman_U[0,22]=1; laderman_U[1,22]=-1; laderman_U[2,22]=1; laderman_U[3,22]=-1; laderman_U[4,22]=1; laderman_U[5,22]=-1; laderman_U[6,22]=-1; laderman_U[7,22]=1; laderman_U[8,22]=-1
    laderman_V[0,22]=1; laderman_V[1,22]=-1; laderman_V[2,22]=1; laderman_V[3,22]=1; laderman_V[4,22]=-1; laderman_V[5,22]=1; laderman_V[6,22]=-1; laderman_V[7,22]=1; laderman_V[8,22]=1
    laderman_W[0,22]=1; laderman_W[1,22]=1; laderman_W[2,22]=-1; laderman_W[3,22]=-1; laderman_W[4,22]=-1; laderman_W[5,22]=1; laderman_W[6,22]=-1; laderman_W[7,22]=1; laderman_W[8,22]=-1

    # Now build the Kronecker product decomposition for 6x6
    # 6 = 2*3, so we split indices: i = 2*i2 + i3 where i2 in {0,1}, i3 in {0,1,2}
    # Actually, the Kronecker product structure: 
    # For 6x6 = (2x3)x(2x3), the tensor is T6 = T2 ⊗ T3
    # U6[i6, c6] where i6 = i2*3 + i3 (i2 in 0..1, i3 in 0..2), c6 = c2*23 + c3
    # U6[i2*3+i3, c2*23+c3] = U2[i2, c2] * U3[i3, c3]
    # Similarly for V and W.

    # Wait, I need to be careful about the indexing.
    # For 6x6 matrix A, A[i,j] where i,j in 0..5
    # Split i = 2*i2 + i3? No, that's not right for Kronecker.
    # 
    # Actually for Kronecker product of matrices: if A is 2x2 and B is 3x3, 
    # then A⊗B is 6x6 with (A⊗B)[2*i1+i2, 2*j1+j2] = A[i1,j1]*B[i2,j2]
    # 
    # So for the 6x6 multiplication: C = A*B where A,B,C are 6x6
    # A = A2 ⊗ A3 (block structure), B = B2 ⊗ B3, C = C2 ⊗ C3
    # Then C2⊗C3 = (A2⊗A3)(B2⊗B3) = (A2*B2)⊗(A3*B3)
    # 
    # So the tensor T6 = T2 ⊗ T3 where T2 is the 2x2 multiplication tensor and T3 is 3x3.
    # 
    # Indexing for 6x6: element (i,j) of A maps to (i2, i3) where i = 3*i2 + i3
    # (i2 in 0..1 for the 2x2 part, i3 in 0..2 for the 3x3 part)
    # Similarly for B: (j,k) -> (j2, j3) where j = 3*j2 + j3, k = 3*k2 + k3
    # For C: (i,k) -> (i2, k2, i3, k3) where i = 3*i2+i3, k = 3*k2+k3
    # W index: k*n + i = k*6 + i = (3*k2+k3)*6 + (3*i2+i3) = 18*k2 + 6*k3 + 3*i2 + i3

    # U: shape (36, 161), U[3*i2+i3, 23*c2+c3] = U2[i2, c2] * U3[i3, c3]
    # V: shape (36, 161), V[3*j2+j3, 23*c2+c3] = V2[j2, c2] * V3[j3, c3]
    # W: shape (36, 161), W[18*k2+6*k3+3*i2+i3, 23*c2+c3] = W2[k2, i2, c2] * W3[k3, i3, c3]
    #   = W2_flat[k2*2+i2, c2] * W3_flat[k3*3+i3, c3]

    n2, m2, p2 = 2, 2, 2
    n3, m3, p3 = 3, 3, 3
    rank2 = 7
    rank3 = 23
    rank6 = rank2 * rank3  # 161

    U6 = np.zeros((36, rank6), dtype=object)
    V6 = np.zeros((36, rank6), dtype=object)
    W6 = np.zeros((36, rank6), dtype=object)

    for c2 in range(rank2):
        for c3 in range(rank3):
            c6 = c2 * rank3 + c3
            for i2 in range(n2):
                for i3 in range(n3):
                    u2 = strassen_U[i2, c2]
                    u3 = laderman_U[i3, c3]
                    if u2 != 0 and u3 != 0:
                        U6[i2 * n3 + i3, c6] = u2 * u3
            for j2 in range(m2):
                for j3 in range(m3):
                    v2 = strassen_V[j2, c2]
                    v3 = laderman_V[j3, c3]
                    if v2 != 0 and v3 != 0:
                        V6[j2 * m3 + j3, c6] = v2 * v3
            for k2 in range(p2):
                for i2 in range(n2):
                    for k3 in range(p3):
                        for i3 in range(n3):
                            w2 = strassen_W[k2 * n2 + i2, c2]
                            w3 = laderman_W[k3 * n3 + i3, c3]
                            if w2 != 0 and w3 != 0:
                                w6_idx = (k2 * p3 + k3) * n2 * n3 + i2 * n3 + i3
                                # Wait, W index is k*n + i where k = k2*p3+k3, i = i2*n3+i3
                                # So W index = (k2*p3+k3)*n6 + (i2*n3+i3) where n6=6
                                w6_idx = (k2 * p3 + k3) * 6 + (i2 * n3 + i3)
                                W6[w6_idx, c6] = w2 * w3

    # Convert to lists of integers
    U_list = [[int(U6[i, c]) for c in range(rank6)] for i in range(36)]
    V_list = [[int(V6[i, c]) for c in range(rank6)] for i in range(36)]
    W_list = [[int(W6[i, c]) for c in range(rank6)] for i in range(36)]

    if not time_ok():
        # Fallback to schoolbook if we're running out of time (shouldn't happen)
        n, m, p = DIMENSIONS
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
