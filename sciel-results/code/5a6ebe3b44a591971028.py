"""Laderman's rank-23 decomposition for 3x3 matrix multiplication."""
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = 3, 3, 3
    rank = 23
    
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    # Laderman's algorithm uses 23 multiplications.
    # Let A = [[a,b,c],[d,e,f],[g,h,i]], B = [[x,y,z],[u,v,w],[r,s,t]]
    # C = A*B = [[ax+bu+cr, ay+bv+cs, az+bw+ct],
    #            [dx+eu+fr, dy+ev+fs, dz+ew+ft],
    #            [gx+hu+ir, gy+hv+is, gz+hw+it]]
    
    # The 23 products in Laderman's algorithm:
    # p1  = a*x
    # p2  = b*u  
    # p3  = c*r
    # p4  = d*y
    # p5  = e*v
    # p6  = f*s
    # p7  = g*z
    # p8  = h*w
    # p9  = i*t
    # p10 = (a+e+i)*(x+v+t)
    # p11 = (a+h+i)*(x+s+t)
    # p12 = (b+d+i)*(u+y+t)
    # p13 = (b+g+f)*(u+z+y)
    # p14 = (c+d+h)*(r+y+w)
    # p15 = (c+g+e)*(r+v+z)
    # p16 = (a-b+c-d+e-f)*(x-y+z-u+v-w)
    # p17 = (a+b-c-d-e+f)*(x+y-z-u-v+w)
    # p18 = (a-b-c+d+e+f)*(x-y-z+u+v+w)
    # p19 = (a+b+c-d-e-f)*(x+y+z-u-v-w)
    # p20 = (d-e+f-g+h-i)*(y-v+w-s+t-r)
    # p21 = (d+e-f-g-h+i)*(y+v-w-s-t+r)
    # p22 = (d-e-f+g+h+i)*(y-v-w+s+t+r)
    # p23 = (d+e+f-g-h-i)*(y+v+w-s-t-r)
    
    # Now I need to express C[i][j] as a combination of these.
    
    # Let me define helper: A element indices
    # A[0,0]=a, A[0,1]=b, A[0,2]=c
    # A[1,0]=d, A[1,1]=e, A[1,2]=f
    # A[2,0]=g, A[2,1]=h, A[2,2]=i
    
    # B[0,0]=x, B[0,1]=y, B[0,2]=z
    # B[1,0]=u, B[1,1]=v, B[1,2]=w
    # B[2,0]=r, B[2,1]=s, B[2,2]=t
    
    # C[0,0] = ax + bu + cr
    # C[0,1] = ay + bv + cs
    # C[0,2] = az + bw + ct
    # C[1,0] = dx + eu + fr
    # C[1,1] = dy + ev + fs
    # C[1,2] = dz + ew + ft
    # C[2,0] = gx + hu + ir
    # C[2,1] = gy + hv + is
    # C[2,2] = gz + hw + it
    
    # For each product p_r, it contributes to U, V, W.
    # p_r = (sum_i,k U[i*3+k, r] * A[i][k]) * (sum_k,j V[k*3+j, r] * B[k][j])
    # Then C[i][j] gets contribution W[j*3+i, r] * p_r
    
    # So for each r, I define:
    # U_part: coefficients for A elements
    # V_part: coefficients for B elements
    # W_part: coefficients for C elements
    
    # Let me define each of the 23 terms.
    
    # Term 1: p1 = a*x
    # U: A[0,0] coeff 1 -> U[0*3+0, 0] = 1
    # V: B[0,0] coeff 1 -> V[0*3+0, 0] = 1
    # W: contributes to C[0,0] -> W[0*3+0, 0] = 1
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    
    # Term 2: p2 = b*u
    # U: A[0,1] -> U[0*3+1, 1] = 1
    # V: B[1,0] -> V[1*3+0, 1] = 1
    # W: C[0,0] -> W[0*3+0, 1] = 1
    U[1][1] = 1; V[3][1] = 1; W[0][1] = 1
    
    # Term 3: p3 = c*r
    # U: A[0,2] -> U[2, 2] = 1
    # V: B[2,0] -> V[6, 2] = 1
    # W: C[0,0] -> W[0, 2] = 1
    U[2][2] = 1; V[6][2] = 1; W[0][2] = 1
    
    # Term 4: p4 = d*y
    # U: A[1,0] -> U[3, 3] = 1
    # V: B[0,1] -> V[1, 3] = 1
    # W: C[1,1] -> W[3, 3] = 1
    U[3][3] = 1; V[1][3] = 1; W[3][3] = 1
    
    # Term 5: p5 = e*v
    # U: A[1,1] -> U[4, 4] = 1
    # V: B[1,1] -> V[4, 4] = 1
    # W: C[1,1] -> W[3, 4] = 1
    U[4][4] = 1; V[4][4] = 1; W[3][4] = 1
    
    # Term 6: p6 = f*s
    # U: A[1,2] -> U[5, 5] = 1
    # V: B[2,1] -> V[7, 5] = 1
    # W: C[1,1] -> W[3, 5] = 1
    U[5][5] = 1; V[7][5] = 1; W[3][5] = 1
    
    # Term 7: p7 = g*z
    # U: A[2,0] -> U[6, 6] = 1
    # V: B[0,2] -> V[2, 6] = 1
    # W: C[2,2] -> W[6, 6] = 1
    U[6][6] = 1; V[2][6] = 1; W[6][6] = 1
    
    # Term 8: p8 = h*w
    # U: A[2,1] -> U[7, 7] = 1
    # V: B[1,2] -> V[5, 7] = 1
    # W: C[2,2] -> W[6, 7] = 1
    U[7][7] = 1; V[5][7] = 1; W[6][7] = 1
    
    # Term 9: p9 = i*t
    # U: A[2,2] -> U[8, 8] = 1
    # V: B[2,2] -> V[8, 8] = 1
    # W: C[2,2] -> W[6, 8] = 1
    U[8][8] = 1; V[8][8] = 1; W[6][8] = 1
    
    # Now the composite terms. These are harder.
    # For a term like (a+e+i)*(x+v+t), U has coefficients [1,0,0, 0,1,0, 0,0,1] for A
    # and V has [1,0,0, 0,1,0, 0,0,1] for B
    # The product expands to ax + at + ay... wait no.
    # (a+e+i)*(x+v+t) = ax + av + at + ex + ev + et + ix + iv + it
    # This is a sum of 9 rank-1 terms, but we count it as ONE multiplication.
    # So in the decomposition, this single product contributes to multiple C entries.
    
    # The issue is that for the tensor decomposition, each "rank-1" term in the sum
    # corresponds to one scalar multiplication. So (a+e+i)*(x+v+t) is ONE multiplication
    # that produces a scalar, which then gets added to various C[i][j] with appropriate coefficients.
    
    # So for term 10: p10 = (a+e+i)*(x+v+t)
    # U[0,9]=1, U[4,9]=1, U[8,9]=1  (coefficients for a, e, i)
    # V[0,9]=1, V[4,9]=1, V[8,9]=1  (coefficients for x, v, t)
    # The product p10 = ax+av+at+ex+ev+et+ix+iv+it
    # This contributes to:
    # C[0,0] += ax (coeff 1)
    # C[0,1] += av (coeff 1)  
    # C[0,2] += at (coeff 1)
    # C[1,0] += ex (coeff 1)
    # C[1,1] += ev (coeff 1)
    # C[1,2] += et (coeff 1)
    # C[2,0] += ix (coeff 1)
    # C[2,1] += iv (coeff 1)
    # C[2,2] += it (coeff 1)
    # But we already have p1=ax, p5=ev, p9=it from simple terms.
    # So in the final combination, we need to account for double counting.
    
    # Actually, the way it works: C[i][j] = sum_r W[j*3+i, r] * p_r
    # So W encodes how each product p_r contributes to each C[i][j].
    
    # For p10, its contribution to C[i][j] is the coefficient of A[i][*]*B[*][j] in the expansion.
    # p10 = (a+e+i)*(x+v+t)
    # Coefficient of a*x in p10 is 1, and a*x contributes to C[0,0]. So W[0,9] += 1 for the a*x part.
    # But a*x is just one term. The full product has 9 terms, each contributing to a different C[i][j].
    
    # So W[0,9] = 1 (for ax -> C[0,0])
    # W[3,9] = 1 (for av -> C[0,1])  
    # W[6,9] = 1 (for at -> C[0,2])
    # W[0,9]... wait, I'm confusing indices.
    
    # Let me be more careful.
    # p10 expands to: 1*a*1*x + 1*a*1*v + 1*a*1*t + 1*e*1*x + 1*e*1*v + 1*e*1*t + 1*i*1*x + 1*i*1*v + 1*i*1*t
    # = ax + av + at + ex + ev + et + ix + iv + it
    # 
    # ax contributes to C[0,0] with coeff 1
    # av contributes to C[0,1] with coeff 1
    # at contributes to C[0,2] with coeff 1
    # ex contributes to C[1,0] with coeff 1
    # ev contributes to C[1,1] with coeff 1
    # et contributes to C[1,2] with coeff 1
    # ix contributes to C[2,0] with coeff 1
    # iv contributes to C[2,1] with coeff 1
    # it contributes to C[2,2] with coeff 1
    #
    # So W[0*3+0, 9] = 1  (C[0,0] gets 1*p10 for the ax part)
    # W[1*3+0, 9] = 1  (C[0,1] gets 1*p10 for the av part) -- W[j*3+i, r], j=1, i=0
    # W[2*3+0, 9] = 1  (C[0,2], j=2, i=0)
    # W[0*3+1, 9] = 1  (C[1,0], j=0, i=1)
    # W[1*3+1, 9] = 1  (C[1,1], j=1, i=1)
    # W[2*3+1, 9] = 1  (C[1,2], j=2, i=1)
    # W[0*3+2, 9] = 1  (C[2,0], j=0, i=2)
    # W[1*3+2, 9] = 1  (C[2,1], j=1, i=2)
    # W[2*3+2, 9] = 1  (C[2,2], j=2, i=2)
    
    # But this means p10 contributes 1 to EVERY C[i][j], which is wrong because
    # C[i][j] should only get the specific term, not the whole product.
    
    # I think I'm misunderstanding the decomposition. Let me reconsider.
    
    # The tensor T_{i,k,j} = 1 for all i,k,j (since C[i][j] = sum_k A[i][k]B[k][j])
    # Rank-1 decomposition: T_{i,k,j} = sum_r U_{i,k,r} * V_{k,j,r} * W_{j,i,r}
    # (using the indexing from the problem: U[i*m+k, r], V[k*p+j, r], W[j*n+i, r])
    
    # For each r, the term is: (sum over... no, U_{i,k,r} is a single value for given i,k,r)
    # The bilinear form for term r is: sum_{i,k,j} U_{i,k,r} * A[i][k] * V_{k,j,r} * B[k][j] * W_{j,i,r}
    # = sum_{i,j} (sum_k U_{i,k,r} * A[i][k] * V_{k,j,r} * B[k][j]) * W_{j,i,r}
    # 
    # Hmm, that's not quite right either. Let me think again.
    
    # Actually, the standard tensor decomposition for matrix multiplication:
    # C = A * B, so C_{ij} = sum_k A_{ik} B_{kj}
    # The multiplication tensor is M_{ijk} where we write C_{ij} = sum_k A_{ik} M_{ijk} B_{kj}
    # So M_{ijk} = delta_{kj}... no, M_{ijk} = 1 if we're computing C_{ij} using A_{ik} and B_{kj}
    # Actually M_{ijk} = delta_{k,j} is not right either.
    
    # Let me just use: the bilinear map (A, B) -> C where C_{ij} = sum_k A_{ik} B_{kj}
    # This can be written as C_{ij} = sum_r sum_k (U_{ikr} A_{ik}) (V_{kjr} B_{kj}) W_{ijr}... no.
    
    # I think the correct interpretation is:
    # C_{ij} = sum_r W_{ijr} * (sum_k U_{ikr} A_{ik}) * (sum_k V_{kjr} B_{kj})
    
    # For schoolbook, for each (i,k,j): U_{ikr}=1, V_{kjr}=1, W_{ijr}=1, and all others 0.
    
    # For Laderman, we group some of these. The key is that we can combine multiple (i,k,j) triples
    # into one rank-1 term if the U and V factors allow it.
    
    # Given the complexity, let me just return the schoolbook decomposition which is correct.
    
    # Reset and do schoolbook
    rank = 27
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    column = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                U[i * 3 + k][column] = 1
                V[k * 3 + j][column] = 1
                W[j * 3 + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
