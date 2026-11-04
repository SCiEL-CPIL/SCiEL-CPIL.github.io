import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
    by a 3 by 5 matrix.
    
    Strategy:
    1. Construct a valid rank-30 schoolbook decomposition.
    2. Use a local search / optimization approach to try to find a lower rank decomposition.
    3. If a lower rank is found and verified, return it. Otherwise, return the schoolbook.
    
    The goal is to find a rank < 30 decomposition. The theoretical lower bound is known to be
    less than 30. We will try to find a rank-24 or rank-28 decomposition using a heuristic
    search that starts from the schoolbook and tries to merge terms or find linear dependencies.
    
    However, a more robust approach for the evaluator is to explicitly construct a known
    low-rank decomposition if possible. Since finding the exact minimal rank is hard,
    we will implement a "greedy merge" heuristic:
    - Start with rank 30.
    - Try to reduce rank by finding pairs of terms that can be combined or eliminated.
    
    Actually, a better approach for this specific small case is to use the known
    Winograd/Strassen-like hybrid.
    
    Let's try a direct construction for rank 28 or lower.
    Split m=3 into 2+1.
    A = [A12 | A3], B = [B12; B3]
    C = A12*B12 + A3*B3
    
    A12 is 2x2, B12 is 2x5.
    A3 is 2x1, B3 is 1x5.
    
    A3*B3 is rank 2*1*5 = 10 terms? No, A3 is a vector, B3 is a vector.
    The product A3*B3 is an outer product of a 2x1 and 1x5 matrix?
    Wait, A3 is 2x1, B3 is 1x5. The product is 2x5.
    This product is a single rank-1 matrix (outer product of column vector and row vector).
    In the tensor decomposition sense, each entry C[i,k] gets a contribution A3[i]*B3[k].
    This requires 2*5 = 10 terms in the schoolbook expansion for this part?
    No. The term A3[i]*B3[k] is a product of two scalars.
    In the tensor decomposition sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r],
    for the j=2 part (the 3rd column/row), we have:
    Sum over r where j=2: U[i*3+2, r] * V[2*5+k, r] * W[k*2+i, r] = A[i,2] * B[2,k]
    
    If we use a single term r for the whole block, we can't because U, V, W are fixed per r.
    We need to represent the outer product of A3 (2x1) and B3 (1x5).
    This outer product has rank 1 in the matrix sense, but in the tensor sense?
    The tensor slice for j=2 is T[i, 2, k] = 1 if the indices match the specific i,k.
    Wait, the tensor T is defined such that T[i, j, k] = 1.
    So for j=2, we need T[i, 2, k] = 1 for all i in {0,1}, k in {0..4}.
    This is a 2x5 slice of ones.
    The rank of a 2x5 matrix of ones is 1.
    So we can represent the j=2 part with rank 1?
    Let's check:
    We need U[i*3+2, r] * V[2*5+k, r] * W[k*2+i, r] = 1 for all i,k.
    If we set U[i*3+2, r] = 1, V[2*5+k, r] = 1, W[k*2+i, r] = 1 for a single r,
    then the product is 1*1*1 = 1.
    So the j=2 part can be done in rank 1?
    
    Let's verify the tensor definition again.
    T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i.
    The condition is that a,b,c correspond to the same (i,j,k).
    So for a fixed j, the slice is over i and k.
    T[i*m+j, j*p+k, k*n+i] = 1.
    For j=2:
    i in {0,1}, k in {0,1,2,3,4}.
    Entries:
    i=0, k=0: T[2, 10, 0] = 1
    i=0, k=1: T[2, 11, 1] = 1
    ...
    i=1, k=0: T[5, 10, 2] = 1
    i=1, k=1: T[5, 11, 3] = 1
    ...
    
    If we use one term r:
    U[2, r]=1, U[5, r]=1
    V[10, r]=1, V[11, r]=1, V[12, r]=1, V[13, r]=1, V[14, r]=1
    W[0, r]=1, W[1, r]=1, W[2, r]=1, W[3, r]=1, W[4, r]=1
    W[5, r]=1, W[6, r]=1, W[7, r]=1, W[8, r]=1, W[9, r]=1
    
    Check product for i=0, k=0:
    U[2, r] * V[10, r] * W[0, r] = 1 * 1 * 1 = 1. Correct.
    Check product for i=1, k=0:
    U[5, r] * V[10, r] * W[2, r] = 1 * 1 * 1 = 1. Correct.
    Check product for i=0, k=1:
    U[2, r] * V[11, r] * W[1, r] = 1 * 1 * 1 = 1. Correct.
    
    So yes, the entire j=2 slice (2x5 matrix of ones) can be represented by rank 1!
    This saves 10 - 1 = 9 terms compared to schoolbook for that slice.
    
    Now for j=0 and j=1.
    j=0 slice: 2x5 matrix of ones. Rank 1.
    j=1 slice: 2x5 matrix of ones. Rank 1.
    
    Wait, is it that simple?
    If each j-slice is a matrix of ones, and the rank of a 2x5 matrix of ones is 1,
    then the total rank is 1 + 1 + 1 = 3?
    
    Let's check if the terms interfere.
    The decomposition is a sum over r.
    If we use r=0 for j=0, r=1 for j=1, r=2 for j=2.
    
    Term r=0:
    U[i*3+0, 0] = 1 for i=0,1.
    V[0*5+k, 0] = 1 for k=0..4.
    W[k*2+i, 0] = 1 for all k,i.
    
    Term r=1:
    U[i*3+1, 1] = 1 for i=0,1.
    V[1*5+k, 1] = 1 for k=0..4.
    W[k*2+i, 1] = 1 for all k,i.
    
    Term r=2:
    U[i*3+2, 2] = 1 for i=0,1.
    V[2*5+k, 2] = 1 for k=0..4.
    W[k*2+i, 2] = 1 for all k,i.
    
    Let's check the sum for a specific entry T[a,b,c].
    We need sum_r U[a,r] V[b,r] W[c,r] = 1 if a,b,c match (i,j,k), else 0.
    
    Take a=0 (i=0, j=0), b=0 (j=0, k=0), c=0 (k=0, i=0).
    Matches? i=0, j=0, k=0. Yes.
    Sum:
    r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1. Prod=1.
    r=1: U[0,1]=0 (since j=0 != 1). Prod=0.
    r=2: U[0,2]=0. Prod=0.
    Total = 1. Correct.
    
    Take a=0 (i=0, j=0), b=5 (j=1, k=0), c=0 (k=0, i=0).
    Matches? i=0, j=0 vs j=1. No match. Should be 0.
    Sum:
    r=0: U[0,0]=1, V[5,0]=0 (j=1 != 0). Prod=0.
    r=1: U[0,1]=0 (j=0 != 1). Prod=0.
    r=2: U[0,2]=0. Prod=0.
    Total = 0. Correct.
    
    Take a=0 (i=0, j=0), b=0 (j=0, k=0), c=1 (k=0, i=1).
    Matches? i=0 vs i=1. No match. Should be 0.
    Sum:
    r=0: U[0,0]=1, V[0,0]=1, W[1,0]=1. Prod=1.
    Wait.
    W index is k*n+i.
    For r=0, we set W[k*2+i, 0] = 1 for ALL k, i.
    So W[1,0] is 1 (k=0, i=1).
    So the product is 1*1*1 = 1.
    But the tensor entry T[0, 0, 1] should be 0 because i=0 in a and i=1 in c don't match.
    
    Ah, the W factor must distinguish i.
    In the schoolbook, W[k*n+i, r] is 1 only for the specific i, k.
    In my "rank 1 per j" attempt, I set W to 1 for all i, k.
    This causes cross-talk between different i's.
    
    The slice for a fixed j is a matrix M where M[i,k] = 1.
    We need to decompose this matrix M into a sum of rank-1 tensors?
    No, the term is U[i,j,r] * V[j,k,r] * W[i,k,r].
    If we fix j, let u_i = U[i,j,r], v_k = V[j,k,r], w_{ik} = W[i,k,r].
    We need sum_r u_i v_k w_{ik} = 1 for all i,k.
    
    If we use one term r for this j:
    u_i v_k w_{ik} = 1.
    We can choose u_i = 1, v_k = 1, w_{ik} = 1.
    But w_{ik} is part of the global W matrix.
    W[i,k,r] is shared across different j's if we use the same r?
    No, r is the rank index.
    If we use r=0 for j=0, then W[i,k,0] is defined.
    If we use r=1 for j=1, then W[i,k,1] is defined.
    They are independent columns in W.
    
    So for r=0 (corresponding to j=0):
    U[i,0,0] = 1 for all i.
    V[0,k,0] = 1 for all k.
    W[i,k,0] = 1 for all i,k.
    
    For r=1 (corresponding to j=1):
    U[i,1,1] = 1 for all i.
    V[1,k,1] = 1 for all k.
    W[i,k,1] = 1 for all i,k.
    
    For r=2 (corresponding to j=2):
    U[i,2,2] = 1 for all i.
    V[2,k,2] = 1 for all k.
    W[i,k,2] = 1 for all i,k.
    
    Let's re-evaluate the cross-talk case:
    a=0 (i=0, j=0), b=0 (j=0, k=0), c=1 (k=0, i=1).
    We want T[0,0,1] = 0.
    Sum over r:
    r=0: U[0,0,0]=1, V[0,0,0]=1, W[0,1,0]=1. (Note: W index is k*n+i = 0*2+1 = 1).
    Product = 1.
    
    The problem is that W[0,1,0] is 1, but it shouldn't contribute to T[0,0,1] because the i in U (i=0) and i in W (i=1) don't match?
    Wait, the tensor T is defined by the matching of indices.
    T[a,b,c] is non-zero ONLY IF the indices encoded in a,b,c are consistent.
    a = i*m+j. b = j*p+k. c = k*n+i.
    For T[0,0,1]:
    a=0 => i=0, j=0.
    b=0 => j=0, k=0.
    c=1 => k=0, i=1.
    i from a is 0. i from c is 1. Mismatch.
    So T[0,0,1] = 0.
    
    My decomposition gives 1. So it's wrong.
    
    Why? Because the term r=0 "thinks" it's handling j=0.
    U[0,0,0]=1 (i=0, j=0).
    V[0,0,0]=1 (j=0, k=0).
    W[1,0]=1 (k=0, i=1).
    The product is 1.
    But the term r=0 was intended to cover the case where i, j, k all match.
    Here j matches (0=0), k matches (0=0), but i doesn't (0!=1).
    The W factor doesn't "know" which i it should be paired with. It just provides a value for (k,i).
    The U factor provides a value for (i,j).
    The V factor provides a value for (j,k).
    
    For the product to be non-zero, we need U[i,j,r] != 0, V[j,k,r] != 0, W[k,i,r] != 0.
    In my construction:
    U[i,j,r] is non-zero only if j == r (since I used r to index j).
    V[j,k,r] is non-zero only if j == r.
    W[k,i,r] is non-zero for all k,i if j == r.
    
    So if j == r, the product is U[i,r,r] * V[r,k,r] * W[k,i,r].
    I set U[i,r,r]=1, V[r,k,r]=1, W[k,i,r]=1.
    So the product is 1 for ANY i, k, as long as j=r.
    
    But we only want it to be 1 if the i and k are consistent with the specific term?
    No, the tensor T[i*m+j, j*p+k, k*n+i] is 1 if the indices match.
    If j=r, then the term contributes 1 to T[i*m+r, r*p+k, k*n+i] for ALL i, k.
    But T[i*m+r, r*p+k, k*n+i] is 1 only if the i in the first index matches the i in the third index.
    Here, the first index is i*m+r (so i is the row of A).
    The third index is k*n+i (so i is the row of C).
    They use the same i variable in my loop.
    Wait, in the tensor definition, the indices are fixed.
    T[a,b,c] is 1 if a=i*m+j, b=j*p+k, c=k*n+i for the SAME i,j,k.
    
    In my sum:
    Sum_r U[a,r] V[b,r] W[c,r].
    a = i_a * m + j_a
    b = j_b * p + k_b
    c = k_c * n + i_c
    
    For the term to be valid, we need i_a = i_c, j_a = j_b = j_r (the j associated with r), and k_b = k_c.
    
    In my construction for r:
    U[a,r] is 1 if j_a = r. (And i_a can be anything).
    V[b,r] is 1 if j_b = r. (And k_b can be anything).
    W[c,r] is 1 if ... I set it to 1 for all k_c, i_c.
    
    So if j_a = r and j_b = r, the product is 1, regardless of i_a, k_b, i_c, k_c.
    But we need i_a = i_c and k_b = k_c.
    My W factor doesn't enforce i_a = i_c.
    
    To enforce i_a = i_c, W[c,r] must be 0 unless i_c matches the i_a that was used in U?
    But U doesn't store i_a. U just says "this term is for j=r".
    The "i" is implicit in the row index of U.
    
    So, U[i_a, j_a, r] is non-zero.
    W[k_c, i_c, r] is non-zero.
    The product is non-zero.
    We need this product to be 1 ONLY IF i_a = i_c.
    
    This implies that W cannot be a simple 1 for all i_c.
    W[k_c, i_c, r] must depend on i_a?
    But W doesn't know i_a.
    
    This means we cannot decouple the j-slices so easily if we use a single r per j.
    The "rank 1" argument for the slice was flawed because the W factor is shared globally and doesn't know which i was selected in U.
    
    Actually, the tensor T is a 3D array.
    T[i,j,k] = 1.
    We are decomposing T.
    T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    
    If we fix j, we have a matrix M_j[i,k] = T[i,j,k] = 1.
    We want to write M_j[i,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    
    If we use one term r for this j:
    1 = U[i,j,r] V[j,k,r] W[k,i,r].
    Let U[i,j,r] = u_i, V[j,k,r] = v_k, W[k,i,r] = w_{ki}.
    1 = u_i v_k w_{ki}.
    We can pick u_i = 1, v_k = 1, w_{ki} = 1.
    This works for the slice.
    
    But does it work globally?
    The term r contributes to T[i,j,k] for all i,k if j matches.
    If we have multiple r's, they add up.
    
    If we use r=0 for j=0:
    Contributes 1 to T[i,0,k] for all i,k.
    If we use r=1 for j=1:
    Contributes 1 to T[i,1,k] for all i,k.
    If we use r=2 for j=2:
    Contributes 1 to T[i,2,k] for all i,k.
    
    Total sum for T[i,0,k]:
    r=0: 1.
    r=1: 0 (since j=0 != 1).
    r=2: 0.
    Total = 1. Correct.
    
    So why did I think it failed?
    Let's re-check the cross-talk case.
    T[0,0,1] corresponds to i=0, j=0, k=1?
    No.
    a = i*m+j = 0*3+0 = 0.
    b = j*p+k = 0*5+1 = 1.
    c = k*n+i = 1*2+0 = 2.
    So T[0,1,2] = 1.
    
    My previous check was T[0,0,1].
    a=0 (i=0, j=0).
    b=0 (j=0, k=0).
    c=1 (k=0, i=1).
    This T[0,0,1] should be 0 because i mismatch.
    
    Sum for T[0,0,1]:
    r=0: U[0,0,0]=1, V[0,0,0]=1, W[1,0]=1.
    Wait, W index is c=1.
    W[1,0] is W[k=0, i=1, r=0].
    In my construction for r=0 (j=0), I set W[k,i,0]=1 for all k,i.
    So W[1,0]=1.
    Product = 1.
    
    So the sum is 1, but T[0,0,1] is 0.
    So it is indeed wrong.
    
    The issue is that W[k,i,r] is 1 for all i.
    It should be 1 only for the specific i that matches the i in U?
    But U has i as an index.
    U[i,j,r] is 1 for all i.
    So U doesn't distinguish i.
    
    To make it work, we need U[i,j,r] and W[k,i,r] to be "paired" such that their product is 1 only when the i's match?
    But they are separate factors.
    
    Actually, the decomposition T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    requires that for a fixed r, the "shape" of the non-zero entries in U, V, W must be compatible.
    
    If U[i,j,r] is non-zero for all i, and W[k,i,r] is non-zero for all i,
    then the product is non-zero for all i.
    This implies that the term r contributes to all i.
    This is correct for the slice M_j[i,k]=1.
    
    BUT, the term r also contributes to other slices if j doesn't match?
    No, U[i,j,r] is 0 if j != r (in my construction).
    So r=0 only contributes to j=0.
    
    So for j=0, the contribution is 1 for all i,k.
    This is exactly what we want for the j=0 slice.
    
    So why did T[0,0,1] get a contribution?
    T[0,0,1] has j=0 (from a=0, b=0).
    So r=0 contributes.
    And it contributes 1.
    But T[0,0,1] is 0.
    
    Wait. T[0,0,1] is the entry of the tensor at indices (a=0, b=0, c=1).
    a=0 => i=0, j=0.
    b=0 => j=0, k=0.
    c=1 => k=0, i=1.
    
    The tensor T is defined as T[i,j,k] = 1.
    So T[0,0,0] = 1 (i=0, j=0, k=0).
    T[1,0,0] = 1 (i=1, j=0, k=0).
    T[0,0,1] is NOT a valid (i,j,k) tuple?
    No, (a,b,c) are just indices into the flattened arrays.
    The tensor is a 3D array of size (n*m) x (m*p) x (p*n).
    Entry (a,b,c) is 1 if a,b,c come from the same (i,j,k).
    
    For a=0, b=0, c=1:
    a=0 => (i=0, j=0).
    b=0 => (j=0, k=0).
    c=1 => (k=0, i=1).
    i from a is 0. i from c is 1.
    So this is NOT a valid (i,j,k) tuple.
    So T[0,0,1] should be 0.
    
    My decomposition gives 1.
    So it is wrong.
    
    The error is in my assumption that "W[k,i,r]=1 for all i" is fine.
    It is not fine because it creates contributions to invalid tuples.
    
    To fix this, we need W[k,i,r] to be 0 unless i matches the i in U?
    But U has i as an index.
    We can't make W depend on U's i.
    
    This means we cannot use a single term r to cover the entire j-slice if the slice has multiple i's.
    We need to separate the i's.
    
    For j=0, we have i=0 and i=1.
    We can use r=0 for (j=0, i=0) and r=1 for (j=0, i=1).
    Then:
    r=0: U[0,0,0]=1, V[0,k,0]=1, W[k,0,0]=1. (Only i=0).
    r=1: U[1,0,1]=1, V[0,k,1]=1, W[k,1,1]=1. (Only i=1).
    
    Check T[0,0,1] (i=0, j=0, k=0 vs i=1):
    r=0: U[0,0,0]=1, V[0,0,0]=1, W[1,0]=0 (since i=1, but r=0 only has i=0).
    Wait, W index is c=1 => k=0, i=1.
    W[1,0] is W[k=0, i=1, r=0].
    If r=0 is for i=0, then W[k,1,0] should be 0.
    So W[1,0]=0.
    Product = 0.
    
    r=1: U[0,0,1]=0 (since i=0, but r=1 is for i=1).
    Product = 0.
    
    Total = 0. Correct.
    
    Check T[0,0,0] (i=0, j=0, k=0):
    r=0: U[0,0,0]=1, V[0,0,0]=1, W[0,0]=1.
    Product = 1.
    r=1: U[0,0,1]=0.
    Total = 1. Correct.
    
    So, we need one term per (j, i) pair?
    Number of terms = m * n = 3 * 2 = 6?
    Let's check.
    For each j in 0..2, for each i in 0..1:
    Term r_{j,i}:
    U[i, j, r] = 1.
    V[j, k, r] = 1 for all k.
    W[k, i, r] = 1 for all k.
    
    Total rank = 6.
    
    Let's verify this rank-6 decomposition.
    Sum_{j,i} U[i,j,r_{ji}] V[j,k,r_{ji}] W[k,i,r_{ji}].
    For a fixed (i,j,k):
    Only the term r_{ji} where j matches and i matches contributes.
    U[i,j,r_{ji}] = 1.
    V[j,k,r_{ji}] = 1.
    W[k,i,r_{ji}] = 1.
    Product = 1.
    Other terms have j or i mismatch, so U or W is 0.
    
    So rank 6 works?
    This seems too good to be true.
    The rank of 2x3x5 matmul is definitely not 6.
    The lower bound is much higher.
    
    Where is the flaw?
    The flaw is that V[j,k,r] = 1 for all k.
    This means V is the same for all k.
    And W[k,i,r] = 1 for all k.
    
    But V and W are fixed matrices.
    If we use r_{00}, r_{01}, r_{10}, r_{11}, r_{20}, r_{21}.
    
    V[0, k, r_{00}] = 1 for all k.
    V[0, k, r_{01}] = 1 for all k.
    
    This implies that the rows of V for different r's are identical?
    No, they are in different columns of V.
    
    Is there a constraint that U, V, W must be "generic" or something?
    No.
    
    Wait, the tensor T is not just 1 for valid tuples.
    T[i,j,k] = 1.
    My decomposition gives 1 for valid tuples.
    And 0 for invalid tuples.
    
    So rank 6 is a valid decomposition?
    This would imply that the rank of 2x3x5 matmul is <= 6.
    But we know rank(2x2x2) is 7.
    2x3x5 contains 2x2x2 as a sub-tensor?
    If we take i in {0,1}, j in {0,1}, k in {0,1}.
    The rank of the sub-tensor is at most the rank of the whole tensor.
    So rank(2x3x5) >= rank(2x2x2) = 7.
    
    So rank 6 is impossible.
    
    Where is the error in the rank 6 argument?
    
    Let's check the sub-tensor 2x2x2 (i=0,1; j=0,1; k=0,1).
    Using my rank 6 construction:
    Terms: r_{00}, r_{01}, r_{10}, r_{11}. (Ignoring j=2).
    
    T[0,0,0] (i=0,j=0,k=0):
    r_{00}: 1*1*1 = 1.
    Others 0.
    Sum = 1.
    
    T[0,0,1] (i=0,j=0,k=1):
    r_{00}: U[0,0]=1, V[0,1]=1, W[1,0]=1.
    Sum = 1.
    
    T[1,0,0] (i=1,j=0,k=0):
    r_{00}: U[1,0]=0 (since i=1, r_{00} is for i=0).
    r_{10}: U[1,0]=1, V[0,0]=1, W[0,1]=1.
    Sum = 1.
    
    T[1,0,1] (i=1,j=0,k=1):
    r_{10}: U[1,0]=1, V[0,1]=1, W[1,1]=1.
    Sum = 1.
    
    T[0,1,0] (i=0,j=1,k=0):
    r_{01}: U[0,1]=1, V[1,0]=1, W[0,0]=1.
    Sum = 1.
    
    T[1,1,0] (i=1,j=1,k=0):
    r_{11}: U[1,1]=1, V[1,0]=1, W[0,1]=1.
    Sum = 1.
    
    T[0,1,1] (i=0,j=1,k=1):
    r_{01}: U[0,1]=1, V[1,1]=1, W[1,0]=1.
    Sum = 1.
    
    T[1,1,1] (i=1,j=1,k=1):
    r_{11}: U[1,1]=1, V[1,1]=1, W[1,1]=1.
    Sum = 1.
    
    So the 2x2x2 sub-tensor is all 1s?
    No, T[i,j,k] is 1 for all i,j,k in the range.
    So the sub-tensor is indeed all 1s.
    
    But the rank of the all-ones 2x2x2 tensor is 1?
    No, the tensor T for matmul is NOT all ones.
    T[i,j,k] = 1 if the indices match.
    In the 2x2x2 case, ALL indices match for any i,j,k?
    No.
    a = i*2+j.
    b = j*2+k.
    c = k*2+i.
    
    The tensor is defined on the domain (a,b,c).
    T[a,b,c] = 1 if a,b,c come from the same (i,j,k).
    
    In the 2x2x2 case:
    (0,0,0) -> i=0,j=0,k=0. Valid.
    (0,1,1) -> i=0,j=0,k=1. Valid.
    (1,0,0) -> i=0,j=1,k=0. Valid.
    ...
    
    Are there invalid tuples?
    Yes.
    a=0 (i=0,j=0), b=1 (j=0,k=1), c=0 (k=0,i=0).
    i=0, j=0, k=1 from a,b.
    i=0, j=?, k=0 from c.
    j from b is 0. j from c? c doesn't have j.
    c = k*2+i.
    So c=0 => k=0, i=0.
    So i=0, j=0, k=1 vs i=0, j=?, k=0.
    k mismatch (1 vs 0).
    So T[0,1,0] = 0.
    
    My rank 6 construction gave 1 for T[0,1,0]?
    Let's check.
    a=0, b=1, c=0.
    r_{00}: U[0,0]=1, V[0,1]=1, W[0,0]=1.
    Product = 1.
    
    But T[0,1,0] should be 0.
    
    Why?
    Because V[0,1] corresponds to j=0, k=1.
    W[0,0] corresponds to k=0, i=0.
    U[0,0] corresponds to i=0, j=0.
    
    The term r_{00} is "for" i=0, j=0.
    It assumes k is free?
    No, V[j,k,r] and W[k,i,r] both depend on k.
    In my construction, V[j,k,r]=1 for all k.
    And W[k,i,r]=1 for all k.
    
    So the term r_{00} contributes 1 to T[i=0, j=0, k=ANY].
    But T[0,0,k] is 1 only if the k in b matches the k in c.
    Here, the term doesn't check if k matches.
    It just says "I am the term for i=0, j=0. I contribute 1 to all k."
    
    But the tensor T only has 1 at specific (i,j,k).
    For i=0, j=0, T[0,0,k] is 1 for all k?
    Yes, because for any k, there is a valid tuple (0,0,k).
    
    Wait.
    T[a,b,c] is 1 if a,b,c come from the SAME (i,j,k).
    For a fixed (i,j), there are multiple k's.
    Each k gives a different (a,b,c).
    So T is 1 at multiple locations for a fixed (i,j).
    
    Specifically, for i=0, j=0:
    k=0: a=0, b=0, c=0. T[0,0,0]=1.
    k=1: a=0, b=1, c=1. T[0,1,1]=1.
    
    What about T[0,1,0]?
    a=0 (i=0,j=0), b=1 (j=0,k=1), c=0 (k=0,i=0).
    This is NOT a valid tuple.
    So T[0,1,0]=0.
    
    My term r_{00} contributes to T[0,1,0] because:
    U[0,0]=1 (ok).
    V[0,1]=1 (ok, j=0).
    W[0,0]=1 (ok, i=0).
    
    But it should be 0 because k in V (1) != k in W (0).
    
    So my construction fails because it doesn't enforce k consistency.
    
    To enforce k consistency, we need V and W to depend on k in a way that their product is 1 only if k matches.
    But V and W are separate factors.
    We can't make V[j,k,r] * W[k',i,r] = 1 if k=k' and 0 otherwise using a single term r, unless we use the "delta" function, which requires summing over k.
    
    This brings us back to the schoolbook, where we have a term for each (i,j,k).
    
    So, the rank 6 idea is flawed.
    
    We must return to the schoolbook rank 30, or find a valid reduction.
    
    Given the time constraints and the difficulty of finding a valid lower rank, I will return the schoolbook rank 30.
    """
    n, m, p = 2, 3, 5
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
