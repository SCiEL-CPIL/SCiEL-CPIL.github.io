"""Matrix multiplication 2x3x5: rank-minimal exact tensor decomposition."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# Known Strassen-based decomposition for 2x3x5 with rank 28
# We use a constructive approach: partition into 2x2 blocks and apply Strassen.
# A is 2x3, B is 3x5.
# Split A = [A1 A2] where A1 is 2x2, A2 is 2x1
# Split B = [B1; B2] where B1 is 2x5, B2 is 1x5
# C = A1*B1 + A2*B2
# A1*B1 is 2x2 * 2x5 -> use Strassen for the 2x2 part times each column
# Actually, let's think more carefully.

# A (2x3) * B (3x5) = C (2x5)
# Let's split the 3 columns of A into 2+1, and 3 rows of B into 2+1.
# A = [A_left (2x2) | A_right (2x1)]
# B = [B_top (2x5); B_bottom (1x5)]
# C = A_left @ B_top + A_right @ B_bottom

# A_left @ B_top: 2x2 times 2x5. We can compute this using 7 Strassen multiplications
# for the 2x2 block, where the right operand has 5 columns.
# Actually, Strassen works for 2x2 * 2x2. For 2x2 * 2x5, we can split the 5 columns
# into 2+2+1 and apply Strassen to each 2x2 * 2x2 block, plus a direct 2x2 * 2x1.
# That gives 7 + 7 + 4 = 18 multiplications for A_left @ B_top.
# Plus A_right @ B_bottom: 2x1 * 1x5 = 2 multiplications (each element of A_right times row of B_bottom).
# Total: 18 + 2 = 20? That's better than 30 but maybe not optimal.

# Actually, let me reconsider. The known rank for 2x3x5 is 24 or 25.
# Let me try a different partitioning.

# Alternative: Use the fact that we can find low-rank decompositions.
# The tensor is T[i,j,k] = delta[i,k] (for matrix mult (i,j) of A and (j,k) of B).
# Wait, for 2x3 * 3x5: C[i,k] = sum_j A[i,j] * B[j,k]
# So T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
# The tensor has indices (i, j, k) with i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4}.
# T[i,j,k] = 1 for all valid (i,j,k) - it's the all-ones tensor in this case? No.
# Actually the matrix multiplication tensor T has T[i,j,k] = 1 if the (i,j) entry of A 
# and (j,k) entry of B contribute to (i,k) entry of C. Since every (i,j,k) triple 
# with valid indices contributes, T[i,j,k] = 1 for all i,j,k.
# Wait no - the tensor is defined such that C[i,k] = sum_j A[i,j] B[j,k].
# The "tensor" T_{ijk} = delta_{ik}... no. 
# In tensor decomposition for matrix mult: T[i,j,k] = 1 if the product A[i,j]*B[j,k] 
# appears in C[i,k]. Since C[i,k] = sum_j A[i,j]*B[j,k], every (i,j,k) contributes.
# So T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
# The rank of the all-ones tensor of size n*m*p is... well, it's not that simple.
# Actually the matrix multiplication tensor is NOT the all-ones tensor.
# T[i,j,k] = 1 if we multiply A[i,j] and B[j,k] to get C[i,k].
# The condition is that the row index of A (i) matches the row index of C (i), 
# and the column index of B (k) matches the column index of C (k), and the inner 
# indices match (j=j). So T[i,j,k] = 1 for all valid i,j,k. Yes, it IS all ones.

# So we need to decompose the all-ones tensor of shape (2,3,5) into rank r.
# A rank-1 term is u_i * v_j * w_k. We need sum_{c=1}^r u_c[i] * v_c[j] * w_c[k] = 1 for all i,j,k.

# This is equivalent to finding the tensor rank of the all-ones tensor J_{2,3,5}.
# Known result: rank of J_{n,m,p} is min(n*m, m*p, n*p) for certain cases, but generally
# it's more complex. For J_{2,3,5}: 
# - Trivial upper bound: 2*3*5 = 30
# - Better: we can do 2*5 = 10 by fixing i and k: sum over c corresponding to (i,k) pairs, 
#   with u_c[i]=1, v_c[j]=1, w_c[k]=1. That gives rank 10.
# - Even better: use structure. 

# Actually, rank of all-ones tensor J_{n,m,p}: 
# We need u_c[i] v_c[j] w_c[k] summed = 1 for all i,j,k.
# One way: pick r = max(n,p) and set up appropriately.
# For n=2, p=5: rank <= 5? Let's see.
# If we set r=5, we need 5 vectors u_c in R^2, v_c in R^3, w_c in R^5 such that
# sum_c u_c[i] v_c[j] w_c[k] = 1 for all i,j,k.
# For fixed k, sum_c u_c[i] v_c[j] w_c[k] = 1. This is a 2x3 matrix that equals all-ones,
# expressed as sum of w_c[k] * (u_c v_c^T).
# If we choose w_c as standard basis e_k, then for each k we need sum_c u_c[i] v_c[j] delta_{c,k} = 1,
# meaning u_k v_k^T = J_{2x3} (all ones). But rank of J_{2x3} is 1, so u_k v_k^T = J requires
# u_k = (1,1)^T, v_k = (1,1,1)^T. Then for each k, we use one term. Total rank = 5.
# Check: sum_{k=0}^{4} u_k[i] v_k[j] w_k[k'] where w_k = e_k.
# = sum_k u_k[i] v_k[j] delta_{k,k'} = u_{k'}[i] v_{k'}[j] = 1*1 = 1. Yes!
# So rank <= 5 for J_{2,3,5}?? That seems too good.

# Wait, I think I'm confusing the tensor structure. Let me re-examine.
# The matrix multiplication tensor for (n,m,p) dimensions:
# T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
# T[i,j,k] = 1 (it's the all-ones tensor)
# And we want T = sum_c U_c V_c W_c^T where U_c is n-vector, V_c is m-vector, W_c is p-vector.
# I.e., T[i,j,k] = sum_c U_c[i] * V_c[j] * W_c[k] = 1 for all i,j,k.

# My construction above: r=5, for each k in 0..4:
# U_k = [1,1]^T (n=2 vector), V_k = [1,1,1]^T (m=3 vector), W_k = e_k (p=5 vector)
# Then T[i,j,k'] = sum_{k=0}^{4} U_k[i] V_k[j] W_k[k'] = sum_k 1*1*delta_{k,k'} = 1. 
# So rank = 5!

# But wait - the task says the output factor follows W[k*n+i, r]. Let me re-read.
# "The output factor follows the official trace convention W[k*n+i,r]"
# And shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
# So U has n*m = 6 rows, V has m*p = 15 rows, W has p*n = 10 rows.
# 
# The decomposition: C[i,k] = sum_j A[i,j] B[j,k]
# In tensor form: the "tensor" T[i,j,k] represents the contribution.
# U is indexed by (i,j) pairs -> U[i*m+j, c]
# V is indexed by (j,k) pairs -> V[j*p+k, c]  
# W is indexed by (k,i) pairs -> W[k*n+i, c]
# 
# And the reconstruction: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should give
# the "coefficient" for A[i,j]*B[j,k] contributing to C[i,k].
# For standard matrix mult, this coefficient is 1 for all valid (i,j,k).
# So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.

# This is exactly the all-ones tensor decomposition I described!
# With my construction: rank = 5.
# U[i*m+j, c] = 1 for all i,j when c=k (since U_k[i]=1 for all i, and it doesn't depend on j... 
# wait, U is indexed by (i,j). In my construction U_c is an n-vector, but here U has n*m rows.

# Let me redo: We need U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c = 1.
# 
# Construction with rank = p = 5:
# For each c = 0..4 (corresponding to k=c):
#   U[i*m+j, c] = 1 for all i, j
#   V[j*p+k, c] = 1 if k == c, else 0  (i.e., V[j*p+c, c] = 1, V[j*p+k, c] = 0 for k != c)
#   W[k'*n+i, c] = 1 for all k', i
# 
# Check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
# = sum_c 1 * (1 if k==c else 0) * 1 = 1. Correct!
# 
# So rank = 5 works! But can we do better?
# 
# Similarly with rank = n = 2:
# For each c = 0..1 (corresponding to i=c):
#   U[i*m+j, c] = 1 if i == c, else 0
#   V[j*p+k, c] = 1 for all j, k
#   W[k*n+i, c] = 1 if i == c, else 0
# Check: sum_c (1 if i==c else 0) * 1 * (1 if i==c else 0) = 1 if i==i else 0 = 1. Correct!
# So rank = 2 works!
# 
# Can we do rank = 1?
# U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1 for all i,j,k.
# This means U[i*m+j] = a_{ij}, V[j*p+k] = b_{jk}, W[k*n+i] = c_{ki} such that
# a_{ij} * b_{jk} * c_{ki} = 1 for all i,j,k.
# From a_{ij} * b_{jk} * c_{ki} = 1 and a_{ij} * b_{jk} * c_{k'i} = 1, we get c_{ki} = c_{k'i} for all k,k'.
# Similarly all a_{ij} are equal, all b_{jk} are equal, all c_{ki} are equal.
# Let a_{ij} = a, b_{jk} = b, c_{ki} = c. Then a*b*c = 1.
# So U = a * ones, V = b * ones, W = c * ones with abc = 1. This works!
# Rank = 1!

# Wait, that can't be right. Let me double-check.
# If U[i*m+j, 0] = a for all i,j, V[j*p+k, 0] = b for all j,k, W[k*n+i, 0] = c for all k,i,
# then sum_c=1 U * V * W = a * b * c for all (i,j,k). We need this to equal 1, so abc = 1.
# Yes! Rank 1 works!

# But that seems way too simple. Let me re-read the problem statement.
# "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 5 matrix."
# "exactly decompose the matrix multiplication tensor"
# 
# Hmm, I think the issue is that the matrix multiplication tensor is NOT the all-ones tensor in the
# way I've been thinking. Let me reconsider.
# 
# In the standard formulation, the matrix multiplication tensor T^{(n,m,p)} has:
# T[i, j, k] = 1 if the (i,j)-entry of A and (j,k)-entry of B multiply to contribute to (i,k)-entry of C.
# 
# For C = AB where A is n x m and B is m x p:
# C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
# 
# So the "tensor" T has T[i,j,k] = 1 for all valid i,j,k (since every term contributes).
# And the decomposition T = sum_c U^{(c)} V^{(c)} W^{(c)} where:
# T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
# 
# Wait, the W index is (k,i) not (i,k). The problem says W[k*n+i, r].
# 
# So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i in [0,2), j in [0,3), k in [0,5).
# 
# My rank-1 construction: U = a*ones(6), V = b*ones(15), W = c*ones(10), abc=1.
# This gives sum = abc = 1. So rank 1 works!
# 
# But this is suspiciously simple. Is the answer really rank 1?
# 
# Actually, I think I need to reconsider what "decompose the matrix multiplication tensor" means.
# Perhaps the tensor is not all-ones. Let me think again...
# 
# Actually, I believe the standard matrix multiplication tensor IS the all-ones tensor in this 
# indexing convention. The "structure" comes from how you index the factors, not from the tensor values.
# 
# But rank 1 seems too trivial for this to be an interesting problem. Let me reconsider.
# 
# OH WAIT. I think I misunderstand. The tensor T[i,j,k] might not be all ones. 
# In some formulations, the matrix multiplication "tensor" is defined differently.
# 
# Let me re-read: "exactly decompose the matrix multiplication tensor"
# The matrix multiplication tensor T satisfies: for any matrices A (n x m) and B (m x p),
# C = A * B where C[i,k] = sum_j A[i,j] B[j,k].
# 
# The tensor T is defined by T[i,j,k] = 1 (the coefficient of A[i,j]B[j,k] in C[i,k]).
# So yes, T is the all-ones tensor, and my rank-1 construction should work.
# 
# Unless... the problem is asking for something else. Let me look at the seed program.
# The seed program uses rank = n*m*p = 30 with U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
# for each (i,j,k) triple. This is the "schoolbook" decomposition.
# 
# If rank 1 worked, the seed would have been rank 1. So I must be misunderstanding something.
# 
# Let me reconsider: maybe the constraint is that the entries must be 0 or 1? No, it says
# "integer/half-integer entries".
# 
# Or maybe the tensor is NOT all ones. Let me think about what tensor actually needs to be decomposed.
# 
# Actually, I think the issue is more subtle. The matrix multiplication tensor in the context
# of tensor rank is usually defined as follows:
# T^{(n,m,p)} = sum_{i,j,k} e_i^{(n)} ⊗ e_j^{(m)} ⊗ e_k^{(p)} 
# where the sum is over all i,j,k such that... hmm, no.
# 
# Let me look at this from a different angle. The standard result is that the rank of the
# matrix multiplication tensor for 2x2x2 is 7 (Strassen), not 1. If the tensor were all-ones,
# rank 1 would work, contradicting the known result that rank(T^{(2,2,2)}) = 7.
# 
# So the matrix multiplication tensor is NOT the all-ones tensor!
# 
# Let me reconsider. For 2x2x2:
# C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
# C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
# C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
# C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
# 
# The tensor T[i,j,k] where i is row of C, j is... wait, I need to be more careful.
# 
# In the standard tensor formulation for matrix multiplication:
# We have three spaces: A-space (n*m dimensions), B-space (m*p dimensions), C-space (n*p dimensions).
# The tensor T lives in A^* ⊗ B^* ⊗ C (or similar).
# T[α, β, γ] = 1 if the α-th entry of A, β-th entry of B, and γ-th entry of C are related
# by matrix multiplication.
# 
# Specifically, if we index A by (i,j) -> α = i*m+j, B by (j,k) -> β = j*p+k, C by (i,k) -> γ = i*p+k:
# T[α, β, γ] = 1 if α=(i,j), β=(j',k), γ=(i',k') and i=i', j=j', k=k'.
# 
# So T[i*m+j, j'*p+k, i'*p+k'] = 1 if i=i', j=j', k=k'.
# This means T[α, β, γ] = 1 iff the (i,j) of A, (j,k) of B, and (i,k) of C form a valid triple.
# 
# In the problem's indexing: U has n*m rows (indexed by i*m+j), V has m*p rows (indexed by j*p+k),
# W has p*n rows (indexed by k*n+i). Note W is indexed by (k,i) not (i,k).
# 
# So the condition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
# for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p).
# 
# This IS the all-ones condition! And for 2x2x2, this would give rank 1, contradicting Strassen.
# 
# I must be wrong about something. Let me look at this more carefully.
# 
# Actually, I think the correct formulation is:
# The matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k (in the range),
# and the rank of this tensor (as a 3-way tensor) is what we're computing.
# For 2x2x2, the all-ones tensor J_{2,2,2} has rank... let me compute.
# 
# J_{2,2,2}[i,j,k] = 1 for all i,j,k in {0,1}.
# Can we write this as sum of r rank-1 tensors?
# Rank 1: u_i v_j w_k. We need sum_c u_c[i] v_c[j] w_c[k] = 1 for all i,j,k.
# With r=1: u[i]v[j]w[k] = 1 for all i,j,k. This requires u, v, w to be constant (all 1s).
# So u=[1,1], v=[1,1], w=[1,1]. Then u[i]v[j]w[k] = 1. Rank 1 works!
# 
# But this contradicts the known result that the matrix multiplication tensor for 2x2x2 has rank 7!
# 
# I think the resolution is that the "matrix multiplication tensor" in the tensor rank literature
# is NOT the all-ones tensor. It's a different tensor.
# 
# The correct matrix multiplication tensor T^{(n,m,p)} is defined as:
# T^{(n,m,p)} = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
# where e_i, e_j, e_k are basis vectors... no, that's the all-ones tensor again.
# 
# Actually, I recall now: the matrix multiplication tensor is:
# T^{(n,m,p)} = sum_{i,j,k} E_{ij}^{(n,m)} ⊗ E_{jk}^{(m,p)} ⊗ E_{ik}^{(n,p)}
# where E_{ij} are matrix units. But as a tensor in the appropriate space, this IS the all-ones
# tensor when you flatten the indices.
# 
# I'm confused. Let me just go with the problem statement and the seed program.
# The seed program uses rank 30 = 2*3*5, which is the "trivial" decomposition.
# The knowledge says Strassen reduces it to 28.
# So the answer is somewhere between 24 and 30, and rank 1 is NOT valid.
# 
# This means my understanding of the tensor is wrong. Let me re-examine.
# 
# AH, I think I see the issue. The tensor is NOT all-ones. The condition is:
# sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{condition}
# where the condition involves the indices in a specific way.
# 
# Actually, for matrix multiplication, the "tensor" that needs to be decomposed satisfies:
# For the multiplication to work, we need the decomposition to reproduce the bilinear form.
# The tensor T[i,j,k] = 1 for all valid (i,j,k) IS correct, and the rank of this tensor
# as a 3-way array IS what we want.
# 
# But then rank 1 works, which contradicts known results. Unless... the entries of U, V, W
# are constrained to be 0 or 1? The problem says "integer/half-integer entries", so that's not it.
# 
# Let me just try rank 1 and see if it's accepted. If not, I'll reconsider.
# Actually, let me re-read the problem more carefully.
# 
# "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 5 matrix."
# 
# I think the key insight I'm missing is that the tensor is not simply "all ones". 
# Let me think about what "decompose the matrix multiplication tensor" means in terms of
# the actual computation.
# 
# In a rank-r decomposition, we compute r intermediate products:
# M_c = (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
# Then C[i,k] = sum_c M_c * W[k*n+i, c]
# 
# For this to equal sum_j A[i,j] * B[j,k] for all A, B:
# sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j' V[j'*p+k, c] * B[j',k])
# = sum_j A[i,j] * B[j,k]
# 
# Expanding: sum_c sum_j sum_j' W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] * A[i,j] * B[j',k]
# = sum_j A[i,j] * B[j,k]
# 
# For this to hold for all A, B:
# sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = delta_{j,j'} (Kronecker delta)
# 
# THIS is the correct condition! It's NOT all-ones! It's the Kronecker delta!
# 
# So we need: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = delta_{j,j'}
# for all i, j, j', k.
# 
# This makes much more sense! For 2x2x2, this is the standard matrix multiplication tensor
# with rank 7 (Strassen).
# 
# Now let me reformulate:
# We need to find U (6 x r), V (15 x r), W (10 x r) such that:
# sum_c U[i*3+j, c] * V[j'*5+k, c] * W[k*2+i, c] = 1 if j==j', else 0
# for all i in {0,1}, j,j' in {0,1,2}, k in {0,1,2,3,4}.
# 
# This is a system of 2*3*3*5 = 90 equations (but many are the same due to symmetry).
# Actually, the condition must hold for all (i, j, j', k), giving 2*3*3*5 = 90 constraints,
# but the equation only depends on (j, j') through delta_{j,j'}, and the LHS must be independent
# of i and k for a given (j, j').
# 
# Wait, that's not quite right. The LHS depends on i, j, j', k. The RHS is delta_{j,j'}.
# So for fixed j, j', the LHS must equal delta_{j,j'} for ALL i, k.
# 
# This means: for each (j, j'), the function f_{j,j'}(i, k) = sum_c U[i*3+j, c] * V[j'*5+k, c] * W[k*2+i, c]
# must be constant (equal to delta_{j,j'}) over all i, k.
# 
# This is a much more interesting problem!
# 
# For the schoolbook decomposition (rank 30):
# U[i*3+j, (i,j,k)] = 1, V[j*5+k, (i,j,k)] = 1, W[k*2+i, (i,j,k)] = 1
# Then sum_c U[i*3+j, c] V[j'*5+k, c] W[k*2+i, c] = 1 if there exists c=(i,j,k) with 
# matching indices, i.e., j=j' (and i,i and k,k match automatically since c is determined by (i,j,k)).
# Wait, c is a triple (i,j,k). For a given (i,j,j',k), c=(i,j,k) matches if j=j'.
# So the sum is 1 if j=j', else 0. Correct!
# 
# Now, can we do better than 30?
# 
# Key insight: The condition must hold for all i, k. So the dependence on i and k must "cancel out"
# in some sense. 
# 
# Let's think about this differently. For fixed j, j', we need:
# sum_c U[i*3+j, c] * V[j'*5+k, c] * W[k*2+i, c] = delta_{j,j'} for all i, k.
# 
# This is a 2x5 matrix (indexed by i, k) that must be constant (all entries = delta_{j,j'}).
# 
# One approach: If we can make U[i*3+j, c] independent of i, V[j'*5+k, c] independent of k,
# and W[k*2+i, c] have a specific structure...
# 
# Actually, let's try a construction. Suppose we use rank = m * something.
# 
# Alternative approach: Use the fact that the problem separates in some way.
# 
# Let me try: rank = 3 * 5 = 15.
# For each (j, k) pair (15 pairs), define a term c = (j, k):
# U[i*3+j', c] = delta_{j', j} (depends on j' only, not i)
# V[j'*5+k', c] = delta_{k', k} (depends on k' only, not j')  
# W[k''*2+i', c] = ?
# 
# Then sum_c U[i*3+j, c] V[j'*5+k, c] W[k*2+i, c]
# = sum_{(j0,k0)} delta_{j,j0} * delta_{k,k0} * W[k*2+i, (j0,k0)]
# = delta_{j,j} * delta_{k,k} * W[k*2+i, (j,k)]  -- wait, this doesn't work because j' and j are different.
# 
# Let me redo: c = (j0, k0) where j0 in {0,1,2}, k0 in {0,1,2,3,4}.
# U[i*3+j', c] = delta_{j', j0}
# V[j'*5+k', c] = delta_{k', k0}  
# Then: sum_c U[i*3+j, c] V[j'*5+k, c] W[k*2+i, c]
# = sum_{j0,k0} delta_{j,j0} * delta_{k,k0} * W[k*2+i, (j0,k0)]
# = W[k*2+i, (j,k)]
# 
# We need this to equal delta_{j,j'} for all i, k. But W[k*2+i, (j,k)] depends on i and k,
# while delta_{j,j'} doesn't. So this doesn't work unless W is very special.
# 
# We need W[k*2+i, (j,k)] = delta_{j,j'} for all i,k, but j' is not in the index of W.
# This approach fails.
# 
# Let me try another construction. Rank = 3 * 2 = 6.
# For each (j, i) pair (6 pairs), c = (j0, i0):
# U[i*3+j', c] = delta_{j', j0} * delta_{i, i0}
# V[j'*5+k, c] = 1 (independent of j', k)
# W[k*2+i', c] = delta_{i', i0}
# 
# Sum: sum_{j0,i0} delta_{j,j0} delta_{i,i0} * 1 * delta_{i,i0}
# = sum_{j0,i0} delta_{j,j0} delta_{i,i0} delta_{i,i0}
# = delta_{j,j} * 1 (since delta_{i,i0}^2 = delta_{i,i0})
# = 1
# 
# This gives 1 for all (j, j'), not delta_{j,j'}. Fails.
# 
# Let me try yet another approach. The key constraint is that the result must be delta_{j,j'}.
# This suggests we need some "orthogonality" in the j index.
# 
# Construction with rank = 3:
# c = j0 in {0,1,2}:
# U[i*3+j', c] = delta_{j', j0} * u_{i, j0}
# V[j'*5+k, c] = delta_{j', j0} * v_{j0, k}  -- wait, V is indexed by (j', k)
# 
# Hmm, let me be more careful.
# 
# Actually, I think the right approach is:
# For the condition sum_c U[i*3+j, c] V[j'*5+k, c] W[k*2+i, c] = delta_{j,j'} to hold for all i,k,
# we can try to make the i and k dependencies factorize.
# 
# Suppose W[k*2+i, c] = w_c * a_c[i] * b_c[k] (separable in i and k).
# And U[i*3+j, c] = u_c[i, j], V[j'*5+k, c] = v_c[j', k].
# 
# Then: sum_c u_c[i,j] v_c[j',k] w_c a_c[i] b_c[k] = delta_{j,j'} for all i,k.
# 
# This is getting complex. Let me just try to use known results or a search approach.
# 
# Given the time constraints, let me implement a reasonable decomposition.
# The known rank for 2x3x5 matrix multiplication is 24 (I believe).
# 
# Actually, let me try a simpler constructive approach:
# 
# For each j in {0,1,2} (3 values), we need to handle the "diagonal" in j.
# The off-diagonal terms (j != j') must give 0.
# 
# One approach: Use 3 "blocks" corresponding to j=0,1,2, and within each block,
# handle the 2x5 multiplication.
# 
# For block j0: we need to compute the contribution where A's column j0 times B's row j0.
# This is a 2x5 outer product: A[:, j0] * B[j0, :].
# This can be done with rank 2*5 = 10 using the standard approach, or rank max(2,5) = 5 
# using a smarter approach.
# 
# Actually, the outer product of a 2-vector and a 5-vector has tensor rank 1 (it's a rank-1 matrix).
# But in our tensor formulation, we need to embed this into the full structure.
# 
# Let me try: For each j0 in {0,1,2}, and for each (i, k) in {0,1}x{0,1,2,3,4}, we could
# have a separate term. That's 3*2*5 = 30 (the schoolbook). 
# 
# To reduce, we can share terms across i or k.
# 
# For fixed j0, the contribution to C[i,k] is A[i,j0] * B[j0,k].
# We can write this as: for each i in {0,1}, sum over k of A[i,j0] * B[j0,k].
# Or: for each k in {0..4}, sum over i of A[i,j0] * B[j0,k].
# 
# The "rank" needed for the 2x5 outer product structure:
# We need U[i*3+j0, c] * V[j0*5+k, c] * W[k*2+i, c] to reproduce A[i,j0]*B[j0,k] * (some weight).
# 
# If we use 5 terms for each j0 (one for each k):
# c = (j0, k0):
# U[i*3+j, c] = delta_{j,j0} * 1
# V[j'*5+k', c] = delta_{j',j0} * delta_{k',k0}
# W[k''*2+i', c] = 1
# 
# Sum over c: sum_{j0,k0} delta_{j,j0} delta_{j',j0} delta_{k',k0} * 1
# = delta_{j,j'} * delta_{j,j0} * 1 (summing over j0 gives 1 if j=j', else 0)
# Wait: sum_{j0,k0} delta_{j,j0} delta_{j',j0} delta_{k',k0} = delta_{j,j'} * 1 (if j=j') * (sum over k0 of delta_{k',k0} = 1)
# = delta_{j,j'}. 
# 
# But this doesn't depend on i or k, and we need it to work for all i,k. Let me recheck.
# 
# sum_c U[i*3+j, c] V[j'*5+k, c] W[k*2+i, c]
# = sum_{j0=0}^{2} sum_{k0=0}^{4} [delta_{j,j0} * 1] * [delta_{j',j0} * delta_{k,k0}] * [1]
# = sum_{j0,k0} delta_{j,j0} delta_{j',j0} delta_{k,k0}
# = delta_{j,j'} * 1 (since sum_{k0} delta_{k,k0} = 1)
# = delta_{j,j'}
# 
# This works! And the rank is 3*5 = 15.
# 
# But wait, I need to double-check that this gives the correct matrix multiplication.
# The condition sum_c U[i*3+j,c] V[j'*5+k,c] W[k*2+i,c] = delta_{j,j'} ensures that
# C[i,k] = sum_j A[i,j] B[j,k]. Let me verify:
# 
# In the decomposition, C[i,k] = sum_c W[k*2+i,c] * (sum_j U[i*3+j,c] A[i,j]) * (sum_j' V[j'*5+k,c] B[j',k])
# 
# With our construction:
# For c=(j0,k0):
# sum_j U[i*3+j,c] A[i,j] = sum_j delta_{j,j0} A[i,j] = A[i,j0]
# sum_j' V[j'*5+k,c] B[j',k] = sum_j' delta_{j',j0} delta_{k,k0} B[j',k] = delta_{k,k0} B[j0,k]
# W[k*2+i,c] = 1
# 
# So C[i,k] = sum_{j0,k0} 1 * A[i,j0] * delta_{k,k0} B[j0,k]
# = sum_{j0} A[i,j0] * B[j0,k]  (since delta_{k,k0} picks out k0=k)
# = (AB)[i,k]. 
# 
# So rank 15 works! That's a big improvement over 30.
# 
# Can we do better? Let's see if we can reduce below 15.
# 
# The constraint is: for each j0, we need to "isolate" the j0-th column of A and j0-th row of B.
# The minimum rank to do this for 3 values of j is at least 3 (one per j).
# For each j, the 2x5 outer product needs some rank.
# 
# For a single j0, the contribution is A[i,j0] * B[j0,k] for all i,k.
# We need terms c with U[i*3+j0,c] and V[j0*5+k,c] and W[k*2+i,c] such that
# sum_c U[i*3+j0,c] V[j0*5+k,c] W[k*2+i,c] = 1 for all i,k (for the diagonal j=j'=j0)
# and sum_c U[i*3+j,c] V[j'*5+k,c] W[k*2+i,c] = 0 for j != j0.
# 
# For the "diagonal" part (j=j'=j0), we need the 2x5 all-ones matrix to be decomposed.
# The 2x5 all-ones matrix has rank 1 as a matrix, but in our tensor setting, the "rank" 
# depends on how we can share the W factor.
# 
# For the 2x5 case: we need sum_c u_c[i] v_c[k] w_c[i,k] = 1 for all i in {0,1}, k in {0..4}.
# If we set w_c[i,k] = 1 for all i,k, then we need sum_c u_c[i] v_c[k] = 1 for all i,k.
# This is the 2x5 all-ones matrix as a sum of rank-1 matrices. Rank of 2x5 all-ones is 1
# (just u=[1,1], v=[1,1,1,1,1]). So 1 term suffices for the diagonal part!
# 
# But we also need the off-diagonal parts to be zero. If we use the same term for all j0,
# it won't work because the U and V factors would need to be delta functions in j.
# 
# Let me try rank = 3 (one per j0):
# c = j0:
# U[i*3+j, c] = delta_{j,j0} * 1
# V[j'*5+k, c] = delta_{j',j0} * 1
# W[k*2+i, c] = 1
# 
# Check diagonal (j=j'=j0): sum_c delta_{j,j0} delta_{j',j0} * 1 = 1 (when j=j'=j0). 
# Check off-diagonal (j != j'): sum_c delta_{j,j0} delta_{j',j0} = 0 (no j0 equals both j and j'). 
# 
# But wait, this gives sum_c U V W = delta_{j,j'} * 1, which is what we need!
# And it doesn't depend on i or k, which is fine since the RHS doesn't either.
# 
# Let me verify the full computation:
# C[i,k] = sum_c W[k*2+i,c] * (sum_j U[i*3+j,c] A[i,j]) * (sum_j' V[j'*5+k,c] B[j',k])
# = sum_{j0=0}^{2} 1 * (sum_j delta_{j,j0} A[i,j]) * (sum_j' delta_{j',j0} B[j',k])
# = sum_{j0=0}^{2} A[i,j0] * B[j0,k]
# = (AB)[i,k]. 
# 
# So rank 3 works! That's amazing.
# 
# But wait, can we do rank 1?
# c=0:
# U[i*3+j, 0] = u_{i,j}, V[j'*5+k, 0] = v_{j',k}, W[k*2+i, 0] = w_{k,i}
# Need: u_{i,j} v_{j',k} w_{k,i} = delta_{j,j'} for all i, j, j', k.
# 
# For j=j': u_{i,j} v_{j,k} w_{k,i} = 1 for all i,k.
# For j!=j': u_{i,j} v_{j',k} w_{k,i} = 0 for all i,k.
# 
# From the j!=j' condition: for any j!=j', u_{i,j} v_{j',k} w_{k,i} = 0 for all i,k.
# This means for each pair (j,j') with j!=j', the "matrix" u_{i,j} v_{j',k} w_{k,i} is zero.
# 
# If w_{k,i} != 0 for some (k,i), then u_{i,j} v_{j',k} = 0 for all i,k when j!=j'.
# This is very restrictive.
# 
# Suppose w_{k,i} = 1 for all i,k. Then u_{i,j} v_{j',k} = 0 for all i,k when j!=j'.
# And u_{i,j} v_{j,k} = 1 for all i,k when j=j'.
# 
# From u_{i,j} v_{j,k} = 1 for all i,k: u_{i,j} = a_j, v_{j,k} = 1/a_j for some nonzero a_j.
# Then u_{i,j} v_{j',k} = a_j / a_{j'} for j!=j', which is not 0. Contradiction.
# 
# So rank 1 doesn't work with w constant. What if w is not constant?
# 
# This is getting complicated. Let me just go with rank 3, which I've verified works.
# Actually, let me double-check with a concrete example.
# 
# n=2, m=3, p=5.
# U shape: (6, 3), V shape: (15, 3), W shape: (10, 3).
# 
# c=0 (j0=0):
# U[i*3+j, 0] = 1 if j==0, else 0. So U[0,0]=1, U[1,0]=0, U[2,0]=0, U[3,0]=1, U[4,0]=0, U[5,0]=0.
# Wait, i*3+j: i=0,j=0 -> 0; i=0,j=1 -> 1; i=0,j=2 -> 2; i=1,j=0 -> 3; i=1,j=1 -> 4; i=1,j=2 -> 5.
# U[0,0]=1, U[1,0]=0, U[2,0]=0, U[3,0]=1, U[4,0]=0, U[5,0]=0.
# 
# V[j'*5+k, 0] = 1 if j'==0, else 0. j'*5+k: j'=0 -> 0..4; j'=1 -> 5..9; j'=2 -> 10..14.
# V[0,0]=1, V[1,0]=1, V[2,0]=1, V[3,0]=1, V[4,0]=1, V[5,0]=0, ..., V[14,0]=0.
# 
# W[k*2+i, 0] = 1 for all k,i. W shape (10, 3). W[k*2+i, 0] = 1 for all 10 entries.
# 
# c=1 (j0=1):
# U[i*3+j, 1] = 1 if j==1, else 0. U[1,1]=1, U[4,1]=1, others 0.
# V[j'*5+k, 1] = 1 if j'==1, else 0. V[5,1]=1, ..., V[9,1]=1, others 0.
# W[k*2+i, 1] = 1 for all.
# 
# c=2 (j0=2):
# U[i*3+j, 2] = 1 if j==2, else 0. U[2,2]=1, U[5,2]=1, others 0.
# V[j'*5+k, 2] = 1 if j'==2, else 0. V[10,2]=1, ..., V[14,2]=1, others 0.
# W[k*2+i, 2] = 1 for all.
# 
# Now check: sum_c U[i*3+j, c] V[j'*5+k, c] W[k*2+i, c]
# = sum_{j0=0}^{2} [1 if j==j0 else 0] * [1 if j'==j0 else 0] * 1
# = 1 if j==j' (there's exactly one j0 that equals both), else 0.
# = delta_{j,j'}. Correct!
# 
# Great, so rank 3 works. Let me code this up.

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    rank = m  # 3
    U = [[0] * rank for _ in range(n * m)]  # 6 x 3
    V = [[0] * rank for _ in range(m * p)]  # 15 x 3
    W = [[0] * rank for _ in range(p * n)]  # 10 x 3
    
    for j0 in range(m):
        # U[i*m + j, j0] = 1 if j == j0
        for i in range(n):
            U[i * m + j0][j0] = 1
        # V[j'*p + k, j0] = 1 if j' == j0
        for k in range(p):
            V[j0 * p + k][j0] = 1
        # W[k*n + i, j0] = 1 for all k, i
        for k in range(p):
            for i in range(n):
                W[k * n + i][j0] = 1
    
    return {"U": U, "V": V, "W": W}
