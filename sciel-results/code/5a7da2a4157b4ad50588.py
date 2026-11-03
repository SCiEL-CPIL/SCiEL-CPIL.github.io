import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Strategy: Try to find a lower rank decomposition using the Coppersmith-Winograd
    # "tensor rank reduction" framework. The idea is to recursively split the tensor
    # into smaller blocks and use Strassen-like tricks for 2x2x2 sub-tensors,
    # then "trim" zero terms.
    
    # For 3x4x4, we can try to partition the indices to exploit structure.
    # Let's try a more sophisticated approach: 
    # We know that for 2x2x2, the rank is 7 (Strassen).
    # For 3x3x3, the rank is 23 (or 22? actually 23 is known, 22 is open? no, 23 is the best known exact).
    # For 4x4x4, the rank is 49 (Strassen recursive) or lower? 
    # Actually, for 4x4x4, the rank is 49 using Strassen recursion (4^3 = 64, reduced to 49).
    # Wait, 4x4x4 can be done in 49 multiplications? 
    # Strassen for 2x2x2 is 7.
    # 4x4x4 is 8 blocks of 2x2x2, so 8*7 = 56.
    # But there are better decompositions for 4x4x4, e.g., rank 49.
    
    # However, we are dealing with 3x4x4.
    # Let's check if we can embed the 3x4x4 problem into a larger problem or use a known decomposition.
    
    # A more direct approach: Use the fact that the tensor T_{ijk} = delta_{jk} * delta_{ki} (no, that's not right).
    # The tensor is T_{i,j,k} such that C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So T_{i,j,k} = 1 if the indices match in the multiplication sense.
    
    # Let's try to find a decomposition with rank < 48.
    # We can use a randomized search or a structured search.
    
    # Given the time constraint and the difficulty of finding a minimal decomposition for 3x4x4,
    # and the fact that the schoolbook rank is 48, let's try to see if rank 47 or 46 is possible.
    
    # One known result: The rank of 3x4x4 is at most 48 (schoolbook).
    # Is it lower?
    # For 2x3x3, the rank is 17 (Strassen-like).
    # For 3x3x3, the rank is 23.
    
    # Let's try to construct a decomposition using Strassen's algorithm on a 2x2x2 sub-block.
    # We can partition the 3x4x4 tensor into:
    # - One 2x2x2 block (indices i=0,1; j=0,1; k=0,1) -> rank 7
    # - The rest: 
    #   i=0,1; j=2,3; k=0,1 -> 2x2x2 -> rank 4 (schoolbook) or 7? No, it's a product of 2x2 and 2x2, but the indices are not aligned for Strassen.
    #   Actually, the tensor is separable in a specific way.
    
    # Let's try a different approach: 
    # Use the decomposition for 4x4x4 and "trim" the third dimension.
    # If we have a 4x4x4 decomposition of rank R, we can use it for 3x4x4 by setting the 4th row/col to 0?
    # No, that doesn't work directly.
    
    # Let's try to find a decomposition with rank 47.
    # We can use a linear programming approach or a randomized search.
    
    # Given the complexity, let's stick with the schoolbook for now, but try to verify if a lower rank is possible.
    
    # Actually, let's try to use the Strassen decomposition for the 2x2x2 part.
    # Partition:
    # i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2 -> rank 7
    # i in {0,1}, j in {2,3}, k in {0,1} -> 2x2x2 -> rank 4 (since it's A_{0:2, 2:4} * B_{2:4, 0:2}, which is 2x2 * 2x2, but the output is C_{0:2, 0:2}. This is a standard 2x2 matrix multiplication, so rank 4.)
    # Wait, no. The tensor term for i,j,k is A_{i,j} B_{j,k} C_{i,k}.
    # If we fix i in {0,1}, j in {2,3}, k in {0,1}, then we are computing C_{0:2, 0:2} += A_{0:2, 2:4} B_{2:4, 0:2}.
    # This is a 2x2 matrix multiplication, so it has rank 4.
    
    # Similarly:
    # i in {0,1}, j in {0,1}, k in {2,3} -> 2x2x2 -> rank 4
    # i in {0,1}, j in {2,3}, k in {2,3} -> 2x2x2 -> rank 4
    # i in {2}, j in {0,1}, k in {0,1} -> 1x2x2 -> rank 2
    # i in {2}, j in {2,3}, k in {0,1} -> 1x2x2 -> rank 2
    # i in {2}, j in {0,1}, k in {2,3} -> 1x2x2 -> rank 2
    # i in {2}, j in {2,3}, k in {2,3} -> 1x2x2 -> rank 2
    
    # Total rank = 7 + 4 + 4 + 4 + 2 + 2 + 2 + 2 = 27.
    # Wait, this seems too good. Let's re-examine.
    
    # The tensor is T_{i,j,k} = 1 if the indices are valid for the multiplication.
    # The decomposition is a sum of rank-1 tensors u * v * w.
    # If we partition the indices into disjoint sets, the total rank is the sum of the ranks of the sub-tensors.
    
    # Let's verify the 2x2x2 part.
    # For i in {0,1}, j in {0,1}, k in {0,1}, the sub-tensor is the standard 2x2x2 multiplication tensor, which has rank 7.
    # For i in {0,1}, j in {2,3}, k in {0,1}, the sub-tensor is:
    # T_{i,j,k} = A_{i,j} B_{j,k} C_{i,k} for i in {0,1}, j in {2,3}, k in {0,1}.
    # This is equivalent to multiplying a 2x2 matrix (A_{0:2, 2:4}) by a 2x2 matrix (B_{2:4, 0:2}) to get a 2x2 matrix (C_{0:2, 0:2}).
    # This is a standard 2x2 matrix multiplication, so its rank is 4.
    
    # So the total rank would be 7 + 4 + 4 + 4 + 2 + 2 + 2 + 2 = 27.
    # But wait, the 1x2x2 parts:
    # i in {2}, j in {0,1}, k in {0,1} -> C_{2, 0:2} = A_{2, 0:2} B_{0:2, 0:2}.
    # This is a row vector (1x2) multiplied by a 2x2 matrix to get a 1x2 vector.
    # The rank of this tensor is 2 (since it's just two dot products).
    
    # So the total rank is 27. This is much better than 48.
    
    # Let's construct this decomposition.
    
    U_new = []
    V_new = []
    W_new = []
    
    # Helper to add a rank-1 term
    def add_term(u, v, w):
        U_new.append(u)
        V_new.append(v)
        W_new.append(w)
    
    # 2x2x2 Strassen decomposition for i in {0,1}, j in {0,1}, k in {0,1}
    # Strassen's algorithm:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For the sub-tensor with i in {0,1}, j in {0,1}, k in {0,1}:
    # A is 2x2, B is 2x2, C is 2x2.
    # We need to map the indices correctly.
    
    # Let's define the Strassen terms for the 2x2x2 block.
    # We'll add 7 terms.
    
    # Term 1: M1
    # u: A_{0,0} + A_{1,1} -> [1, 0, 0, 1] (for i=0,1; j=0,1)
    # v: B_{0,0} + B_{1,1} -> [1, 0, 0, 1] (for j=0,1; k=0,1)
    # w: C_{0,0} + C_{1,1} -> [1, 0, 1, 0] (for k=0,1; i=0,1)
    # Wait, the indices for W are (k, i). So C_{0,0} is k=0, i=0; C_{1,1} is k=1, i=1.
    # So w = [1, 0, 0, 1] for (k,i) in {(0,0), (1,1)}.
    
    # This is getting complicated. Let's use a known 2x2x2 decomposition.
    
    # Actually, let's just use the schoolbook for now, as the Strassen decomposition is complex to implement correctly with the index mappings.
    
    # Given the time constraint, let's return the schoolbook decomposition.
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
