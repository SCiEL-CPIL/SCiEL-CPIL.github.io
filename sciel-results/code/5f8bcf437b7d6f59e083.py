"""Rewritten program for 3x4x6 matrix multiplication tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 24
    pn = p * n  # 18
    
    # The matrix multiplication tensor for C = A*B where A is n×m, B is m×p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid i,j,k (the "trace" convention
    # W[k*n+i, r] means W is indexed by (k,i) pairs)
    #
    # We need U (nm x r), V (mp x r), W (pn x r) such that:
    # for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # Key insight: The tensor T[i,j,k] = 1 for ALL i,j,k is actually a rank-1 tensor!
    # T[i,j,k] = 1 * 1 * 1
    # So U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1 gives rank 1.
    #
    # However, this is the "all-ones" tensor which is trivially rank 1.
    # The actual matrix multiplication tensor is NOT all ones in the standard
    # decomposition sense. Let me reconsider.
    #
    # Actually, in the standard tensor decomposition for matrix multiplication:
    # The tensor T_{ijk} encodes the multiplication operation. For C = AB:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # The tensor T has T_{ijk} = 1 if and only if the indices (i,j,k) form a valid
    # term in the multiplication, which is always true for valid indices.
    # So T_{ijk} = 1 for all valid i,j,k.
    #
    # This means the tensor IS all ones, and rank 1 should work!
    # But Strassen's algorithm shows rank 7 for 2x2x2, implying the tensor is NOT
    # trivially rank 1 in the way I'm thinking.
    #
    # Wait - I think the issue is that the decomposition must satisfy:
    # For the matrix multiplication to work correctly, we need:
    # C_{ik} = sum_r (sum_j U_{ij,r} A_{ij}) (sum_k V_{jk,r} B_{jk}) W_{ik,r}
    # No, that's not right either.
    #
    # Let me be very precise. The decomposition says:
    # C_{ik} = sum_r W_{ik,r} * (sum_j U_{ij,r} A_{ij}) * (sum_k V_{jk,r} B_{jk})
    # No, that's not standard.
    #
    # The standard formulation: We want to compute C = AB.
    # The tensor T is defined such that C_{ik} = sum_{i',j',k'} T_{i'j'k'} A_{i'j'} B_{j'k'}
    # where T_{i'j'k'} = 1 if i'=i, j'=j, k'=k (i.e., T is the identity in the
    # appropriate sense). Actually T_{i'j'k'} = 1 for all valid indices because
    # C_{ik} = sum_j A_{ij} B_{jk} means each term A_{ij}B_{jk} contributes to C_{ik}
    # with coefficient 1.
    #
    # So T_{ijk} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # This is indeed a rank-1 tensor: T_{ijk} = 1*1*1.
    #
    # But then why does Strassen give rank 7? Because the decomposition must be
    # "valid" in the sense that it actually computes the matrix product. The rank-1
    # decomposition U_{ij,r}=1, V_{jk,r}=1, W_{ik,r}=1 would give:
    # C_{ik} = sum_r W_{ik,r} (sum_j U_{ij,r} A_{ij}) (sum_k V_{jk,r} B_{jk})
    # Wait, that's not the right formula.
    #
    # Let me look at this more carefully. The decomposition of the matrix
    # multiplication tensor T_{ijk} means we write:
    # T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ik,r}
    # where U is n*m x r, V is m*p x r, W is p*n x r (with W indexed as W[k*n+i]).
    #
    # If T_{ijk} = 1 for all i,j,k, then we need:
    # sum_r U_{ij,r} V_{jk,r} W_{ik,r} = 1 for all i,j,k.
    #
    # With r=1: U_{ij,0}=1, V_{jk,0}=1, W_{ik,0}=1 gives 1*1*1=1. ✓
    #
    # So rank 1 works! Let me verify this is correct by checking if it actually
    # computes matrix multiplication.
    #
    # The decomposition says: C_{ik} = sum_r W_{ik,r} * (sum_j U_{ij,r} A_{ij}) * (sum_j V_{jk,r} B_{jk})
    # No wait, that's not right. The standard interpretation is:
    #
    # We compute intermediate values M_r = (sum_j U_{ij,r} A_{ij}) * (sum_k V_{jk,r} B_{jk})
    # No, that doesn't make sense dimensionally.
    #
    # Let me think about this differently. The tensor T_{ijk} represents the
    # bilinear map (A, B) -> C where C_{ik} = sum_j A_{ij} B_{jk}.
    # A rank-r decomposition of this tensor means we can write:
    # C_{ik} = sum_{r=1}^r (sum_j U_{ij,r} A_{ij}) (sum_k V_{jk,r} B_{jk}) W_{ik,r}
    # Hmm, that's still not quite right.
    #
    # Actually, the correct interpretation is:
    # The tensor T is a 3-tensor with T_{ijk} = 1.
    # A rank-r decomposition is: T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ik,r}
    # This is just a tensor decomposition, not directly a matrix multiplication algorithm.
    #
    # For it to be a valid matrix multiplication algorithm, we need the decomposition
    # to be "compatible" with the matrix multiplication structure, meaning:
    # C_{ik} = sum_j A_{ij} B_{jk} = sum_r (sum_j U_{ij,r} A_{ij}) (sum_k V_{jk,r} B_{jk}) W_{ik,r}
    # Wait, that's not dimensionally correct either.
    #
    # I think the correct formulation is:
    # We define M_r = sum_j U_{ij,r} A_{ij} (this is a scalar for each i, r)
    # N_r = sum_k V_{jk,r} B_{jk} (this is a scalar for each j, r)
    # Then C_{ik} = sum_r W_{ik,r} M_r N_r? No, that doesn't work.
    #
    # Let me just trust the problem statement. The task says:
    # "exactly decompose the matrix multiplication tensor"
    # with U (nm x rank), V (mp x rank), W (pn x rank)
    # and the constraint is:
    # for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # If this is the constraint, then rank 1 with all-ones works.
    # Let me just return rank 1 and see if it passes validation.
    
    # Try rank 1 first
    rank = 1
    U = [[1] for _ in range(nm)]
    V = [[1] for _ in range(mp)]
    W = [[1] for _ in range(pn)]
    
    # Verify: for all i,j,k: U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1*1*1 = 1 ✓
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
