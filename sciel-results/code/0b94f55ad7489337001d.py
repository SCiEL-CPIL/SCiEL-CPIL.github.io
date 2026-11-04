"""Rank-minimal tensor decomposition for multiplying a 2x3 matrix by a 3x9 matrix."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We'll try to find a rank-minimal decomposition.
    # The schoolbook rank is n*m*p = 54.
    # However, for 2x3 by 3x9, we can potentially do better.
    # Let's think about the structure.
    # A is 2x3, B is 3x9. Output is 2x9.
    # The tensor is T_{i,k,j} where i in [0,2), k in [0,9), j in [0,3).
    # T_{i,k,j} = 1 if the multiplication C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So C_{i,k} depends on A_{i,j} and B_{j,k}.
    # In the tensor decomposition, we have:
    # C_{i,k} = sum_c U_{i,j} V_{j,k} ... wait, no.
    # The standard form is:
    # C_{i,k} = sum_j sum_c U_{i,j,c} V_{j,k,c} ... no.
    # Let's use the notation from the prompt.
    # U has shape (n*m, rank) = (6, rank)
    # V has shape (m*p, rank) = (27, rank)
    # W has shape (p*n, rank) = (18, rank)
    # The decomposition is:
    # C_{i,k} = sum_c W_{k*n+i, c} * U_{i*m+j, c} * V_{j*p+k, c}
    # Wait, this is a rank-1 tensor term.
    # A rank-1 term is U_{i,j} V_{j,k} W_{k,i} where the indices are fixed.
    # Actually, the tensor T_{i,k,j} is decomposed as sum_c U_{i,j}^{(c)} V_{j,k}^{(c)} W_{k,i}^{(c)}.
    # Here U is indexed by (i,j), V by (j,k), W by (k,i).
    # U has shape (n*m, rank), so U[(i*m+j), c] is the coefficient for A_{i,j} in the c-th term.
    # V[(j*p+k), c] is the coefficient for B_{j,k} in the c-th term.
    # W[(k*n+i), c] is the coefficient for C_{k,i} in the c-th term. Wait, C is indexed by (i,k).
    # The prompt says W has shape (p*n, rank) and W[k*n+i, r]. So W is indexed by (k,i).
    # And C_{i,k} = sum_c W_{k*n+i, c} * ... wait.
    # Let's check the trace convention.
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # So W[k*n+i, c] is the coefficient for the output element C_{i,k} in the c-th term.
    # The c-th term contributes W_{k*n+i, c} * U_{i*m+j, c} * V_{j*p+k, c} to C_{i,k}?
    # No, that's not right. In a rank-1 tensor decomposition of the matrix multiplication tensor:
    # T_{i,k,j} = sum_c U_{i,j}^{(c)} V_{j,k}^{(c)} W_{k,i}^{(c)}
    # Here T_{i,k,j} = delta_{i,k,j} which is 1 if the multiplication involves A_{i,j} and B_{j,k} to produce C_{i,k}.
    # Actually, the matrix multiplication tensor is defined such that:
    # C_{i,k} = sum_j A_{i,j} B_{j,k} = sum_j sum_c U_{i,j}^{(c)} V_{j,k}^{(c)} W_{i,k}^{(c)}
    # Wait, the standard decomposition is:
    # C_{i,k} = sum_c (sum_j U_{i,j}^{(c)} A_{i,j}) (sum_k V_{j,k}^{(c)} B_{j,k}) ... no.
    # Let's look at the shapes again.
    # U: (n*m, rank) -> U[i*m+j, c] multiplies A[i,j]
    # V: (m*p, rank) -> V[j*p+k, c] multiplies B[j,k]
    # W: (p*n, rank) -> W[k*n+i, c] multiplies C[i,k] (since W is indexed by k*n+i)
    # The decomposition is:
    # C_{i,k} = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_k' V[j*p+k', c] B[j,k'])
    # This doesn't look right. Let's think about it differently.
    # A rank-1 decomposition of the multiplication tensor T is:
    # T_{i,k,j} = sum_c U_{i,j}^{(c)} V_{j,k}^{(c)} W_{k,i}^{(c)}
    # where T_{i,k,j} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}, and 0 otherwise.
    # Actually, T_{i,k,j} is just the identity tensor for the indices:
    # T_{i,k,j} = 1 for all i,j,k in the schoolbook case.
    # No, the multiplication tensor has entries T_{i,k,j} = 1 if the (i,k)-th entry of C depends on A_{i,j} and B_{j,k}.
    # Since C_{i,k} = sum_j A_{i,j} B_{j,k}, the tensor is:
    # T_{i,k,j} = 1 for all valid i, j, k.
    # And the decomposition is:
    # T_{i,k,j} = sum_c U_{i,j}^{(c)} V_{j,k}^{(c)} W_{i,k}^{(c)}
    # Wait, the indices in W are (k, i) according to the prompt: W[k*n+i, r].
    # So W_{k,i}^{(c)} = W[k*n+i, c].
    # And the decomposition is:
    # T_{i,k,j} = sum_c U_{i,j}^{(c)} V_{j,k}^{(c)} W_{k,i}^{(c)}
    # In the schoolbook decomposition, for each (i,j,k), we have one term:
    # U_{i,j}^{(c)} = 1, V_{j,k}^{(c)} = 1, W_{k,i}^{(c)} = 1, and all other entries in that term are 0.
    # This gives rank = n*m*p = 54.
    
    # Can we do better?
    # For 2x3 by 3x9, the rank is at least the max of the three dimensions' products?
    # Actually, the rank of M_{2,3,9} is at least the rank of M_{2,3,3} times 3? No.
    # The rank is at least the max of (n*m, m*p, p*n) = max(6, 27, 18) = 27.
    # So the rank is at least 27.
    # Can we achieve rank 27?
    # Let's try to construct a rank-27 decomposition.
    # We need to decompose T_{i,k,j} for i in [0,2), k in [0,9), j in [0,3).
    # T_{i,k,j} = 1 for all i, j, k.
    # We want T_{i,k,j} = sum_{c=0}^{26} U_{i,j}^{(c)} V_{j,k}^{(c)} W_{k,i}^{(c)}.
    # This is a system of 2*9*3 = 54 equations.
    # We have 27 * (6 + 27 + 18) = 27 * 51 = 1377 variables.
    # So there should be a solution.
    # Let's try to find it.
    # We can set U_{i,j}^{(c)} = delta_{j, c%3} for c in [0,27).
    # Then V_{j,k}^{(c)} = delta_{k, (c//3)%9} for c in [0,27).
    # Then W_{k,i}^{(c)} = ?
    # Let's see. T_{i,k,j} = sum_c delta_{j, c%3} delta_{k, (c//3)%9} W_{k,i}^{(c)}.
    # For a fixed (i,k,j), the sum is over c such that c%3 = j and (c//3)%9 = k.
    # c = 3*j + 9*(k//1) ... wait, c is in [0,27).
    # c%3 = j means c = 3*j + 3*t for t in [0,9).
    # (c//3)%9 = k means (j + t)%9 = k, so t = (k - j) mod 9.
    # So for each (i,k,j), there is exactly one c: c = 3*j + 3*((k - j) mod 9).
    # Then T_{i,k,j} = W_{k,i}^{(c)}.
    # We need W_{k,i}^{(c)} = 1 for all i, k, j.
    # So W_{k,i}^{(c)} = 1 for all k, i, and the specific c.
    # But c depends on j. So for a fixed (k,i), W_{k,i}^{(c)} must be 1 for all c that can appear.
    # c = 3*j + 3*((k - j) mod 9). As j varies from 0 to 2, c varies.
    # So W_{k,i}^{(c)} must be 1 for all c in [0,27).
    # Then T_{i,k,j} = sum_c delta_{j, c%3} delta_{k, (c//3)%9} * 1 = 1.
    # This works!
    # So the decomposition is:
    # U[i*m+j, c] = 1 if c%3 == j else 0
    # V[j*p+k, c] = 1 if (c//3)%9 == k else 0
    # W[k*n+i, c] = 1 for all c
    # Let's verify.
    # For a fixed (i,j,k), the term in the sum is U[i*3+j, c] * V[j*9+k, c] * W[k*2+i, c].
    # U[i*3+j, c] = 1 iff c%3 == j.
    # V[j*9+k, c] = 1 iff (c//3)%9 == k.
    # W[k*2+i, c] = 1.
    # So the product is 1 iff c%3 == j and (c//3)%9 == k.
    # As shown before, there is exactly one such c in [0,27).
    # So the sum is 1.
    # This gives a rank-27 decomposition!
    
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(rank):
                    if c % m == j and (c // m) % p == k:
                        U[i * m + j][c] = 1
                        V[j * p + k][c] = 1
                        W[k * n + i][c] = 1
                    else:
                        W[k * n + i][c] = 1  # Wait, W should be 1 for all c?
    
    # Let's re-examine.
    # We need T_{i,k,j} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    # If W[k*n+i, c] = 1 for all c, then the sum is sum_c U[i*m+j, c] * V[j*p+k, c].
    # U[i*m+j, c] = 1 iff c%m == j.
    # V[j*p+k, c] = 1 iff (c//m)%p == k.
    # So the sum is the number of c in [0,rank) such that c%m == j and (c//m)%p == k.
    # If rank = m*p, then c = m*j + p*k is the unique solution? No.
    # c%m == j means c = m*j + m*t.
    # (c//m)%p == k means (j + t)%p == k, so t = (k - j) mod p.
    # So c = m*j + m*((k - j) mod p).
    # This is in [0, m*p) if rank = m*p.
    # So the sum is 1.
    # So W[k*n+i, c] should be 1 for all c, and the sum is 1.
    # Let's set W[k*n+i, c] = 1 for all c.
    
    for i in range(n):
        for j in range(m):
            for c in range(rank):
                if c % m == j:
                    U[i * m + j][c] = 1
    for j in range(m):
        for k in range(p):
            for c in range(rank):
                if (c // m) % p == k:
                    V[j * p + k][c] = 1
    for k in range(p):
        for i in range(n):
            for c in range(rank):
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
