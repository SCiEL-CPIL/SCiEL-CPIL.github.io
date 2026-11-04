"""Program for matrix multiplication tensor decomposition (4x5 * 5x6)."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for multiplying a 4x5 matrix by a 5x6 matrix.
    Uses a recursive splitting strategy with Strassen-like decomposition for 2x2 blocks.
    Target rank: 90.
    """
    n, m, p = DIMENSIONS
    
    # We split p=6 into 3 blocks of size 2.
    # For each block, we compute A @ B_block (4x5 @ 5x2 = 4x2).
    # We split A into two 2x5 blocks.
    # Each 2x5 @ 5x2 is computed using 15 terms (7+7+1 from Strassen on 2x2x2 sub-problems).
    # Total rank = 3 * 2 * 15 = 90.
    
    rank = 90
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen decomposition for 2x2x2
    # Inputs: i_offset (row offset in A), j_offset (col offset in B? No, B is 5x2, j is 0..4), k_offset (col offset in C)
    # Actually, we are decomposing 2x5x2.
    # We split the 5 middle dimension into 2+2+1.
    # Sub-problem 1: 2x2x2 (indices 0,1 of middle)
    # Sub-problem 2: 2x2x2 (indices 2,3 of middle)
    # Sub-problem 3: 2x1x2 (index 4 of middle)
    
    def add_strassen_2x2x2(base_rank, i_off, j_off, k_off):
        """
        Adds a 2x2x2 Strassen decomposition.
        i_off: offset for i in U (0..1)
        j_off: offset for j in V (0..1)
        k_off: offset for k in W (0..1)
        Note: U index is i*m + j. Here m=5.
        V index is j*p + k. Here p=6.
        W index is k*n + i. Here n=4.
        """
        # Strassen coefficients for 2x2x2
        # U, V, W are 2x2 matrices in the local sense.
        # Local U[i, j], V[j, k], W[k, i]
        # Global U[i_off + i, j_off + j]
        # Global V[j_off + j, k_off + k]
        # Global W[k_off + k, i_off + i]
        
        # 7 terms
        # P1: (U00+U11)(V00+V11)
        # P2: (U10+U11)V00
        # P3: U00(V01-V11)
        # P4: U11(V10-V00)
        # P5: (U00+U01)V11
        # P6: (U10-U00)(V00+V01)
        # P7: (U01-U11)(V10+V11)
        
        terms = [
            # (u_i, u_j, v_j, v_k, w_k, w_i) -> coefficients
            # We need to map local indices to global.
            # Local U[i,j] corresponds to Global U[(i_off+i)*m + (j_off+j)]
            # Local V[j,k] corresponds to Global V[(j_off+j)*p + (k_off+k)]
            # Local W[k,i] corresponds to Global W[(k_off+k)*n + (i_off+i)]
            
            # Term 1
            (0, 0, 0, 0, 0, 0, 1), (0, 0, 1, 1, 0, 0, 1),
            (1, 1, 0, 0, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1),
            # Term 2
            (1, 0, 0, 0, 0, 1, 1), (1, 1, 0, 0, 0, 1, 1),
            (0, 0, 1, 0, 0, 0, 1),
            # Term 3
            (0, 0, 0, 1, 0, 0, 1),
            (0, 0, 1, 1, 0, 0, -1),
            # Term 4
            (1, 1, 1, 0, 1, 1, 1),
            (1, 1, 0, 0, 1, 1, -1),
            # Term 5
            (0, 0, 1, 1, 1, 0, 1), (0, 1, 1, 1, 1, 0, 1),
            # Term 6
            (1, 0, 0, 0, 1, 1, 1), (0, 0, 0, 0, 1, 1, -1),
            (1, 0, 0, 1, 1, 1, 1), (0, 0, 0, 1, 1, 1, -1),
            # Term 7
            (0, 1, 1, 0, 1, 0, 1), (1, 1, 1, 0, 1, 0, -1),
            (0, 1, 1, 1, 1, 0, 1), (1, 1, 1, 1, 1, 0, -1)
        ]
        
        for t in terms:
            u_i, u_j, v_j, v_k, w_k, w_i, coeff = t
            u_idx = (i_off + u_i) * m + (j_off + u_j)
            v_idx = (j_off + v_j) * p + (k_off + v_k)
            w_idx = (k_off + w_k) * n + (i_off + w_i)
            
            U[u_idx][base_rank] += coeff
            V[v_idx][base_rank] += coeff
            W[w_idx][base_rank] += coeff
            base_rank += 1
            
        return base_rank

    def add_rank1_2x1x2(base_rank, i_off, j_idx, k_off):
        """
        Adds a 2x1x2 rank 1 decomposition.
        j_idx is the single middle index.
        """
        # T[i, j, k] = U[i, j] V[j, k] W[k, i]
        # We want sum = A_i B_k
        # Set U[i, j] = 1, V[j, k] = 1, W[k, i] = 1?
        # Then sum = 1*1*1 = 1.
        # But we need A_i B_k.
        # The decomposition is linear in A and B.
        # The coefficients in U, V, W are fixed.
        # The values A_i, B_k are the inputs.
        # So U[i, j, c] should be 1, V[j, k, c] should be 1, W[k, i, c] should be 1.
        # Then the term is 1 * A_i * 1 * B_k * 1 = A_i B_k.
        
        u_idx = (i_off + 0) * m + j_idx
        v_idx = j_idx * p + (k_off + 0)
        w_idx = (k_off + 0) * n + (i_off + 0)
        
        U[u_idx][base_rank] = 1
        V[v_idx][base_rank] = 1
        W[w_idx][base_rank] = 1
        
        u_idx = (i_off + 1) * m + j_idx
        v_idx = j_idx * p + (k_off + 1)
        w_idx = (k_off + 1) * n + (i_off + 1)
        
        # Wait, rank 1 means one column in U, V, W.
        # The term is U[i,j,c] V[j,k,c] W[k,i,c].
        # If we set U[i,j,c]=1 for all i, V[j,k,c]=1 for all k, W[k,i,c]=1 for all k,i?
        # Then sum = 1*1*1 = 1.
        # This gives 1, not A_i B_k.
        # The inputs A and B are multiplied by U and V.
        # So U[i,j,c] multiplies A[i,j].
        # So if U[i,j,c]=1, V[j,k,c]=1, W[k,i,c]=1, the contribution is A[i,j] B[j,k] * 1.
        # This is exactly the term we want.
        
        # So for 2x1x2, we need 2 terms?
        # T[i,0,k] = A[i,0] B[0,k].
        # We can use one term: U[i,0,c]=1, V[0,k,c]=1, W[k,i,c]=1.
        # Then sum_c ... = A[i,0] B[0,k].
        # So rank 1 is sufficient.
        
        # But we have 4 entries in W? No, W is indexed by k,i.
        # The term is a single scalar product.
        
        # Let's re-verify.
        # c is the rank index.
        # U[i,0,c] = 1
        # V[0,k,c] = 1
        # W[k,i,c] = 1
        # Product: 1 * A[i,0] * 1 * B[0,k] * 1 = A[i,0]B[0,k].
        # This works for all i,k simultaneously with one c.
        
        # So we just need to set these values.
        for i in range(2):
            for k in range(2):
                u_idx = (i_off + i) * m + j_idx
                v_idx = j_idx * p + (k_off + k)
                w_idx = (k_off + k) * n + (i_off + i)
                
                U[u_idx][base_rank] = 1
                V[v_idx][base_rank] = 1
                W[w_idx][base_rank] = 1
                
        return base_rank + 1

    current_rank = 0
    # 3 blocks for k
    for b in range(3):
        k_off = b * 2
        # 2 blocks for i (top/bot)
        for a in range(2):
            i_off = a * 2
            # Sub-problem 1: j=0,1
            current_rank = add_strassen_2x2x2(current_rank, i_off, 0, k_off)
            # Sub-problem 2: j=2,3
            current_rank = add_strassen_2x2x2(current_rank, i_off, 2, k_off)
            # Sub-problem 3: j=4
            current_rank = add_rank1_2x1x2(current_rank, i_off, 4, k_off)
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
