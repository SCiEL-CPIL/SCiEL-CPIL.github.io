"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Build the matrix multiplication tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Convention: W[k*n + i, r] contributes to C[i,k]
    # So T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # First, build the rank-42 Strassen decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        """
        a_idx: [a00, a01, a10, a11] indices into U
        b_idx: [b00, b01, b10, b11] indices into V
        w_idx: [w00, w01, w10, w11] indices into W for the 2x2 output block
        """
        # Strassen:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        c = start_col
        # M1
        U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
        V[b_idx[0]][c] += 1; V[b_idx[3]][c] += 1
        W[w_idx[0]][c] += 1; W[w_idx[3]][c] += 1
        # M2
        U[a_idx[2]][c+1] += 1; U[a_idx[3]][c+1] += 1
        V[b_idx[0]][c+1] += 1
        W[w_idx[2]][c+1] += 1; W[w_idx[3]][c+1] -= 1
        # M3
        U[a_idx[0]][c+2] += 1
        V[b_idx[1]][c+2] += 1; V[b_idx[3]][c+2] -= 1
        W[w_idx[1]][c+2] += 1; W[w_idx[3]][c+2] += 1
        # M4
        U[a_idx[3]][c+3] += 1
        V[b_idx[2]][c+3] += 1; V[b_idx[0]][c+3] -= 1
        W[w_idx[0]][c+3] += 1; W[w_idx[2]][c+3] += 1
        # M5
        U[a_idx[0]][c+4] += 1; U[a_idx[1]][c+4] += 1
        V[b_idx[3]][c+4] += 1
        W[w_idx[0]][c+4] -= 1; W[w_idx[1]][c+4] += 1
        # M6
        U[a_idx[2]][c+5] += 1; U[a_idx[0]][c+5] -= 1
        V[b_idx[0]][c+5] += 1; V[b_idx[1]][c+5] += 1
        W[w_idx[3]][c+5] += 1
        # M7
        U[a_idx[1]][c+6] += 1; U[a_idx[3]][c+6] -= 1
        V[b_idx[2]][c+6] += 1; V[b_idx[3]][c+6] += 1
        W[w_idx[0]][c+6] += 1
    
    # A is 2x4: indices i*m+j
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    a_L = [0, 1, 4, 5]  # a00, a01, a10, a11
    a_R = [2, 3, 6, 7]
    
    # B is 4x6: indices j*p+k
    # For each pair of rows (r1,r2) and pair of cols (c1,c2):
    # b00=B[r1,c1], b01=B[r1,c2], b10=B[r2,c1], b11=B[r2,c2]
    
    # Block structure: split A into left 2 cols and right 2 cols
    # Split B into 3 blocks of 2 cols each
    # For each A block and B col-block, we have a 2x2 * 2x2 multiplication
    
    # B col block 0: cols 0,1
    #   Rows 0,1: b00=B[0,0]=0, b01=B[0,1]=1, b10=B[1,0]=6, b11=B[1,1]=7
    #   Rows 2,3: b00=B[2,0]=12, b01=B[2,1]=13, b10=B[3,0]=18, b11=B[3,1]=19
    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]
    
    # W indices for C cols 0,1: C[i,k] -> W[k*n+i]
    # C[0,0]->W[0], C[0,1]->W[2], C[1,0]->W[1], C[1,1]->W[3]
    w_1 = [0, 2, 1, 3]
    
    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)
    
    # B col block 1: cols 2,3
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]
    
    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)
    
    # B col block 2: cols 4,5
    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    w_3 = [8, 10, 9, 11]
    
    add_strassen_block(a_L, b_3_T, w_3, 28)
    add_strassen_block(a_R, b_3_B, w_3, 35)
    
    # Verify the decomposition
    def verify_decomp(U, V, W, rank):
        """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = delta for M_{n,m,p}"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    if not verify_decomp(U, V, W, rank):
        # Fallback: return schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    # Now try to reduce rank using the peeling algorithm
    # The idea: if there's a rank-1 term u ⊗ v ⊗ w in the decomposition,
    # then for any vector a orthogonal to u, the contraction <a, T> should have
    # rank at most rank-1. We can try to find such vectors.
    
    # Convert to numpy arrays for numerical work
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Try to find linear dependencies among the rank-1 terms
    # If sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0 for some nonzero c,
    # then we can eliminate one term.
    
    # Build the matrix of all possible rank-1 tensors
    # Each rank-1 term u_r ⊗ v_r ⊗ w_r is a vector in R^(n*m*m*p*p*n)
    # but that's too large. Instead, we work with the matrix flattening.
    
    # For a fixed r, the tensor u_r ⊗ v_r ⊗ w_r can be seen as:
    # - Matrix U_r (n*m x 1) * V_r (m*p x 1)^T * W_r (p*n x 1)
    # This is a 3-tensor. We can flatten it to a vector of size (n*m)*(m*p)*(p*n)
    
    # But that's 8*24*12 = 2304 dimensions, which is manageable.
    
    # Actually, let's use a smarter approach. We know the tensor is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's check if we can find a non-trivial linear combination that gives zero.
    # This is equivalent to checking if the vectors {u_r ⊗ v_r ⊗ w_r} are linearly independent.
    
    # Flatten each rank-1 term to a vector
    def flatten_term(U_col, V_col, W_col):
        """Flatten u ⊗ v ⊗ w to a vector of size (n*m)*(m*p)*(p*n)"""
        term = np.zeros((n*m, m*p, p*n))
        for idx_u in range(n*m):
            for idx_v in range(m*p):
                for idx_w in range(p*n):
                    term[idx_u, idx_v, idx_w] = U_col[idx_u] * V_col[idx_v] * W_col[idx_w]
        return term.flatten()
    
    # Build the matrix of all rank-1 terms
    terms = np.zeros((rank, (n*m)*(m*p)*(p*n)))
    for r in range(rank):
        terms[r] = flatten_term(U_np[:, r], V_np[:, r], W_np[:, r])
    
    # Check the rank of this matrix
    # If rank(terms) < rank, then there's a linear dependency
    tol = 1e-10
    try:
        matrix_rank = np.linalg.matrix_rank(terms, tol=tol)
    except:
        matrix_rank = rank
    
    if matrix_rank < rank:
        # There's a linear dependency. Find the null space vector.
        # terms @ c = 0 for some nonzero c
        _, S, Vt = np.linalg.svd(terms.T)
        # The last row of Vt gives a null space vector
        c = Vt[-1]
        
        # Find the index with the largest |c_r|
        max_idx = np.argmax(np.abs(c))
        
        # We can eliminate term max_idx by rewriting the others
        # T = sum_{r != max_idx} (c_r / c_max_idx) * (u_r ⊗ v_r ⊗ w_r) + c_max_idx * (u_max ⊗ v_max ⊗ w_max)
        # But this doesn't directly give a lower rank decomposition.
        
        # Instead, we note that if there's a dependency, we might be able to
        # find a different decomposition with lower rank.
        
        # For now, let's just check if we can remove the term with max_idx
        # and see if the remaining terms still span the tensor.
        
        # This is complex. Let's try a different approach:
        # Try to find if any single term can be expressed as a linear combination
        # of the others in a way that allows elimination.
        
        pass
    
    # Alternative approach: try to find a rank reduction by looking for
    # specific algebraic structures.
    
    # For the 2x4x6 case, let's try to see if we can merge some Strassen blocks.
    # The key insight is that when multiplying (2x2)*(2x2) blocks, some of the
    # intermediate products might be shared or cancellable.
    
    # Let's try a direct search for a lower-rank decomposition.
    # We'll use a randomized approach: perturb the decomposition and see if we
    # can find a lower-rank one.
    
    # Actually, let's try a more systematic approach.
    # The tensor M_{2,4,6} has a known structure. Let's try to exploit it.
    
    # One approach: use the fact that n=2 is small. We can write:
    # C[0,k] = sum_j A[0,j] * B[j,k]
    # C[1,k] = sum_j A[1,j] * B[j,k]
    
    # This is equivalent to:
    # C = A * B where A is 2x4 and B is 4x6.
    
    # We can think of this as: for each column k of B, we compute
    # C[:,k] = A * B[:,k]
    
    # The multiplication of a 2x4 matrix by a 4x1 vector can be done
    # with rank at most 8 (schoolbook), but maybe less?
    
    # Actually, the tensor for multiplying 2x4 by 4x1 is M_{2,4,1},
    # which has rank at most 8.
    
    # But we need to do this for 6 different columns, and the results
    # go into different positions in C.
    
    # Let's try a different decomposition strategy.
    # Instead of using Strassen on 2x2 blocks, let's try to use
    # a decomposition that exploits the full structure.
    
    # For now, let's return the rank-42 decomposition if we can't find
    # a better one quickly.
    
    # Try one more thing: check if we can reduce rank by 1 by finding
    # a specific linear dependency.
    
    # Let's try to solve for a rank-41 decomposition by setting up
    # a system of equations.
    
    # This is computationally expensive, so let's only do it if we have time.
    elapsed = time.time() - start_time
    if elapsed < 100:  # If we have more than 45 seconds left
        # Try to find a rank-41 decomposition
        # We'll use a heuristic: try to eliminate one of the Strassen terms
        # by expressing it as a combination of others.
        
        # For each term r, check if it can be expressed as a linear combination
        # of the other terms.
        for r_elim in range(rank):
            if time.time() - start_time > deadline - 5:
                break
            
            # Build the matrix of all terms except r_elim
            mask = [i for i in range(rank) if i != r_elim]
            terms_reduced = terms[mask]
            
            # Check if term r_elim is in the span of terms_reduced
            # This is equivalent to checking if the rank of [terms_reduced; terms[r_elim]]
            # equals the rank of terms_reduced.
            
            # Use least squares to check
            try:
                # Solve terms_reduced @ x = terms[r_elim]
                x, residuals, rank_r, s = np.linalg.lstsq(terms_reduced.T, terms[r_elim], rcond=tol)
                
                # Check if the residual is small
                residual_norm = np.linalg.norm(terms[r_elim] - terms_reduced.T @ x)
                term_norm = np.linalg.norm(terms[r_elim])
                
                if term_norm > 0 and residual_norm / term_norm < 1e-8:
                    # Term r_elim is in the span of the others!
                    # We can eliminate it.
                    new_rank = rank - 1
                    new_U = np.delete(U_np, r_elim, axis=1)
                    new_V = np.delete(V_np, r_elim, axis=1)
                    new_W = np.delete(W_np, r_elim, axis=1)
                    
                    # Verify the new decomposition
                    if verify_decomp(new_U.tolist(), new_V.tolist(), new_W.tolist(), new_rank):
                        return {"U": new_U.tolist(), "V": new_V.tolist(), "W": new_W.tolist()}
            except:
                pass
    
    # Return the rank-42 decomposition
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
