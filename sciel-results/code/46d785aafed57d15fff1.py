DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 19
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        col = add_term.r
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        add_term.r += 1

    add_term.r = 0

    # Indices:
    # A: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
    # B: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
    # W (Output): Index k*n + i.
    # C[0,0]->0, C[0,1]->1, C[0,2]->2
    # C[1,0]->3, C[1,1]->4, C[1,2]->5
    # C[2,0]->6, C[2,1]->7, C[2,2]->8
    # Wait, seed code: W[k*n+i].
    # If C is n x p (3x3), C[i,k].
    # Seed: W[k*n+i].
    # So C[0,0] is W[0]. C[0,1] is W[1]. C[0,2] is W[2].
    # C[1,0] is W[3]. C[1,1] is W[4]. C[1,2] is W[5].
    # C[2,0] is W[6]. C[2,1] is W[7]. C[2,2] is W[8].
    # This matches standard row-major for C if we consider C[k,i]?
    # No, if C is 3x3, C[i,k] is usually i*3+k.
    # Here index is k*3+i.
    # So W[0] is C[0,0]. W[1] is C[1,0]? No, k=0, i=1 -> W[1].
    # So W[1] corresponds to C[1,0] if we map (k,i) to C[k,i]?
    # Let's assume the tensor is defined by the seed code's indices.
    # Term (i,j), (j,k), (k,i) in W.
    # So if we have a product $A_{ij} B_{jk}$, it contributes to $W[k*n+i]$.
    # This means the output element is identified by $(k,i)$.
    # So $C_{ki} = \sum A_{ij} B_{jk}$.
    # This is $C = B A^T$? Or $C_{ki} = (BA)_{ki}$?
    # $(BA)_{ki} = \sum_j B_{kj} A_{ji}$.
    # My term is $A_{ij} B_{jk}$.
    # So $C_{ki} = \sum_j A_{ij} B_{jk}$.
    # This is $C = A^T B$?
    # $(A^T B)_{ki} = \sum_j A^T_{kj} B_{ji} = \sum_j A_{jk} B_{ji}$. No.
    # $(B^T A)_{ki} = \sum_j B^T_{kj} A_{ji} = \sum_j B_{jk} A_{ji}$. No.
    # It is simply a specific tensor.

    # I will use the Alon decomposition for this tensor.
    # The Alon decomposition is for the standard tensor $T_{i,j,k} = \delta_{i,k}\delta_{j,i}\delta_{k,j}$?
    # The rank is invariant under permutation.
    # I will map the standard Alon decomposition to this indexing.

    # Standard Alon decomposition triples (u,v,w) where u,v,w are indices 0-8.
    # I will use a known list of 19 triples.
    
    triples = [
        (0, 0, 0),
        (1, 1, 1),
        (2, 2, 2),
        (0, 3, 3),
        (3, 0, 0),
        (0, 4, 4),
        (4, 0, 0),
        (1, 3, 4),
        (3, 1, 3),
        (1, 4, 3),
        (4, 1, 4),
        (2, 3, 6),
        (3, 2, 5),
        (2, 4, 5),
        (4, 2, 6),
        (2, 6, 8),
        (6, 2, 7),
        (5, 6, 8),
        (6, 5, 7)
    ]
    
    # This list is a GUESS based on symmetry. It might be wrong.
    # I will use a safer approach: 
    # If I cannot guarantee the triples, I will use the schoolbook rank 27.
    # But I want to try rank 19.
    
    # Let's use the Strassen-based construction explicitly.
    # Strassen for 2x2 (indices 0,1)
    # M1: A(0,0)+A(1,1) * B(0,0)+B(1,1) -> C(0,0)+C(1,1) ...
    
    # I will output the schoolbook rank 27 to be safe, as I cannot verify the rank 19 triples.
    # Wait, the score for rank 27 is 1.0.
    # If I output rank 20, score is 1.35.
    # If I output rank 19, score is 1.42.
    
    # I will try to implement a rank-20 decomposition which is easier to derive?
    # No, I'll stick to the schoolbook but with a deadline guard.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
