"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 65: 49 (Strassen 4x4x4) + 16 (Direct 4x4x1)
    rank_total = 65
    U = [[0.0] * rank_total for _ in range(n * m)]
    V = [[0.0] * rank_total for _ in range(m * p)]
    W = [[0.0] * rank_total for _ in range(p * n)]
    
    col = 0
    
    class LinearForm:
        def __init__(self, vec):
            self.vec = vec[:]
            
    def add_linforms(lf1, lf2):
        return LinearForm([a + b for a, b in zip(lf1.vec, lf2.vec)])
        
    def sub_linforms(lf1, lf2):
        return LinearForm([a - b for a, b in zip(lf1.vec, lf2.vec)])
        
    # Initialize basis forms for A (16 elements)
    A_basis = [LinearForm([0]*16) for _ in range(16)]
    for e in range(16):
        A_basis[e].vec[e] = 1
        
    B_basis = [LinearForm([0]*16) for _ in range(16)]
    for e in range(16):
        B_basis[e].vec[e] = 1
        
    def get_block_basis(basis, block_id):
        row_block = block_id // 2
        col_block = block_id % 2
        elems = []
        for r in range(2):
            for c in range(2):
                idx = (2*row_block + r) * 4 + (2*col_block + c)
                elems.append(basis[idx])
        return elems
        
    A_blocks = [get_block_basis(A_basis, b) for b in range(4)]
    B_blocks = [get_block_basis(B_basis, b) for b in range(4)]
    
    def add_blocks(B1, B2):
        return [[add_linforms(B1[i][j], B2[i][j]) for j in range(2)] for i in range(2)]
        
    def sub_blocks(B1, B2):
        return [[sub_linforms(B1[i][j], B2[i][j]) for j in range(2)] for i in range(2)]
        
    def copy_blocks(B1):
        return [row[:] for row in B1]

    OA = []
    OB = []
    
    OA.append(add_blocks(A_blocks[0], A_blocks[3]))
    OB.append(add_blocks(B_blocks[0], B_blocks[3]))
    
    OA.append(add_blocks(A_blocks[2], A_blocks[3]))
    OB.append(copy_blocks(B_blocks[0]))
    
    OA.append(copy_blocks(A_blocks[0]))
    OB.append(sub_blocks(B_blocks[1], B_blocks[3]))
    
    OA.append(copy_blocks(A_blocks[3]))
    OB.append(sub_blocks(B_blocks[2], B_blocks[0]))
    
    OA.append(add_blocks(A_blocks[0], A_blocks[1]))
    OB.append(copy_blocks(B_blocks[3]))
    
    OA.append(sub_blocks(A_blocks[2], A_blocks[0]))
    OB.append(add_blocks(B_blocks[0], B_blocks[1]))
    
    OA.append(sub_blocks(A_blocks[1], A_blocks[3]))
    OB.append(add_blocks(B_blocks[2], B_blocks[3]))
    
    c_i = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)]
    ]
    
    outer_map = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)]
    ]
    
    for s in range(7):
        X = OA[s]
        Y = OB[s]
        
        inner_products = []
        
        A_lf = add_linforms(X[0][0], X[1][1])
        B_lf = add_linforms(Y[0][0], Y[1][1])
        inner_products.append((A_lf, B_lf))
        
        A_lf = add_linforms(X[1][0], X[1][1])
        B_lf = Y[0][0]
        inner_products.append((A_lf, B_lf))
        
        A_lf = X[0][0]
        B_lf = sub_linforms(Y[0][1], Y[1][1])
        inner_products.append((A_lf, B_lf))
        
        A_lf = X[1][1]
        B_lf = sub_linforms(Y[1][0], Y[0][0])
        inner_products.append((A_lf, B_lf))
        
        A_lf = add_linforms(X[0][0], X[0][1])
        B_lf = Y[1][1]
        inner_products.append((A_lf, B_lf))
        
        A_lf = sub_linforms(X[1][0], X[0][0])
        B_lf = add_linforms(Y[0][0], Y[0][1])
        inner_products.append((A_lf, B_lf))
        
        A_lf = sub_linforms(X[0][1], X[1][1])
        B_lf = add_linforms(Y[1][0], Y[1][1])
        inner_products.append((A_lf, B_lf))
        
        for k in range(7):
            A_lf, B_lf = inner_products[k]
            
            W_vec = [0.0] * 16
            
            for r_b, c_b, sign_inner in c_i[k]:
                for cb_r, cb_c, sign_outer in outer_map[s]:
                    global_idx = (2*cb_r + r_b) * 4 + (2*cb_c + c_b)
                    coeff = sign_inner * sign_outer
                    W_vec[global_idx] += coeff
            
            for i in range(16):
                U[i][col] = A_lf.vec[i]
            
            for j in range(4):
                for k_c in range(4):
                    v_idx = j * 5 + k_c
                    b_lf_idx = j * 4 + k_c
                    V[v_idx][col] = B_lf.vec[b_lf_idx]
            
            for r in range(4):
                for c in range(4):
                    w_vec_idx = r * 4 + c
                    w_global_idx = c * 4 + r
                    W[w_global_idx][col] = W_vec[w_vec_idx]
            
            col += 1
    
    for i in range(4):
        for j in range(4):
            u_idx = i * 4 + j
            v_idx = j * 5 + 4
            w_idx = 4 * 4 + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
