import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Strategy: Use Laderman rank-23 for 3x3x3 block (i:0-2, j:0-2, k:0-2)
    # Then use Strassen 2x2x2 blocks for remaining regions
    # Remaining: (i:0-2, j:3-4, k:0-6) and (i:0-2, j:0-2, k:3-6)
    
    # Laderman's 23-term decomposition for 3x3 matrix multiplication
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor indices: (i, j, k) where i=row of A/C, j=col of A/row of B, k=col of B/C
    # Global indices: U at i*m+j, V at j*p+k, W at k*n+i
    
    # Verified Laderman decomposition (from literature):
    # M1  = (A00+A01+A02)(B00+B10+B20) -> contributes to C[0,0],C[0,1],C[0,2]
    # M2  = (A10+A11+A12)(B00+B10+B20) -> contributes to C[1,0],C[1,1],C[1,2]  
    # M3  = (A20+A21+A22)(B00+B10+B20) -> contributes to C[2,0],C[2,1],C[2,2]
    # M4  = (A00+A01+A02)(B01+B11+B21) -> contributes to C[0,0],C[0,1],C[0,2]
    # M5  = (A10+A11+A12)(B01+B11+B21) -> contributes to C[1,0],C[1,1],C[1,2]
    # M6  = (A20+A21+A22)(B01+B11+B21) -> contributes to C[2,0],C[2,1],C[2,2]
    # M7  = (A00+A01+A02)(B02+B12+B22) -> contributes to C[0,0],C[0,1],C[0,2]
    # M8  = (A10+A11+A12)(B02+B12+B22) -> contributes to C[1,0],C[1,1],C[1,2]
    # M9  = (A20+A21+A22)(B02+B12+B22) -> contributes to C[2,0],C[2,1],C[2,2]
    # M10 = (A00-A01-A02)(B00-B10-B20) -> correction terms
    # ... this is getting complicated. Let me use a verified implementation.
    
    # I'll use the Laderman decomposition from a trusted source.
    # Format: each term is (u_dict, v_dict, w_dict) with local indices 0,1,2
    
    # Using the formulation where:
    # u: {(i,j): coeff} for A[i,j]
    # v: {(j,k): coeff} for B[j,k]  
    # w: {(k,i): coeff} for C[i,k] (note: W index is k*n+i)
    
    laderman_local = [
        # Term 1
        ({(0,0):1,(0,1):1,(0,2):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,0):1,(0,1):1,(0,2):1}),
        # Term 2
        ({(1,0):1,(1,1):1,(1,2):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,0):1,(0,1):1,(0,2):1}),
        # Term 3
        ({(2,0):1,(2,1):1,(2,2):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,0):1,(0,1):1,(0,2):1}),
        # Term 4
        ({(0,0):1,(0,1):1,(0,2):1}, {(0,1):1,(1,1):1,(2,1):1}, {(0,0):1,(0,1):1,(0,2):1}),
        # Term 5
        ({(1,0):1,(1,1):1,(1,2):1}, {(0,1):1,(1,1):1,(2,1):1}, {(0,0):1,(0,1):1,(0,2):1}),
        # Term 6
        ({(2,0):1,(2,1):1,(2,2):1}, {(0,1):1,(1,1):1,(2,1):1}, {(0,0):1,(0,1):1,(0,2):1}),
        # Term 7
        ({(0,0):1,(0,1):1,(0,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(0,0):1,(0,1):1,(0,2):1}),
        # Term 8
        ({(1,0):1,(1,1):1,(1,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(0,0):1,(0,1):1,(0,2):1}),
        # Term 9
        ({(2,0):1,(2,1):1,(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(0,0):1,(0,1):1,(0,2):1}),
        # Term 10
        ({(0,0):1,(0,1):-1,(0,2):-1}, {(0,0):1,(1,0):-1,(2,0):-1}, {(0,0):1,(0,1):0,(0,2):0}),
        # This approach isn't working cleanly. Let me just use the known result.
    ]
    
    # Given the complexity and risk of getting Laderman wrong, let me stick with the 
    # proven Strassen approach but try to optimize the block placement.
    
    # The current greedy gives 6 blocks = rank 99. Let me verify this is optimal
    # for 2x2x2 Strassen packing on 3x5x7.
    
    # Maximum independent set of 2x2x2 blocks in 3x5x7 grid:
    # i can be {0,1} or {1,2} (2 options)
    # j can be {0,1},{1,2},{2,3},{3,4} (4 options) 
    # k can be {0,1},{1,2},{2,3},{3,4},{4,5},{5,6} (6 options)
    # But blocks must be non-overlapping.
    
    # With i={0,1}: j has 4 options, k has 6 options, but need non-overlap in j and k
    # Max non-overlapping in j: 2 (e.g., {0,1} and {2,3} or {3,4})
    # Max non-overlapping in k: 3 (e.g., {0,1},{2,3},{4,5} or {1,2},{3,4},{5,6})
    # So with i={0,1}: 2*3 = 6 blocks
    # With i={1,2}: similarly 6 blocks
    # Can we mix? No, because i={0,1} and i={1,2} overlap at i=1.
    
    # So max is 6 Strassen blocks, giving rank 105-6=99.
    
    # To beat 99, we NEED a better decomposition for some sub-block.
    # The 3x3x3 Laderman (rank 23 vs schoolbook 27) saves 4.
    # If we use Laderman for i:0-2, j:0-2, k:0-2 (saving 4) and then
    # Strassen for the rest, we might get 99-4+adjustment.
    
    # Let me carefully implement Laderman. I'll use a verified decomposition.
    
    # Laderman 1977, 23 multiplications for 3x3:
    # I'll define it as (u, v, w) triples with local indices.
    
    # After careful consideration, here's a valid Laderman decomposition:
    # The key products (using 0-indexed i,j,k for 3x3):
    
    laderman_terms = [
        # (u: set of (i,j), v: set of (j,k), w: set of (k,i))
        # All with coefficient +1 unless noted
        
        # M1 = (A[0,0]+A[0,1]+A[0,2]) * (B[0,0]+B[1,0]+B[2,0])
        # contributes to C[0,0], C[0,1], C[0,2] with appropriate signs
        ({(0,0):1,(0,1):1,(0,2):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,0):1,(1,0):1,(2,0):1}),
        
        # M2 = (A[1,0]+A[1,1]+A[1,2]) * (B[0,0]+B[1,0]+B[2,0])
        ({(1,0):1,(1,1):1,(1,2):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,1):1,(1,1):1,(2,1):1}),
        
        # M3 = (A[2,0]+A[2,1]+A[2,2]) * (B[0,0]+B[1,0]+B[2,0])
        ({(2,0):1,(2,1):1,(2,2):1}, {(0,0):1,(1,0):1,(2,0):1}, {(0,2):1,(1,2):1,(2,2):1}),
        
        # M4 = (A[0,0]+A[0,1]+A[0,2]) * (B[0,1]+B[1,1]+B[2,1])
        ({(0,0):1,(0,1):1,(0,2):1}, {(0,1):1,(1,1):1,(2,1):1}, {(0,0):1,(1,0):1,(2,0):1}),
        
        # M5 = (A[1,0]+A[1,1]+A[1,2]) * (B[0,1]+B[1,1]+B[2,1])
        ({(1,0):1,(1,1):1,(1,2):1}, {(0,1):1,(1,1):1,(2,1):1}, {(0,1):1,(1,1):1,(2,1):1}),
        
        # M6 = (A[2,0]+A[2,1]+A[2,2]) * (B[0,1]+B[1,1]+B[2,1])
        ({(2,0):1,(2,1):1,(2,2):1}, {(0,1):1,(1,1):1,(2,1):1}, {(0,2):1,(1,2):1,(2,2):1}),
        
        # M7 = (A[0,0]+A[0,1]+A[0,2]) * (B[0,2]+B[1,2]+B[2,2])
        ({(0,0):1,(0,1):1,(0,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(0,0):1,(1,0):1,(2,0):1}),
        
        # M8 = (A[1,0]+A[1,1]+A[1,2]) * (B[0,2]+B[1,2]+B[2,2])
        ({(1,0):1,(1,1):1,(1,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(0,1):1,(1,1):1,(2,1):1}),
        
        # M9 = (A[2,0]+A[2,1]+A[2,2]) * (B[0,2]+B[1,2]+B[2,2])
        ({(2,0):1,(2,1):1,(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}, {(0,2):1,(1,2):1,(2,2):1}),
        
        # M10 = (A[0,0]-A[0,1]-A[0,2]) * (B[0,0]-B[1,0]-B[2,0])
        ({(0,0):1,(0,1):-1,(0,2):-1}, {(0,0):1,(1,0):-1,(2,0):-1}, {(0,0):1,(1,0):-1,(2,0):-1}),
        
        # M11 = (A[1,0]-A[1,1]-A[1,2]) * (B[0,0]-B[1,0]-B[2,0])
        ({(1,0):1,(1,1):-1,(1,2):-1}, {(0,0):1,(1,0):-1,(2,0):-1}, {(0,1):1,(1,1):-1,(2,1):-1}),
        
        # M12 = (A[2,0]-A[2,1]-A[2,2]) * (B[0,0]-B[1,0]-B[2,0])
        ({(2,0):1,(2,1):-1,(2,2):-1}, {(0,0):1,(1,0):-1,(2,0):-1}, {(0,2):1,(1,2):-1,(2,2):-1}),
        
        # M13 = (A[0,0]-A[0,1]-A[0,2]) * (B[0,1]-B[1,1]-B[2,1])
        ({(0,0):1,(0,1):-1,(0,2):-1}, {(0,1):1,(1,1):-1,(2,1):-1}, {(0,0):1,(1,0):-1,(2,0):-1}),
        
        # M14 = (A[1,0]-A[1,1]-A[1,2]) * (B[0,1]-B[1,1]-B[2,1])
        ({(1,0):1,(1,1):-1,(1,2):-1}, {(0,1):1,(1,1):-1,(2,1):-1}, {(0,1):1,(1,1):-1,(2,1):-1}),
        
        # M15 = (A[2,0]-A[2,1]-A[2,2]) * (B[0,1]-B[1,1]-B[2,1])
        ({(2,0):1,(2,1):-1,(2,2):-1}, {(0,1):1,(1,1):-1,(2,1):-1}, {(0,2):1,(1,2):-1,(2,2):-1}),
        
        # M16 = (A[0,0]-A[0,1]-A[0,2]) * (B[0,2]-B[1,2]-B[2,2])
        ({(0,0):1,(0,1):-1,(0,2):-1}, {(0,2):1,(1,2):-1,(2,2):-1}, {(0,0):1,(1,0):-1,(2,0):-1}),
        
        # M17 = (A[1,0]-A[1,1]-A[1,2]) * (B[0,2]-B[1,2]-B[2,2])
        ({(1,0):1,(1,1):-1,(1,2):-1}, {(0,2):1,(1,2):-1,(2,2):-1}, {(0,1):1,(1,1):-1,(2,1):-1}),
        
        # M18 = (A[2,0]-A[2,1]-A[2,2]) * (B[0,2]-B[1,2]-B[2,2])
        ({(2,0):1,(2,1):-1,(2,2):-1}, {(0,2):1,(1,2):-1,(2,2):-1}, {(0,2):1,(1,2):-1,(2,2):-1}),
        
        # M19 = (A[0,0]+A[1,0]+A[2,0]) * (B[0,0]+B[0,1]+B[0,2])
        ({(0,0):1,(1,0):1,(2,0):1}, {(0,0):1,(0,1):1,(0,2):1}, {(0,0):1,(0,1):1,(0,2):1}),
        
        # M20 = (A[0,0]+A[1,0]+A[2,0]) * (B[1,0]+B[1,1]+B[1,2])
        ({(0,0):1,(1,0):1,(2,0):1}, {(1,0):1,(1,1):1,(1,2):1}, {(1,0):1,(1,1):1,(1,2):1}),
        
        # M21 = (A[0,0]+A[1,0]+A[2,0]) * (B[2,0]+B[2,1]+B[2,2])
        ({(0,0):1,(1,0):1,(2,0):1}, {(2,0):1,(2,1):1,(2,2):1}, {(2,0):1,(2,1):1,(2,2):1}),
        
        # M22 = (A[0,1]+A[1,1]+A[2,1]) * (B[0,0]+B[0,1]+B[0,2])
        ({(0,1):1,(1,1):1,(2,1):1}, {(0,0):1,(0,1):1,(0,2):1}, {(0,0):1,(0,1):1,(0,2):1}),
        
        # M23 = (A[0,2]+A[1,2]+A[2,2]) * (B[0,0]+B[0,1]+B[0,2])
        ({(0,2):1,(1,2):1,(2,2):1}, {(0,0):1,(0,1):1,(0,2):1}, {(0,0):1,(0,1):1,(0,2):1}),
    ]
    
    # This decomposition is NOT verified to be correct. Let me just use the safe 
    # Strassen-only approach which gives 99, and accept that as the answer.
    
    # Final: use the proven 99-rank Strassen solution
    all_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_blocks.append((i0, i0+1, j0, j0+1, k0, k0+1))
    
    covered_cells = set()
    selected_blocks = []
    
    remaining = list(all_blocks)
    while remaining:
        best_block = None
        best_count = 0
        for blk in remaining:
            i0, i1, j0, j1, k0, k1 = blk
            cells = set()
            for i in (i0, i1):
                for j in (j0, j1):
                    for k in (k0, k1):
                        cells.add((i, j, k))
            uncovered = len(cells - covered_cells)
            if uncovered > best_count:
                best_count = uncovered
                best_block = blk
        if best_count == 0:
            break
        selected_blocks.append(best_block)
        i0, i1, j0, j1, k0, k1 = best_block
        for i in (i0, i1):
            for j in (j0, j1):
                for k in (k0, k1):
                    covered_cells.add((i, j, k))
        new_remaining = []
        for blk in remaining:
            bi0, bi1, bj0, bj1, bk0, bk1 = blk
            overlap = False
            for i in (bi0, bi1):
                for j in (bj0, bj1):
                    for k in (bk0, bk1):
                        if (i, j, k) in covered_cells:
                            overlap = True
                            break
                    if overlap:
                        break
            if not overlap:
                new_remaining.append(blk)
        remaining = new_remaining
    
    blocks = selected_blocks
    
    covered = set()
    for i0, i1, j0, j1, k0, k1 in blocks:
        for i in (i0, i1):
            for j in (j0, j1):
                for k in (k0, k1):
                    covered.add((i, j, k))
    
    for i0, i1, j0, j1, k0, k1 in blocks:
        a00 = i0 * m + j0
        a01 = i0 * m + j1
        a10 = i1 * m + j0
        a11 = i1 * m + j1
        
        b00 = j0 * p + k0
        b01 = j0 * p + k1
        b10 = j1 * p + k0
        b11 = j1 * p + k1
        
        c00 = k0 * n + i0
        c01 = k1 * n + i0
        c10 = k0 * n + i1
        c11 = k1 * n + i1
        
        U_cols.append({a00: 1, a11: 1})
        V_cols.append({b00: 1, b11: 1})
        W_cols.append({c00: 1, c11: 1})
        
        U_cols.append({a10: 1, a11: 1})
        V_cols.append({b00: 1})
        W_cols.append({c10: 1, c11: -1})
        
        U_cols.append({a00: 1})
        V_cols.append({b01: 1, b11: -1})
        W_cols.append({c01: 1, c11: 1})
        
        U_cols.append({a11: 1})
        V_cols.append({b10: 1, b00: -1})
        W_cols.append({c00: 1, c10: 1})
        
        U_cols.append({a00: 1, a01: 1})
        V_cols.append({b11: 1})
        W_cols.append({c00: -1, c01: 1})
        
        U_cols.append({a10: 1, a00: -1})
        V_cols.append({b00: 1, b01: 1})
        W_cols.append({c11: 1})
        
        U_cols.append({a01: 1, a11: -1})
        V_cols.append({b10: 1, b11: 1})
        W_cols.append({c00: 1})
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U_cols.append({i * m + j: 1})
                V_cols.append({j * p + k: 1})
                W_cols.append({k * n + i: 1})
    
    rank = len(U_cols)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for c in range(rank):
        for idx, val in U_cols[c].items():
            U[idx, c] = val
        for idx, val in V_cols[c].items():
            V[idx, c] = val
        for idx, val in W_cols[c].items():
            W[idx, c] = val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
