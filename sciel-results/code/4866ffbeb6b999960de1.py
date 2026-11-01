import numpy as np
import time

def construct():
    """
    Rank-28 exact tensor decomposition for 2x3 @ 3x5 matrix multiplication.
    
    U: (6, 28)  - factors for A (2x3), indexed as U[i*3+j, r]
    V: (15, 28) - factors for B (3x5), indexed as V[j*5+k, r]
    W: (10, 28) - factors for C (2x5), indexed as W[k*2+i, r]
    
    Convention: C[i,k] = sum_j A[i,j] * B[j,k]
    Tensor: T[i,j,k] = sum_r U[3*i+j, r] * V[5*j+k, r] * W[2*k+i, r]
    """
    t0 = time.time()
    n, m, p = 2, 3, 5
    R = 28
    
    # Build factor lists as (row_U, row_V, row_W, val_U, val_V, val_W)
    # Each term r contributes U[u,r]*V[v,r]*W[w,r] to T[i,j,k]
    # where u = 3*i+j, v = 5*j+k, w = 2*k+i
    
    terms = []
    
    # --- Block A: j=0, all k (schoolbook, 2*5=10 terms) ---
    # For j=0: A[i,0]*B[0,k] -> C[i,k]
    # Term: U[3*i+0] * V[5*0+k] * W[2*k+i] = 1
    for i in range(n):
        for k in range(p):
            terms.append((3*i + 0, 5*0 + k, 2*k + i, 1, 1, 1))
    # That's 10 terms, cols 0-9
    
    # --- Block B: j in {1,2}, k in {0,1} (Strassen 2x2x2, 7 terms) ---
    # Local: ii=i, jj=j-1, kk=k
    # U index: 3*i + (1+jj), V index: 5*(1+jj) + kk, W index: 2*kk + i
    def strassen_block(j_base, k_base):
        """Yield 7 Strassen terms for a 2x2x2 block at (j_base, k_base)."""
        # M1: (A[i,0]+A[i,1])*(B[0,0]+B[1,1]) -> C[i,0] and C[i,1] contribution
        # Actually let me be more careful with the standard Strassen for matrix mult
        # For multiplying (2x2) * (2x2) -> (2x2):
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
        
        # In our indexing: i in {0,1}, j in {j_base, j_base+1}, k in {k_base, k_base+1}
        # A[i, j_base+jj], B[j_base+jj, k_base+kk], C[i, k_base+kk]
        # U row: 3*i + (j_base+jj)
        # V row: 5*(j_base+jj) + (k_base+kk)
        # W row: 2*(k_base+kk) + i
        
        jb, kb = j_base, k_base
        
        # M1: (A[0,jb]+A[1,jb+1])(B[jb,kb]+B[jb+1,kb+1])
        # Contributes to C[0,kb] (+1) and C[1,kb+1] (+1)
        yield (3*0+jb, 5*jb+kb, 2*kb+0, 1, 1, 1)
        yield (3*1+jb+1, 5*jb+kb, 2*kb+0, 1, 1, 1)  # This is wrong, let me redo
        
        # Let me be very explicit:
        # M1: (A[0,jb]+A[0,jb+1]) * (B[jb,kb]+B[jb+1,kb+1])
        # Wait, standard Strassen for C = AB where A is 2x2, B is 2x2:
        # a00=A[0,0], a01=A[0,1], a10=A[1,0], a11=A[1,1]
        # b00=B[0,0], b01=B[0,1], b10=B[1,0], b11=B[1,1]
        # M1=(a00+a11)(b00+b11)
        # M2=(a10+a11)b00
        # M3=a00(b01-b11)
        # M4=a11(b10-b00)
        # M5=(a00+a01)b11
        # M6=(a10-a00)(b00+b01)
        # M7=(a01-a11)(b10+b11)
        
        # In our case: A is 2x2 with rows i=0,1 and cols j=jb,jb+1
        # So a00=A[0,jb], a01=A[0,jb+1], a10=A[1,jb], a11=A[1,jb+1]
        # B is 2x2 with rows j=jb,jb+1 and cols k=kb,kb+1
        # So b00=B[jb,kb], b01=B[jb,kb+1], b10=B[jb+1,kb], b11=B[jb+1,kb+1]
        # C is 2x2 with rows i=0,1 and cols k=kb,kb+1
        # c00=C[0,kb], c01=C[0,kb+1], c10=C[1,kb], c11=C[1,kb+1]
        
        # M1=(a00+a11)(b00+b11) -> c00+=M1, c11+=M1
        # U entries: U[3*0+jb]+=1, U[3*1+jb+1]+=1
        # V entries: V[5*jb+kb]+=1, V[5*(jb+1)+(kb+1)]+=1
        # W entries: W[2*kb+0]+=1, W[2*(kb+1)+1]+=1
        yield (3*0+jb, 5*jb+kb, 2*kb+0, 1, 1, 1)
        yield (3*1+jb+1, 5*jb+kb, 2*kb+0, 1, 1, 1)  # WRONG - same V,W
        
        # I need to rethink. Each Strassen term is ONE rank-1 term.
        # M1 = (a00+a11)(b00+b11) is a single scalar that contributes to two outputs.
        # As a rank-1 tensor term: it's the product of:
        #   (delta_{i,0}*delta_{j,jb} + delta_{i,1}*delta_{j,jb+1}) for A part
        #   (delta_{j,jb}*delta_{k,kb} + delta_{j,jb+1}*delta_{k,kb+1}) for B part
        #   (delta_{k,kb}*delta_{i,0} + delta_{k,kb+1}*delta_{i,1}) for C part
        # This is NOT rank 1! It's rank 4.
        
        # I think the issue is that Strassen doesn't directly give rank-1 terms for the tensor.
        # Let me go back to the original approach which worked.
        pass
        
    # Let me just directly construct the known working decomposition with different code structure
    terms = []
    
    # Block A: j=0 (10 terms)
    for i in range(n):
        for k in range(p):
            terms.append((3*i, k, 2*k+i, 1, 1, 1))
    
    # Block B: Strassen for j in {1,2}, k in {0,1} (7 terms)
    # Using the same structure as the original but written differently
    # Original Strassen M1-M7 for block at j_offset=1, k_offset=0
    
    # M1: U[1,1]=1, U[2,2]=1, V[5,0]=1, V[6,1]=1, W[0,0]=1, W[5,1]=1
    # In my indexing: U[3*0+1]=U[1], U[3*1+2]=U[5], V[5*1+0]=V[5], V[5*2+1]=V[11]
    # Wait, let me just copy the exact indices from the original.
    
    # I'll build a list of (u_idx, v_idx, w_idx, u_val, v_val, w_val) for each of the 28 terms
    
    # Block A (10 terms): j=0
    for i in range(n):
        for k in range(p):
            terms.append((3*i+0, 0*5+k, 2*k+i, 1, 1, 1))
    
    # Block B (7 terms): Strassen j={1,2}, k={0,1}
    # From original _strassen_2x2x2 with i_offset=0, j_offset=1, k_offset=0, n=2, m=3, p=5
    # M1: U[0*3+1]=U[1]=1, U[1*3+2]=U[5]=1, V[1*5+0]=V[5]=1, V[2*5+1]=V[11]=1, W[0*2+0]=W[0]=1, W[1*2+1]=W[3]=1
    terms.append((1, 5, 0, 1, 1, 1))    # U[1], V[5], W[0]
    terms.append((5, 5, 0, 1, 1, 1))    # WRONG - these are separate rank-1 terms
    # No, each term in the list is a separate rank-1 term. The original had separate assignments.
    # Let me re-read the original code.
    
    # Original: for M1, it sets:
    # U[1*3+1][col] = 1  -> U[4][col] = 1  (wait, i_offset=0, so U[0*3+1]=U[1])
    # Actually: U[i_offset*m + j_offset] = U[0*3+1] = U[1]
    # U[(i_offset+1)*m + (j_offset+1)] = U[1*3+2] = U[5]
    # V[j_offset*p + k_offset] = V[1*5+0] = V[5]
    # V[(j_offset+1)*p + (k_offset+1)] = V[2*5+1] = V[11]
    # W[k_offset*n + i_offset] = W[0*2+0] = W[0]
    # W[(k_offset+1)*n + (i_offset+1)] = W[1*2+1] = W[3]
    
    # But these are ALL set in the SAME column! So it's one rank-1 term with:
    # U[1]=1, U[5]=1, V[5]=1, V[11]=1, W[0]=1, W[3]=1
    # That's not a single rank-1 term in the usual sense... 
    
    # Actually, in tensor decomposition, a rank-1 term means:
    # T[i,j,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For a single r, we pick one entry from U, one from V, one from W.
    # So if U[1, r]=1 and U[5, r]=1, that means this rank-1 term has non-zero values at two positions in U.
    # That's still valid - U[i*m+j, r] can be non-zero for multiple (i,j) pairs.
    
    # So each column r defines a rank-1 term in the bilinear decomposition.
    # The term contributes: sum over all (i,j,k) of U[3i+j, r] * V[5j+k, r] * W[2k+i, r]
    
    # OK so I just need to replicate the exact same non-zero patterns.
    # Let me build the full 28-term decomposition directly.
    
    terms = []
    
    # Block A: 10 terms (j=0)
    for i in range(n):
        for k in range(p):
            terms.append({
                'u': {3*i+0: 1},
                'v': {0*5+k: 1},
                'w': {2*k+i: 1}
            })
    
    # Block B: 7 Strassen terms for j={1,2}, k={0,1}
    # M1
    terms.append({
        'u': {1: 1, 5: 1},
        'v': {5: 1, 11: 1},
        'w': {0: 1, 3: 1}
    })
    # M2: (A[1,1]+A[1,2])B[1,0] -> C[1,0], -C[1,1]
    terms.append({
        'u': {4: 1, 5: 1},
        'v': {5: 1},
        'w': {0: 1, 3: -1}
    })
    # M3: A[0,1](B[2,1]-B[1,1]) -> C[0,1], C[1,1]
    terms.append({
        'u': {1: 1},
        'v': {11: 1, 5: -1},
        'w': {3: 1, 1: 1}
    })
    # M4: A[1,2](B[1,1]-B[2,0]) -> C[0,0], C[1,0]
    terms.append({
        'u': {5: 1},
        'v': {10: 1, 5: -1},
        'w': {0: 1, 2: 1}
    })
    # M5: (A[0,1]+A[0,2])B[2,1] -> -C[0,0], C[0,1]
    terms.append({
        'u': {1: 1, 2: 1},
        'v': {11: 1},
        'w': {0: -1, 3: 1}
    })
    # M6: (A[1,1]-A[0,1])(B[1,0]+B[1,1]) -> C[1,1]
    terms.append({
        'u': {4: 1, 1: -1},
        'v': {5: 1, 6: 1},
        'w': {3: 1}
    })
    # M7: (A[0,2]-A[1,2])(B[2,0]+B[2,1]) -> C[0,0]
    terms.append({
        'u': {2: 1, 5: -1},
        'v': {10: 1, 11: 1},
        'w': {0: 1}
    })
    
    # Block C: 4 terms (j={1,2}, k=2)
    for i in range(n):
        for j in [1, 2]:
            terms.append({
                'u': {3*i+j: 1},
                'v': {5*j+2: 1},
                'w': {2*2+i: 1}
            })
    
    # Block D: 7 Strassen terms for j={1,2}, k={3,4}
    # Same structure as Block B but with k_offset=3
    # M1
    terms.append({
        'u': {1: 1, 5: 1},
        'v': {5*1+3: 1, 5*2+4: 1},
        'w': {2*3+0: 1, 2*4+1: 1}
    })
    # M2
    terms.append({
        'u': {4: 1, 5: 1},
        'v': {5*1+3: 1},
        'w': {2*3+0: 1, 2*4+1: -1}
    })
    # M3
    terms.append({
        'u': {1: 1},
        'v': {5*2+4: 1, 5*1+4: -1},
        'w': {2*4+0: 1, 2*4+1: 1}
    })
    # M4
    terms.append({
        'u': {5: 1},
        'v': {5*2+3: 1, 5*1+3: -1},
        'w': {2*3+0: 1, 2*3+1: 1}
    })
    # M5
    terms.append({
        'u': {1: 1, 2: 1},
        'v': {5*2+4: 1},
        'w': {2*3+0: -1, 2*4+0: 1}
    })
    # M6
    terms.append({
        'u': {4: 1, 1: -1},
        'v': {5*1+3: 1, 5*1+4: 1},
        'w': {2*4+1: 1}
    })
    # M7
    terms.append({
        'u': {2: 1, 5: -1},
        'v': {5*2+3: 1, 5*2+4: 1},
        'w': {2*3+0: 1}
    })
    
    assert len(terms) == R, f"Expected {R} terms, got {len(terms)}"
    
    # Build U, V, W matrices
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    for r, term in enumerate(terms):
        for idx, val in term['u'].items():
            U[idx, r] = val
        for idx, val in term['v'].items():
            V[idx, r] = val
        for idx, val in term['w'].items():
            W[idx, r] = val
    
    # Verify: check that the decomposition is correct
    # T[i,j,k] should equal sum_r U[3i+j, r] * V[5j+k, r] * W[2k+i, r]
    # And T[i,j,k] = 1 for all (i,j,k) since it's matrix multiplication
    # Actually T[i,j,k] = 1 always for the matrix multiplication tensor
    # Let's verify a few entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[3*i+j, r] * V[5*j+k, r] * W[2*k+i, r] for r in range(R))
                if abs(val - 1) > 1e-10:
                    raise ValueError(f"Decomposition error at ({i},{j},{k}): got {val}, expected 1")
    
    # Deadline guard
    if time.time() - t0 > 110:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
