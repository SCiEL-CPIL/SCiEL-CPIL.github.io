def _verify(U, V, W, n, m, p, rank):
    Ua = np.array(U, dtype=np.float64).reshape(n, m, rank)
    Va = np.array(V, dtype=np.float64).reshape(m, p, rank)
    Wa = np.array(W, dtype=np.float64).reshape(p, n, rank)
    
    # Compute C_{i, l} = sum_{j, r} U_{i, j, r} V_{j, l, r} W_{l, i, r}
    # Use einsum: 'ijr, jlr, lir -> il'
    C_decomp = np.einsum('ijr,jlr,lir->il', Ua, Va, Wa)
    
    # Check against true matrix multiplication
    # We need to verify the tensor identity.
    # The tensor identity is:
    # T_{i, j, l} = sum_r U_{i, j, r} V_{j, l, r} W_{l, i, r} ?
    # No, the tensor T is in V_A^* \otimes V_B^* \otimes V_C^*.
    # The decomposition is T = sum U_r \otimes V_r \otimes W_r.
    # So T_{i, j, l} (where i,j are A, j,l are B, i,l are C?)
    # Actually, the indices in the decomposition match the free indices of the bilinear map.
    # The bilinear map is (A, B) -> C.
    # C_{i, l} = sum_{j, r} U_{i, j, r} V_{j, l, r} W_{l, i, r}.
    # This equation must hold for the tensor coefficients.
    # This is equivalent to checking:
    # For all i, l: C_{i, l} (from decomp) matches the structure constant.
    # The structure constant for C_{i, l} is 1 if it depends on A_{i, j} B_{j, l}.
    # Actually, the easiest check is to test with random A, B.
    
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_true = A @ B
    
    # Compute C_decomp using the formula derived
    # C_decomp[i, l] = sum_j sum_r U[i, j, r] V[j, l, r] W[l, i, r]
    # Note: W is p x n, so W[l, i, r] is correct.
    
    # The einsum 'ijr,jlr,lir->il' computes exactly this.
    
    if not np.allclose(C_decomp, C_true, atol=1e-9):
        return False
    return True
