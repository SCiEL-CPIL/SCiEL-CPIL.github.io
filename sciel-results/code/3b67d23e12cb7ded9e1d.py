"""Matrix multiplication 3x5x5 rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Safety margin: finish well before 120s
    deadline = time.time() + 110.0
    
    # Strassen pattern for 2x2x2 block multiplication (rank 7)
    # U: (2,2), V: (2,2), W: (2,2) entries are (i, j, val)
    strassen_222 = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    def get_compositions(total, max_parts=10):
        """Generate all ordered compositions of 'total'."""
        def gen(rem, curr, path):
            if rem == 0:
                path.append(tuple(curr))
                return
            for k in range(1, rem + 1):
                if len(curr) >= max_parts:
                    break
                curr.append(k)
                gen(rem - k, curr, path)
                curr.pop()
        comps = []
        gen(total, [], comps)
        return comps

    # Pre-compute rank cost for a block of size (a, b, c)
    # If a=2, b=2, c=2 -> rank 7
    # Otherwise -> rank a*b*c
    # Note: This simple cost model assumes independent optimal decomposition of blocks.
    # A more advanced model could try further splits, but for 3,5,5 depth 2 or 3 is sufficient.
    
    def block_rank(a, b, c):
        if a == 2 and b == 2 and c == 2:
            return 7
        return a * b * c

    def compute_total_rank(n_comp, m_comp, p_comp):
        r = 0
        for i_block in n_comp:
            for j_block in m_comp:
                for k_block in p_comp:
                    r += block_rank(i_block, j_block, k_block)
        return r

    def build_decomposition(n_comp, m_comp, p_comp):
        n_offsets = [0]
        for s in n_comp[:-1]:
            n_offsets.append(n_offsets[-1] + s)
        m_offsets = [0]
        for s in m_comp[:-1]:
            m_offsets.append(m_offsets[-1] + s)
        p_offsets = [0]
        for s in p_comp[:-1]:
            p_offsets.append(p_offsets[-1] + s)

        total_rank = compute_total_rank(n_comp, m_comp, p_comp)
        
        # Initialize matrices with zeros
        U = np.zeros((n * m, total_rank), dtype=np.int64)
        V = np.zeros((m * p, total_rank), dtype=np.int64)
        W = np.zeros((p * n, total_rank), dtype=np.int64)
        
        col = 0
        for i_idx, i_block in enumerate(n_comp):
            i_off = n_offsets[i_idx]
            for j_idx, j_block in enumerate(m_comp):
                j_off = m_offsets[j_idx]
                for k_idx, k_block in enumerate(p_comp):
                    k_off = p_offsets[k_idx]
                    
                    if i_block == 2 and j_block == 2 and k_block == 2:
                        # Apply Strassen
                        for u_entries, v_entries, w_entries in strassen_222:
                            # U updates
                            for i_loc, j_loc, val in u_entries:
                                gi = i_off + i_loc
                                gj = j_off + j_loc
                                U[gi * m + gj, col] += val
                                
                            # V updates
                            for j_loc, k_loc, val in v_entries:
                                gj = j_off + j_loc
                                gk = k_off + k_loc
                                V[gj * p + gk, col] += val
                                
                            # W updates
                            for k_loc, i_loc, val in w_entries:
                                gk = k_off + k_loc
                                gi = i_off + i_loc
                                W[gk * n + gi, col] += val
                            
                            col += 1
                    else:
                        # Naive multiplication for this block
                        for i_loc in range(i_block):
                            for j_loc in range(j_block):
                                for k_loc in range(k_block):
                                    gi = i_off + i_loc
                                    gj = j_off + j_loc
                                    gk = k_off + k_loc
                                    U[gi * m + gj, col] = 1
                                    V[gj * p + gk, col] = 1
                                    W[gk * n + gi, col] = 1
                                    col += 1

        assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
        return U, V, W, total_rank

    def verify(U, V, W, rank):
        # Reshape for tensor product: U[i,j,r], V[j,k,r], W[k,i,r]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # We check if T == 1 (identity tensor for mat mult)
        
        # Vectorized check
        # M_r = V3 * W3 permuted?
        # V3[j,k,r] * W3[k,i,r] is tricky to align directly without loops or einsum
        # np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        try:
            T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
        except Exception:
            return False
            
        # Target is identity matrix tensor: T[i,j,k] = 1 if k==j? No.
        # Matrix multiplication (A * B)_{i,k} = sum_j A_{i,j} B_{j,k}
        # The tensor M_{i,j,k} such that C_{i,k} = sum_j M_{i,j,k} A_{i,j} B_{j,k}
        # The standard matrix multiplication tensor has M_{i,j,k} = 1 for all i,j,k?
        # No, wait.
        # The tensor is T^{ijk}. The multiplication is T(A,B) = sum_{i,j,k} T^{ijk} A_{ij} B_{jk}.
        # For matrix multiplication, T^{ijk} = delta_{ij, jk? No.
        # (AB)_{ik} = sum_j A_{ij} B_{jk}.
        # This corresponds to T^{i, j, k} = 1 if the indices match the summation structure?
        # Let's look at the definition:
        # U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}
        # Sum over r.
        # We want sum_r U_{ij}^r V_{jk}^r W_{ki}^r = delta_{j,k}? No.
        # We want the resulting tensor T_{i,j,k} to satisfy:
        # (AB)_{i,k} = sum_j T_{i,j,k} A_{i,j} B_{j,k}
        # In standard matrix multiplication, (AB)_{i,k} = sum_j 1 * A_{i,j} B_{j,k}.
        # So T_{i,j,k} should be 1 for all i, j, k?
        # Let's verify with trivial rank n*m*p:
        # U_{i,j}^{(i,j,k)} = 1, V_{j,k}^{(i,j,k)} = 1, W_{k,i}^{(i,j,k)} = 1.
        # Sum_r T_{i,j,k}^{(r)} = sum_{r=(i,j,k)} 1*1*1 = 1.
        # Yes, the target tensor T_{i,j,k} is the all-ones tensor?
        # Wait, the indices in the product are:
        # U has indices (i,j), V has (j,k), W has (k,i).
        # The output matrix C has indices (i,k).
        # The tensor M has indices (i,j,k).
        # M_{i,j,k} * A_{i,j} * B_{j,k}.
        # Summing over i,j,k? No, summing over j for each i,k.
        # So sum_j M_{i,j,k} A_{i,j} B_{j,k} = C_{i,k}.
        # For this to equal sum_j A_{i,j} B_{j,k}, we need M_{i,j,k} = 1.
        # So yes, the target tensor is the all-ones tensor of shape (n, m, p).
        
        target = np.ones((n, m, p), dtype=np.int64)
        return np.array_equal(T, target)

    # Generate candidate compositions
    n_comps = get_compositions(n)
    m_comps = get_compositions(m)
    p_comps = get_compositions(p)
    
    # Estimate ranks and sort
    candidates = []
    for nc in n_comps:
        for mc in m_comps:
            for pc in p_comps:
                r = compute_total_rank(nc, mc, pc)
                candidates.append((r, nc, mc, pc))
                
    candidates.sort(key=lambda x: x[0])
    
    best_rank = float('inf')
    best_result = None
    
    # Fallback trivial result (always valid)
    U_fallback, V_fallback, W_fallback, rank_fallback = build_decomposition((n,), (m,), (p,))
    best_result = {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}
    best_rank = rank_fallback
    
    # Search for better ranks
    for r, nc, mc, pc in candidates:
        if time.time() > deadline:
            break
        if r >= best_rank:
            break
            
        try:
            U, V, W, actual_rank = build_decomposition(nc, mc, pc)
            if verify(U, V, W, actual_rank):
                if actual_rank < best_rank:
                    best_rank = actual_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            continue
            
    return best_result
# EVOLVE-BLOCK-END
