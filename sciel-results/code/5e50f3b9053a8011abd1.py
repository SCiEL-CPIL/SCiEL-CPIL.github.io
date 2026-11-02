import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s out of 150s budget
    
    n, m, p = DIMENSIONS
    
    def verify(U, V, W, rank):
        """Verify tensor decomposition correctness"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    if s != 1:
                        return False
        return True
    
    def build_decomposition(partition_m, partition_p):
        """
        Build decomposition given partitions of m and p dimensions.
        partition_m: list of sizes for m dimension (should sum to m)
        partition_p: list of sizes for p dimension (should sum to p)
        Returns (U, V, W, rank) or None if invalid
        """
        # n=2 is fixed, we handle 2-row outputs
        max_rank = 60
        U = np.zeros((n * m, max_rank), dtype=np.int64)
        V = np.zeros((m * p, max_rank), dtype=np.int64)
        W = np.zeros((p * n, max_rank), dtype=np.int64)
        col = 0
        
        # Process each pair of (m_block, p_block)
        # For each m_block of size ms and p_block of size ps,
        # we need to decompose 2 x ms x ps multiplication
        
        def add_block(i0, j0, k0, ms, ps):
            nonlocal col
            # For 2 x ms x ps block starting at row i0, col j0, output row k0
            # We'll use a hybrid approach:
            # If ms == 2 and ps == 2: Strassen (7)
            # If ms == 2 and ps == 1: schoolbook (4)  
            # If ms == 1 and ps == 2: schoolbook (4)
            # If ms == 2 and ps == 3: Strassen on 2+1 (7+4=11)
            # If ms == 2 and ps == 4: 2 Strassen (14)
            # General: split p into 2s and 1s
            
            # Simple approach: decompose p into blocks of size 2 and 1
            remaining_p = ps
            k_offset = 0
            
            # Try to use Strassen for 2x2x2 blocks
            while remaining_p >= 2:
                if ms == 2:
                    # Strassen block 2x2x2
                    _strassen_block(U, V, W, i0, j0, k0 + k_offset, col)
                    col += 7
                else:
                    # Schoolbook for 1x2x2 or general
                    for ii in range(ms):
                        for jj in range(ms):
                            # This is getting complex, use schoolbook
                            pass
                    # For ms > 2, use pure schoolbook
                    for ii in range(2):
                        for jj in range(ms):
                            for kk in range(2):
                                u_idx = (i0 + ii) * m + (j0 + jj)
                                v_idx = (j0 + jj) * p + (k0 + k_offset + kk)
                                w_idx = (k0 + k_offset + kk) * n + (i0 + ii)
                                U[u_idx, col] += 1
                                V[v_idx, col] += 1
                                W[w_idx, col] += 1
                                col += 1
                remaining_p -= 2
                k_offset += 2
            
            if remaining_p == 1:
                for ii in range(2):
                    for jj in range(ms):
                        u_idx = (i0 + ii) * m + (j0 + jj)
                        v_idx = (j0 + jj) * p + (k0 + k_offset)
                        w_idx = (k0 + k_offset) * n + (i0 + ii)
                        U[u_idx, col] += 1
                        V[v_idx, col] += 1
                        W[w_idx, col] += 1
                        col += 1
        
        def _strassen_block(U, V, W, i0, j0, k0, start_col):
            """Add Strassen 2x2x2 block at given position"""
            # i0, j0, k0 are the starting indices
            # A block: rows i0, i0+1; cols j0, j0+1; output rows k0, k0+1
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            
            e_idx = j0 * p + k0
            f_idx = j0 * p + (k0 + 1)
            g_idx = (j0 + 1) * p + k0
            h_idx = (j0 + 1) * p + (k0 + 1)
            
            c00_idx = k0 * n + i0
            c01_idx = (k0 + 1) * n + i0
            c10_idx = k0 * n + (i0 + 1)
            c11_idx = (k0 + 1) * n + (i0 + 1)
            
            r = start_col
            U[a_idx, r] += 1; U[d_idx, r] += 1
            V[e_idx, r] += 1; V[h_idx, r] += 1
            W[c00_idx, r] += 1; W[c11_idx, r] += 1
            r += 1
            
            U[c_idx, r] += 1; U[d_idx, r] += 1
            V[e_idx, r] += 1
            W[c10_idx, r] += 1; W[c11_idx, r] -= 1
            r += 1
            
            U[a_idx, r] += 1
            V[f_idx, r] += 1; V[h_idx, r] -= 1
            W[c01_idx, r] += 1; W[c11_idx, r] += 1
            r += 1
            
            U[d_idx, r] += 1
            V[g_idx, r] += 1; V[e_idx, r] -= 1
            W[c00_idx, r] += 1; W[c10_idx, r] += 1
            r += 1
            
            U[a_idx, r] += 1; U[b_idx, r] += 1
            V[h_idx, r] += 1
            W[c00_idx, r] -= 1; W[c01_idx, r] += 1
            r += 1
            
            U[c_idx, r] += 1; U[a_idx, r] -= 1
            V[e_idx, r] += 1; V[f_idx, r] += 1
            W[c11_idx, r] += 1
            r += 1
            
            U[b_idx, r] += 1; U[d_idx, r] -= 1
            V[g_idx, r] += 1; V[h_idx, r] += 1
            W[c00_idx, r] += 1
            r += 1
            
            return r
        
        # For n=2, we have 2 rows in output
        # Partition m into blocks, partition p into blocks
        j_positions = []
        pos = 0
        for sz in partition_m:
            j_positions.append((pos, sz))
            pos += sz
        
        k_positions = []
        pos = 0
        for sz in partition_p:
            k_positions.append((pos, sz))
            pos += sz
        
        # For each (j_block, k_block) pair, add decomposition
        # n=2 is handled within each block (both output rows)
        for (j0, ms) in j_positions:
            for (k0, ps) in k_positions:
                # Decompose 2 x ms x ps
                # Split ps into 2s and 1s
                ps_remaining = ps
                k_off = 0
                while ps_remaining >= 2:
                    if ms == 2:
                        new_col = _strassen_block(U, V, W, 0, j0, k0 + k_off, col)
                        col = new_col
                    else:
                        # Schoolbook for ms rows x 2 cols output
                        for ii in range(2):
                            for jj in range(ms):
                                for kk in range(2):
                                    u_idx = ii * m + (j0 + jj)
                                    v_idx = (j0 + jj) * p + (k0 + k_off + kk)
                                    w_idx = (k0 + k_off + kk) * n + ii
                                    U[u_idx, col] += 1
                                    V[v_idx, col] += 1
                                    W[w_idx, col] += 1
                                    col += 1
                    ps_remaining -= 2
                    k_off += 2
                
                if ps_remaining == 1:
                    for ii in range(2):
                        for jj in range(ms):
                            u_idx = ii * m + (j0 + jj)
                            v_idx = (j0 + jj) * p + (k0 + k_off)
                            w_idx = (k0 + k_off) * n + ii
                            U[u_idx, col] += 1
                            V[v_idx, col] += 1
                            W[w_idx, col] += 1
                            col += 1
        
        if col > max_rank:
            return None
        
        return U[:, :col].tolist(), V[:, :col].tolist(), W[:, :col].tolist(), col
    
    def generate_partitions(target, max_part_size=4):
        """Generate all partitions of target into parts of size <= max_part_size"""
        if target == 0:
            yield []
            return
        if target < 0:
            return
        for first in range(1, min(max_part_size, target) + 1):
            for rest in generate_partitions(target - first, max_part_size):
                yield [first] + rest
    
    # Search over partitions of m=4 and p=7
    best_rank = 50  # current best
    best_result = None
    
    # Generate candidate partitions
    m_partitions = list(generate_partitions(4, 3))
    p_partitions = list(generate_partitions(7, 3))
    
    # Deduplicate (order matters for some, but let's try all)
    seen_m = set()
    unique_m_parts = []
    for part in m_partitions:
        key = tuple(sorted(part))
        if key not in seen_m:
            seen_m.add(key)
            unique_m_parts.append(part)
    
    seen_p = set()
    unique_p_parts = []
    for part in p_partitions:
        key = tuple(sorted(part))
        if key not in seen_p:
            seen_p.add(key)
            unique_p_parts.append(part)
    
    # Try each combination
    for m_part in unique_m_parts:
        if time.time() > deadline - 20:
            break
        for p_part in unique_p_parts:
            if time.time() > deadline - 20:
                break
            
            # Estimate rank first
            est_rank = 0
            for ms in m_part:
                for ps in p_part:
                    if ms == 2 and ps >= 2:
                        pairs = ps // 2
                        est_rank += pairs * 7
                        if ps % 2 == 1:
                            est_rank += ms * 2  # 2 output rows * 1 col * ms... 
                            # Actually for ps odd: last block is size 1
                            est_rank += 2 * ms  # 2 rows output, ms cols input, 1 output col
                    elif ms == 2 and ps == 1:
                        est_rank += 2 * ms  # 4
                    else:
                        est_rank += 2 * ms * ps  # schoolbook: 2 output rows * ms * ps
            
            if est_rank >= best_rank:
                continue
            
            # Build and verify
            result = build_decomposition(m_part, p_part)
            if result is None:
                continue
            
            U, V, W, rank = result
            
            if rank >= best_rank:
                continue
            
            if verify(np.array(U), np.array(V), np.array(W), rank):
                best_rank = rank
                best_result = {"U": U, "V": V, "W": W}
                print(f"Found rank {rank} with m_part={m_part}, p_part={p_part}")
    
    if best_result is not None:
        return best_result
    
    # Fallback: use the known rank-50 decomposition
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    
    def add_strassen_block(i0, j0, k0, c):
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        U[a_idx, c] += 1; U[d_idx, c] += 1
        V[e_idx, c] += 1; V[h_idx, c] += 1
        W[c00_idx, c] += 1; W[c11_idx, c] += 1
        c += 1
        U[c_idx, c] += 1; U[d_idx, c] += 1
        V[e_idx, c] += 1
        W[c10_idx, c] += 1; W[c11_idx, c] -= 1
        c += 1
        U[a_idx, c] += 1
        V[f_idx, c] += 1; V[h_idx, c] -= 1
        W[c01_idx, c] += 1; W[c11_idx, c] += 1
        c += 1
        U[d_idx, c] += 1
        V[g_idx, c] += 1; V[e_idx, c] -= 1
        W[c00_idx, c] += 1; W[c10_idx, c] += 1
        c += 1
        U[a_idx, c] += 1; U[b_idx, c] += 1
        V[h_idx, c] += 1
        W[c00_idx, c] -= 1; W[c01_idx, c] += 1
        c += 1
        U[c_idx, c] += 1; U[a_idx, c] -= 1
        V[e_idx, c] += 1; V[f_idx, c] += 1
        W[c11_idx, c] += 1
        c += 1
        U[b_idx, c] += 1; U[d_idx, c] -= 1
        V[g_idx, c] += 1; V[h_idx, c] += 1
        W[c00_idx, c] += 1
        c += 1
        return c
    
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            col = add_strassen_block(0, j_start, k_start, col)
        # Schoolbook for k=6
        for ii in range(2):
            for jj in range(2):
                u_idx = ii * m + (j_start + jj)
                v_idx = (j_start + jj) * p + 6
                w_idx = 6 * n + ii
                U[u_idx, col] += 1
                V[v_idx, col] += 1
                W[w_idx, col] += 1
                col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    if not verify(U, V, W, rank):
        raise ValueError("Fallback decomposition failed verification")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
