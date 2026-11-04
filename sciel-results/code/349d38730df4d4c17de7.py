import time
import numpy as np
import random
import itertools
from collections import defaultdict

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    total_terms = n * m * p
    
    # Precompute Strassen coefficients for a 2x2x2 block
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[1, 1] = 1; W_s[3, 1] = -1
    
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[2, 2] = 1; W_s[3, 2] = 1
    
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[1, 3] = 1
    
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[2, 4] = 1
    
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1
    
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1

    # Generate all possible 2x2x2 blocks
    blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                terms = frozenset((i0 + di, j0 + dj, k0 + dk) 
                                  for di in range(2) 
                                  for dj in range(2) 
                                  for dk in range(2))
                blocks.append((i0, j0, k0, terms))
    
    num_blocks_total = len(blocks)
    
    # Precompute conflict matrix
    conflict = [[False] * num_blocks_total for _ in range(num_blocks_total)]
    for i in range(num_blocks_total):
        for j in range(i + 1, num_blocks_total):
            if blocks[i][3] & blocks[j][3]:
                conflict[i][j] = True
                conflict[j][i] = True

    def greedy_init():
        indices = list(range(num_blocks_total))
        random.shuffle(indices)
        selected = []
        for idx in indices:
            if all(not conflict[idx][s] for s in selected):
                selected.append(idx)
        return selected

    def local_search(initial_set, max_iters=1000):
        current_set = set(initial_set)
        current_score = len(current_set)
        best_set = set(current_set)
        best_score = current_score
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            added = False
            candidates = [i for i in range(num_blocks_total) if i not in current_set]
            random.shuffle(candidates)
            for idx in candidates:
                if all(not conflict[idx][s] for s in current_set):
                    current_set.add(idx)
                    added = True
                    break
            
            if added:
                current_score += 1
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
                continue
            
            if current_set:
                remove_candidates = list(current_set)
                random.shuffle(remove_candidates)
                for rem_idx in remove_candidates:
                    temp_set = current_set - {rem_idx}
                    new_adds = []
                    remaining_candidates = [i for i in range(num_blocks_total) if i not in temp_set]
                    random.shuffle(remaining_candidates)
                    for idx in remaining_candidates:
                        if all(not conflict[idx][s] for s in temp_set):
                            temp_set.add(idx)
                            new_adds.append(idx)
                            
                    if len(new_adds) > 1:
                        current_set = temp_set
                        current_score += len(new_adds) - 1
                        if current_score > best_score:
                            best_score = current_score
                            best_set = set(current_set)
                        break
        
        return best_set

    best_blocks_indices = []
    best_count = 0
    
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        init_set = greedy_init()
        improved_set = local_search(init_set, max_iters=200)
        
        if len(improved_set) > best_count:
            best_count = len(improved_set)
            best_blocks_indices = list(improved_set)
            
            if best_count >= 13:
                break

    # Construct U, V, W
    num_blocks = len(best_blocks_indices)
    rank = total_terms - num_blocks
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    covered = set()
    
    for idx in best_blocks_indices:
        i0, j0, k0, terms = blocks[idx]
        
        for t in range(7):
            for li in range(2):
                for lj in range(2):
                    lu = li * 2 + lj
                    val = U_s[lu, t]
                    if val != 0:
                        gu = (i0 + li) * m + (j0 + lj)
                        U[gu, col] += val
                        
            for lj in range(2):
                for lk in range(2):
                    lv = lj * 2 + lk
                    val = V_s[lv, t]
                    if val != 0:
                        gv = (j0 + lj) * p + (k0 + lk)
                        V[gv, col] += val
                        
            for lk in range(2):
                for li in range(2):
                    lw = lk * 2 + li
                    val = W_s[lw, t]
                    if val != 0:
                        gw = (k0 + lk) * n + (i0 + li)
                        W[gw, col] += val
                        
            col += 1
        covered |= terms
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
                    
    if col < rank:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        rank = col
        
    # Rank-1 term fusion: merge terms that share identical U and V factor rows
    def fuse_terms(U, V, W):
        """Iteratively merge terms that share the same U row and V row."""
        U, V, W = U.copy(), V.copy(), W.copy()
        rank = U.shape[1]
        
        # Build hash maps for U rows and V rows
        def row_hash(row):
            return tuple(row)
        
        u_map = defaultdict(list)
        v_map = defaultdict(list)
        for c in range(rank):
            u_map[row_hash(U[:, c])].append(c)
            v_map[row_hash(V[:, c])].append(c)
        
        # Find pairs that share both U and V rows
        merged = False
        changed = True
        while changed and rank > 1:
            if time.time() > deadline - 5:
                break
            changed = False
            
            # Rebuild maps after each merge
            u_map = defaultdict(list)
            v_map = defaultdict(list)
            for c in range(rank):
                u_map[row_hash(U[:, c])].append(c)
                v_map[row_hash(V[:, c])].append(c)
            
            # Find all U rows with multiple terms
            for u_key, cols in u_map.items():
                if len(cols) < 2:
                    continue
                # For each pair in this U group, check if they share V row
                for a in range(len(cols)):
                    for b in range(a + 1, len(cols)):
                        ca, cb = cols[a], cols[b]
                        v_key_a = row_hash(V[:, ca])
                        v_key_b = row_hash(V[:, cb])
                        if v_key_a == v_key_b:
                            # Merge: combine W coefficients
                            W[:, ca] += W[:, cb]
                            # Remove column cb
                            U = np.delete(U, cb, axis=1)
                            V = np.delete(V, cb, axis=1)
                            W = np.delete(W, cb, axis=1)
                            rank -= 1
                            merged = True
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
            if changed:
                break  # Restart the while loop with updated rank
        
        return U, V, W, rank
    
    U, V, W, rank = fuse_terms(U, V, W)
    
    # Verify
    def verify(U, V, W):
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        
        T = np.zeros((n, m, p))
        for c in range(U.shape[1]):
            T += np.einsum('ij,jk,ki->ijk', U_r[:, :, c], V_r[:, :, c], W_r[:, :, c])
            
        return np.allclose(T, 1.0, atol=1e-9)

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        U_sb = np.zeros((n * m, total_terms), dtype=float)
        V_sb = np.zeros((m * p, total_terms), dtype=float)
        W_sb = np.zeros((p * n, total_terms), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1.0
                    V_sb[j * p + k, col] = 1.0
                    W_sb[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
